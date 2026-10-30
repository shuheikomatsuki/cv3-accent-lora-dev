"""Main LoRA training (research plan §18.1 step 8, settings §11 【確定】).

Base model + new LoRA (Q/K/V/O, r 16 / alpha 64 / dropout 0.05) + the two tag rows, on the whole training set
(excluded recordings dropped, JVS tokens trimmed). Batch 8, 20,000 steps, AdamW (0.9, 0.98, eps 1e-8, weight
decay 0), lr 1e-4 with 5 % linear warmup then cosine to 0, grad-norm clip 1.0, base fp32 + bf16 autocast.
No early stopping: every --save-every steps the adapter is saved and the teacher-forced loss of the dev set
(§12) and of a fixed training subset (overfitting check) is measured; the model is chosen afterwards (§12).
A step whose loss or gradient norm is not finite is skipped; --max-nonfinite-run in a row stops the run.

Resume: --resume continues from the last saved state (optimizer, RNG, data order), as if not interrupted.
Without --resume, an existing run is never overwritten.

Output: <work_root>/train/<run>/ckpt/stepNNNNN.pt (adapters, load with cv3lora.lora.load_adapter),
        <work_root>/train/<run>/state.pt (for --resume);
        results/train/<run>/{train_log.tsv (every step), dev_log.tsv (every save), train.json, train_subset_utts.txt}
        Plot at any time, also during training: uv run python scripts/train/02_plot_log.py --dir results/train/<run>
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/train/04_train.py [--run main] [--resume]
"""
import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from cv3lora.cosyvoice_loader import cosyvoice_commit, load_cosyvoice3
from cv3lora.lm_loss import PREFIX, sequence_logits, text_tokens
from cv3lora.lora import TAG_IDS, apply_lora, register_tags, save_adapter
from cv3lora.paths import REPO, load_paths
from cv3lora.train_data import load_set
from cv3lora.train_loop import batches, lr_at, tf_loss

TRAIN_COLS = ["step", "loss", "n_tok", "grad_norm", "lr", "nonfinite", "tag0_norm", "tag1_norm",
              "tag0_dist", "tag1_dist", "lora_norm", "sec"]
DEV_COLS = ["step", "dev_loss", "dev_acc", "train_subset_loss", "train_subset_acc"]
LORA = {"r": 16, "alpha": 64, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def keep_rows(path, cols, last_step) -> None:
    """On resume: drop log rows written after the saved state."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if int(r["step"]) <= last_step]
    with open(path, "w") as f:
        w = csv.DictWriter(f, cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def atomic_save(obj, path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="data order and torch (LoRA init, dropout)")
    ap.add_argument("--subset-n", type=int, default=250, help="fixed training utterances for the overfitting check")
    ap.add_argument("--subset-seed", type=int, default=20261030)
    ap.add_argument("--max-nonfinite-run", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    paths = load_paths()
    work = paths["work_root"] / "train" / args.run
    res = REPO / "results/train" / args.run
    state_path = work / "state.pt"
    if args.resume:
        assert state_path.exists(), f"nothing to resume: {state_path}"
    else:
        assert not state_path.exists() and not (res / "train_log.tsv").exists(), \
            f"run '{args.run}' exists; use --resume or another --run"
    (work / "ckpt").mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cv = load_cosyvoice3()
    lm, fe = cv.model.llm, cv.frontend
    del cv.model.flow, cv.model.hift
    torch.cuda.empty_cache()
    register_tags(fe.tokenizer)

    train, dev = load_set(paths, "train"), load_set(paths, "dev")
    for it in train + dev:
        it["tt"] = text_tokens(fe, it["text"])
    subset = random.Random(args.subset_seed).sample(train, args.subset_n)
    (res / "train_subset_utts.txt").write_text("".join(sorted(it["id"] + "\n" for it in subset)))

    apply_lora(lm, **LORA)
    frozen0 = {n: p.detach().cpu().clone() for n, p in lm.named_parameters() if not p.requires_grad}
    emb = lm.llm.model.model.embed_tokens
    tag0 = emb.tag_rows.detach().clone()  # the original rows 151924/151925 (also on resume)
    lora_params = [p for n, p in lm.named_parameters() if "lora_" in n]
    params = lora_params + [emb.tag_rows]
    opt = torch.optim.AdamW([{"params": lora_params, "weight_decay": 0.0},
                             {"params": [emb.tag_rows], "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    start, nonfinite, step_s = 0, 0, 0.0
    if args.resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        assert st["args"]["steps"] == args.steps and st["args"]["seed"] == args.seed and st["args"]["bs"] == args.bs
        named = dict(lm.named_parameters())
        with torch.no_grad():
            for n, v in st["trainable"].items():
                named[n].copy_(v)
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng_cpu"])
        torch.cuda.set_rng_state(st["rng_cuda"])
        start, nonfinite, step_s = st["step"], st["nonfinite"], st["step_s"]
        keep_rows(res / "train_log.tsv", TRAIN_COLS, start)
        keep_rows(res / "dev_log.tsv", DEV_COLS, start)
        print(f"resumed at step {start}", flush=True)

    meta = {"args": vars(args), "lora": LORA, "optimizer": "AdamW betas (0.9, 0.98) eps 1e-8, weight decay 0",
            "autocast": "bf16 (base weights fp32)", "prefix": PREFIX, "n_train": len(train),
            "n_train_tagged": sum(TAG_IDS[0] in it["tt"].tolist() for it in train), "n_dev": len(dev),
            "ckpt_dir": str(work / "ckpt"), "cosyvoice_commit": cosyvoice_commit(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name()}

    def write_meta(**kw):
        meta.update(kw, updated=datetime.now().isoformat(timespec="seconds"))
        (res / "train.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    started = datetime.now().isoformat(timespec="seconds")
    if args.resume:
        started = json.loads((res / "train.json").read_text()).get("started", started)
    write_meta(status="running", started=started)

    dev_f = open(res / "dev_log.tsv", "a" if args.resume else "w")
    dev_w = csv.DictWriter(dev_f, DEV_COLS, delimiter="\t")
    log = open(res / "train_log.tsv", "a" if args.resume else "w")
    w = csv.DictWriter(log, TRAIN_COLS, delimiter="\t")
    if not args.resume:
        dev_w.writeheader()
        w.writeheader()

    def evaluate(step):
        d, da = tf_loss(lm, dev)
        t, ta = tf_loss(lm, subset)
        dev_w.writerow({"step": step, "dev_loss": f"{d:.5f}", "dev_acc": f"{da:.5f}",
                        "train_subset_loss": f"{t:.5f}", "train_subset_acc": f"{ta:.5f}"})
        dev_f.flush()
        print(f"[eval] step {step} dev {d:.4f} train-subset {t:.4f}", flush=True)

    def save(step):
        save_adapter(work / "ckpt" / f"step{step:05d}.pt", lm, {"lora": LORA, "args": vars(args), "step": step})
        atomic_save({"step": step, "args": vars(args), "nonfinite": nonfinite, "step_s": step_s,
                     "trainable": {n: p.detach().cpu().clone() for n, p in lm.named_parameters() if p.requires_grad},
                     "opt": opt.state_dict(), "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()},
                    state_path)

    if start == 0:
        evaluate(0)  # LoRA B = 0 and tag rows = original rows: identical to the base model
    warmup = max(1, round(args.steps * args.warmup_ratio))
    stream = batches(train, args.bs, args.seed)
    for _ in range(start):  # same data order as an uninterrupted run
        next(stream)
    lm.train()
    torch.cuda.reset_peak_memory_stats()
    run = 0  # consecutive non-finite steps
    for step in range(start, args.steps):
        t0 = time.time()
        b = next(stream)
        lr = lr_at(step, args.steps, warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, y, _, n = sequence_logits(lm, [it["tt"] for it in b], [it["st"] for it in b])
        loss = F.cross_entropy(logits, y)  # mean over all target tokens of the batch
        opt.zero_grad(set_to_none=True)
        gn = float("nan")
        if torch.isfinite(loss):
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
        if math.isfinite(gn):
            opt.step()
            run = 0
        else:
            nonfinite, run = nonfinite + 1, run + 1
        torch.cuda.synchronize()
        step_s += time.time() - t0
        with torch.no_grad():
            tr = emb.tag_rows.detach()
            w.writerow({"step": step + 1, "loss": f"{loss.item():.5f}", "n_tok": int(n.sum()), "grad_norm": f"{gn:.4f}",
                        "lr": f"{lr:.3e}", "nonfinite": nonfinite,
                        "tag0_norm": f"{tr[0].norm():.5f}", "tag1_norm": f"{tr[1].norm():.5f}",
                        "tag0_dist": f"{(tr[0] - tag0[0]).norm():.5f}", "tag1_dist": f"{(tr[1] - tag0[1]).norm():.5f}",
                        "lora_norm": f"{torch.sqrt(sum((p.float() ** 2).sum() for p in lora_params)):.4f}",
                        "sec": f"{time.time() - t0:.3f}"})
        if (step + 1) % 10 == 0:
            log.flush()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1} loss {loss.item():.4f} gn {gn:.3f} lr {lr:.2e}", flush=True)
        if run >= args.max_nonfinite_run:
            log.flush()
            write_meta(status=f"stopped: {run} non-finite steps in a row at step {step + 1}")
            raise SystemExit(f"stopped: {run} non-finite steps in a row at step {step + 1}")
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            log.flush()
            evaluate(step + 1)
            save(step + 1)
            write_meta(status="running", last_saved_step=step + 1)
    log.close()
    dev_f.close()

    changed = [n for n, p in lm.named_parameters() if not p.requires_grad and not torch.equal(p.detach().cpu(), frozen0[n])]
    write_meta(status="done", peak_mem_gb_this_process=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               train_s_excluding_eval=round(step_s, 1), s_per_step=round(step_s / args.steps, 4),
               nonfinite_steps=nonfinite, frozen_params_changed=changed, n_frozen_params=len(frozen0))
    print("done", json.dumps({"nonfinite": nonfinite, "frozen_params_changed": changed}), flush=True)


if __name__ == "__main__":
    main()
