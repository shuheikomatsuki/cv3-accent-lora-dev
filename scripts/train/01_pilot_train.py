"""Pilot LoRA training on a few hundred utterances (research plan §18.1 step 6, settings §11 provisional).

Checks of items 1-2: prefix + tagged input, tag tokens, finite / decreasing loss, teacher-forced loss of the
training subset before vs after, frozen weights unchanged (bit-exact), reload gives the same logits.
Settings follow UtterTune configs/train/jsutjvs.yaml where it has one (batch 8, lr 1e-4, AdamW
0.9/0.98, warmup 5 % + cosine, grad-norm clip 1.0, weight decay 0); all provisional.

Output: <work_root>/pilot/adapter.pt; results/pilot/{train_log,dev_log}.tsv, train.json (settings, checks).
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/train/01_pilot_train.py
"""
import argparse
import csv
import json
import math
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from cv3lora.cosyvoice_loader import cosyvoice_commit, load_cosyvoice3
from cv3lora.lm_loss import PREFIX, sequence_logits, text_tokens
from cv3lora.lora import TAG_IDS, apply_lora, load_adapter, register_tags, save_adapter
from cv3lora.paths import REPO, load_paths
from cv3lora.train_loop import batches, lr_at, tf_loss
from cv3lora.train_data import load_set

RESULTS = REPO / "results/pilot"
TRAIN_COLS = ["step", "loss", "n_tok", "grad_norm", "lr", "nonfinite", "tag0_norm", "tag1_norm",
              "tag0_dist", "tag1_dist", "lora_norm", "sec"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utts", type=int, default=300)
    ap.add_argument("--select-seed", type=int, default=20261027)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = load_paths()
    out = paths["work_root"] / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cv = load_cosyvoice3()
    lm, fe = cv.model.llm, cv.frontend
    del cv.model.flow, cv.model.hift
    torch.cuda.empty_cache()
    register_tags(fe.tokenizer)
    checks = {}

    # item 1: tokenization of prefix + tagged text
    probe = "今日は<PHON_START>ア'メ<PHON_END>が降る。"
    ids = text_tokens(fe, probe).tolist()
    plain_prefix = text_tokens(fe, "").tolist()
    checks["tag_tokens"] = {"probe": probe, "ids": ids,
                            "prefix_ids_unchanged": ids[: len(plain_prefix)] == plain_prefix,
                            "tags_single_tokens": ids.count(TAG_IDS[0]) == 1 and ids.count(TAG_IDS[1]) == 1}

    all_train = load_set(paths, "train")
    train = random.Random(args.select_seed).sample(all_train, args.n_utts)
    dev = load_set(paths, "dev")
    for it in train + dev:
        it["tt"] = text_tokens(fe, it["text"])
    n_tagged = sum(TAG_IDS[0] in it["tt"].tolist() for it in train)
    (RESULTS / "train_utts.txt").write_text("".join(sorted(it["id"] + "\n" for it in train)))

    lora_cfg = {"r": 16, "alpha": 64, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    apply_lora(lm, **lora_cfg)
    frozen0 = {n: p.detach().cpu().clone() for n, p in lm.named_parameters() if not p.requires_grad}
    emb = lm.llm.model.model.embed_tokens
    tag0 = emb.tag_rows.detach().clone()
    lora_params = [p for n, p in lm.named_parameters() if "lora_" in n]
    opt = torch.optim.AdamW([{"params": lora_params, "weight_decay": 0.0},
                             {"params": [emb.tag_rows], "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    dev_rows = []

    def evaluate(step):
        d, da = tf_loss(lm, dev)
        t, ta = tf_loss(lm, train)
        dev_rows.append({"step": step, "dev_loss": d, "dev_acc": da, "train_subset_loss": t, "train_subset_acc": ta})
        print(f"[eval] step {step} dev {d:.4f} train-subset {t:.4f}", flush=True)

    evaluate(0)  # LoRA B = 0 and tag rows = original rows: identical to the base model
    log = open(RESULTS / "train_log.tsv", "w")
    w = csv.DictWriter(log, TRAIN_COLS, delimiter="\t")
    w.writeheader()
    warmup = max(1, round(args.steps * args.warmup_ratio))
    lm.train()
    torch.cuda.reset_peak_memory_stats()
    stream, nonfinite, step_s = batches(train, args.bs, args.seed), 0, 0.0
    for step in range(args.steps):
        t0 = time.time()
        b = next(stream)
        lr = lr_at(step, args.steps, warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, y, _, n = sequence_logits(lm, [it["tt"] for it in b], [it["st"] for it in b])
        loss = F.cross_entropy(logits, y)  # mean over all target tokens of the batch
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            nonfinite += 1
            gn = float("nan")
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(lora_params + [emb.tag_rows], args.clip).item()
            if math.isfinite(gn):
                opt.step()
            else:
                nonfinite += 1
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
            print(f"step {step + 1} loss {loss.item():.4f} gn {gn:.3f} lr {lr:.2e}", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate(step + 1)
    log.close()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    with open(RESULTS / "dev_log.tsv", "w") as f:
        dw = csv.DictWriter(f, list(dev_rows[0]), delimiter="\t")
        dw.writeheader()
        dw.writerows({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()} for r in dev_rows)

    # item 2: frozen weights unchanged, tag rows changed, effective embedding outside the tags unchanged
    lm.eval()
    changed = [n for n, p in lm.named_parameters() if not p.requires_grad and not torch.equal(p.detach().cpu(), frozen0[n])]
    with torch.no_grad():
        allids = torch.arange(emb.base.weight.shape[0], device=emb.base.weight.device)
        eff = emb(allids).cpu()
        base_w = frozen0["llm.model.model.embed_tokens.base.weight"]
        keep = torch.ones(len(allids), dtype=torch.bool)
        keep[list(TAG_IDS)] = False
        checks["weights"] = {
            "frozen_params_changed": changed, "n_frozen_params": len(frozen0),
            "other_rows_max_abs_diff": (eff[keep] - base_w[keep]).abs().max().item(),
            "n_other_rows": int(keep.sum()),
            "tag_rows_dist_from_init": (emb.tag_rows.detach() - tag0).norm(dim=1).tolist(),
        }
    save_adapter(out / "adapter.pt", lm, {"lora": lora_cfg, "args": vars(args)})

    # reload: a fresh model + saved adapter gives the same logits on dev batches
    ref_items = dev[:16]
    with torch.inference_mode():
        lg_mem, *_ = sequence_logits(lm, [it["tt"] for it in ref_items], [it["st"] for it in ref_items])
    lg_mem = lg_mem.cpu()
    del cv, lm, emb, opt
    torch.cuda.empty_cache()
    cv2 = load_cosyvoice3()
    del cv2.model.flow, cv2.model.hift
    load_adapter(cv2.model.llm, out / "adapter.pt")
    cv2.model.llm.eval()
    with torch.inference_mode():
        lg_new, *_ = sequence_logits(cv2.model.llm, [it["tt"] for it in ref_items], [it["st"] for it in ref_items])
    checks["reload_max_abs_logit_diff"] = (lg_new.cpu() - lg_mem).abs().max().item()

    tl = [float(r["loss"]) for r in csv.DictReader(open(RESULTS / "train_log.tsv"), delimiter="\t")]
    checks["loss"] = {"all_finite": nonfinite == 0 and all(math.isfinite(x) for x in tl),
                      "first20_mean": sum(tl[:20]) / 20, "last20_mean": sum(tl[-20:]) / 20}
    (RESULTS / "train.json").write_text(json.dumps({
        "args": vars(args), "lora": lora_cfg, "optimizer": "AdamW betas (0.9, 0.98) eps 1e-8, weight decay 0",
        "schedule": f"linear warmup {warmup} steps, then cosine to 0", "autocast": "bf16 (base weights fp32)",
        "n_train_pool": len(all_train), "n_train_used": len(train), "n_train_tagged": n_tagged, "n_dev": len(dev),
        "prefix": PREFIX, "adapter": str(out / "adapter.pt"), "peak_mem_gb": round(peak_gb, 2),
        "train_s_excluding_eval": round(step_s, 1), "s_per_step": round(step_s / args.steps, 3), "checks": checks,
        "cosyvoice_commit": cosyvoice_commit(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(checks, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
