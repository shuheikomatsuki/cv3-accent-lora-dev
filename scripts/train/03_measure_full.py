"""Memory and step time on the full training set, for fixing the main training settings (§18.1 step 7, §11).

Same model, LoRA and optimizer as the pilot (scripts/train/01_pilot_train.py). Measures:
1. sequence lengths (text + speech tokens) of the full training set vs the pilot's 300 utterances;
2. worst-case memory: one training step on the 8 longest training utterances;
3. step time: --steps steps of random batches from the full set (lr follows a --total-step schedule);
4. one dev teacher-forced loss pass (249 utterances) and one adapter save.
The weights are thrown away. Nothing here is a training result.

Output: results/train_measure/{steps.tsv, measure.json}
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/train/03_measure_full.py
"""
import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from cv3lora.cosyvoice_loader import cosyvoice_commit, load_cosyvoice3
from cv3lora.lm_loss import sequence_logits, text_tokens
from cv3lora.lora import apply_lora, register_tags, save_adapter
from cv3lora.paths import REPO, load_paths
from cv3lora.train_data import load_set
from cv3lora.train_loop import batches, lr_at, tf_loss

RESULTS = REPO / "results/train_measure"


def pct(xs, ps=(0, 50, 90, 99, 100)) -> dict:
    s = sorted(xs)
    return {f"p{p}": s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))] for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--total", type=int, default=20000, help="schedule length the lr follows (UtterTune max_steps)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = load_paths()
    out = paths["work_root"] / "measure"
    out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cv = load_cosyvoice3()
    lm, fe = cv.model.llm, cv.frontend
    del cv.model.flow, cv.model.hift
    torch.cuda.empty_cache()
    register_tags(fe.tokenizer)

    t0 = time.time()
    train, dev = load_set(paths, "train"), load_set(paths, "dev")
    for it in train + dev:
        it["tt"] = text_tokens(fe, it["text"])
    load_s = time.time() - t0
    pos = lambda it: it["tt"].numel() + it["st"].numel() + 3  # noqa: E731
    pilot_ids = set((REPO / "results/pilot/train_utts.txt").read_text().split())
    lengths = {"full_train": pct([pos(it) for it in train]),
               "pilot_300": pct([pos(it) for it in train if it["id"] in pilot_ids]),
               "full_train_mean": statistics.mean(pos(it) for it in train),
               "pilot_300_mean": statistics.mean(pos(it) for it in train if it["id"] in pilot_ids)}

    apply_lora(lm, r=16, alpha=64, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "o_proj"])
    emb = lm.llm.model.model.embed_tokens
    lora_params = [p for n, p in lm.named_parameters() if "lora_" in n]
    params = lora_params + [emb.tag_rows]
    opt = torch.optim.AdamW([{"params": lora_params, "weight_decay": 0.0},
                             {"params": [emb.tag_rows], "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    lm.train()

    def step(b, lr):
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, y, _, n = sequence_logits(lm, [it["tt"] for it in b], [it["st"] for it in b])
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.clip).item()
        if math.isfinite(gn):
            opt.step()
        torch.cuda.synchronize()
        return loss.item(), gn, int(n.sum())

    longest = sorted(train, key=pos)[-args.bs:]
    torch.cuda.reset_peak_memory_stats()
    step(longest, 0.0)
    worst = {"peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
             "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
             "positions": [pos(it) for it in longest]}

    warmup = max(1, round(args.total * 0.05))
    torch.cuda.reset_peak_memory_stats()
    stream, rows = batches(train, args.bs, args.seed), []
    with open(RESULTS / "steps.tsv", "w") as f:
        w = csv.DictWriter(f, ["step", "loss", "grad_norm", "n_tok", "max_pos", "sec"], delimiter="\t")
        w.writeheader()
        for s in range(args.steps):
            b = next(stream)
            t = time.time()
            loss, gn, ntok = step(b, lr_at(s, args.total, warmup, args.lr))
            r = {"step": s + 1, "loss": f"{loss:.5f}", "grad_norm": f"{gn:.4f}", "n_tok": ntok,
                 "max_pos": max(pos(it) for it in b), "sec": f"{time.time() - t:.4f}"}
            w.writerow(r)
            rows.append(r)
            if (s + 1) % 50 == 0:
                print(f"step {s + 1} loss {loss:.4f} gn {gn:.3f} {r['sec']}s", flush=True)
    random_peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    t = time.time()
    dev_loss, _ = tf_loss(lm, dev)
    dev_s = time.time() - t
    t = time.time()
    save_adapter(out / "adapter.pt", lm, {"note": "measurement only, not a training result"})
    save_s = time.time() - t

    secs = [float(r["sec"]) for r in rows[10:]]  # first 10 steps: warm-up of kernels
    gns = [float(r["grad_norm"]) for r in rows]
    s_step = statistics.mean(secs)
    (RESULTS / "measure.json").write_text(json.dumps({
        "args": vars(args), "n_train": len(train), "n_dev": len(dev), "data_load_s": round(load_s, 1),
        "lengths_positions": lengths, "worst_case_step": worst, "random_steps_peak_mem_gb": random_peak,
        "s_per_step_mean": round(s_step, 4), "s_per_step_median": round(statistics.median(secs), 4),
        "s_per_step_p90": pct(secs, (90,))["p90"],
        "grad_norm": {"min": min(gns), "median": statistics.median(gns), "max": max(gns),
                      "share_clipped": sum(g > args.clip for g in gns) / len(gns)},
        "dev_tf_loss_s": round(dev_s, 1), "dev_loss_after": round(dev_loss, 4),
        "adapter_save_s": round(save_s, 2), "adapter_mb": round((out / "adapter.pt").stat().st_size / 2**20, 1),
        "steps_per_epoch": len(train) // args.bs,
        "estimate_20000_steps_min": {"train": round(20000 * s_step / 60, 1),
                                     "dev_loss_every_1000": round(20 * dev_s / 60, 1)},
        "cosyvoice_commit": cosyvoice_commit(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "date": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
