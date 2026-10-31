"""Checkpoint selection by the rules of research plan §12 (fixed before the span loss was computed).

From utt_loss.tsv (02_span_loss.py): loss = sum of ce / sum of tokens over the dev utterances. For each
measure, the paired bootstrap over utterances (10,000 resamples, seed 20261031) gives the 95 % percentile
interval of (loss of a checkpoint - loss of the minimum); checkpoints whose interval contains 0 are
"indistinguishable from the minimum" (the minimum included). Rule: if the dev-loss minimum is in that set
for the span loss, it is adopted; otherwise the span loss is checked by the user (§12). Also reported: the
same sets for the full loss and the widened spans (±1 token is the rule, ±2 reference), and the validity
check (accent-changed tag minus correct tag on the span, with its bootstrap interval) for every step.

Output: results/ckpt_select/selection.tsv (one row per step and measure), validity.tsv, decision.json
Usage: uv run python scripts/select/03_select.py
"""
import csv
import json

import numpy as np

from cv3lora.paths import REPO

OUT = REPO / "results/ckpt_select"
N_BOOT, SEED = 10000, 20261031
MEASURES = {"full": "full", "span": "span", "span_w1": "span_w1", "span_w2": "span_w2"}


def main() -> None:
    with open(OUT / "utt_loss.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    steps = sorted({int(r["step"]) for r in rows})
    ids = sorted({r["id"] for r in rows})
    col = {i: k for k, i in enumerate(ids)}

    def mat(name):  # [steps, utterances]
        m = np.zeros((len(steps), len(ids)))
        for r in rows:
            m[steps.index(int(r["step"])), col[r["id"]]] = float(r[name])
        return m

    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    counts = np.stack([np.bincount(b, minlength=len(ids)) for b in boot])  # [N_BOOT, utterances]
    cand = [k for k, s in enumerate(steps) if s > 0]  # the 20 checkpoints; step 0 (base) is a reference

    out, sets = [], {}
    for m, pre in MEASURES.items():
        ce, n = mat(f"{pre}_ce"), mat(f"{pre}_n")
        loss = ce.sum(1) / n.sum(1)
        best = min(cand, key=lambda k: loss[k])
        bl = (counts @ ce.T) / (counts @ n.T)  # [N_BOOT, steps]: resampled loss of every step
        diff = bl - bl[:, [best]]
        lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
        sets[m] = [steps[k] for k in cand if lo[k] <= 0 <= hi[k]]
        for k, s in enumerate(steps):
            out.append({"measure": m, "step": s, "loss": f"{loss[k]:.5f}", "diff_from_min": f"{loss[k] - loss[best]:.5f}",
                        "ci_lo": f"{lo[k]:.5f}", "ci_hi": f"{hi[k]:.5f}", "is_min": int(k == best),
                        "indistinguishable": int(s in sets[m])})
    with open(OUT / "selection.tsv", "w") as f:
        w = csv.DictWriter(f, list(out[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    ce, wce, n = mat("span_ce"), mat("wrong_span_ce"), mat("span_n")
    d = (counts @ wce.T - counts @ ce.T) / (counts @ n.T)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    point = (wce.sum(1) - ce.sum(1)) / n.sum(1)
    worse = (wce > ce).mean(1)  # share of utterances where the changed tag gives the larger span loss
    val = [{"step": s, "wrong_minus_correct": f"{point[k]:.5f}", "ci_lo": f"{lo[k]:.5f}", "ci_hi": f"{hi[k]:.5f}",
            "share_utts_wrong_larger": f"{worse[k]:.3f}"} for k, s in enumerate(steps)]
    with open(OUT / "validity.tsv", "w") as f:
        w = csv.DictWriter(f, list(val[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(val)

    full_min = next(int(r["step"]) for r in out if r["measure"] == "full" and r["is_min"])
    span_min = next(int(r["step"]) for r in out if r["measure"] == "span" and r["is_min"])
    dec = {"dev_loss_min": full_min, "span_loss_min": span_min,
           "indistinguishable": sets, "earliest_indistinguishable": {m: min(v) for m, v in sets.items()},
           "dev_loss_min_in_span_set": full_min in sets["span"],
           "same_with_span_w1": full_min in sets["span_w1"], "same_with_span_w2": full_min in sets["span_w2"],
           "rule": "§12: adopt the dev-loss minimum if it is in the span-loss set; otherwise the user checks",
           "n_utts": len(ids), "n_boot": N_BOOT, "seed": SEED}
    (OUT / "decision.json").write_text(json.dumps(dec, indent=2) + "\n")
    print(json.dumps(dec, indent=2))
    sel = {s: val[k] for k, s in enumerate(steps)}
    print("validity at the dev-loss minimum:", sel[full_min])


if __name__ == "__main__":
    main()
