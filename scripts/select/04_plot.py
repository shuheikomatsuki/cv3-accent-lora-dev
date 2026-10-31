"""Per-step table and figure of the selection losses (research plan §12; reads 03_select.py's output).

Left: full dev loss and target-span loss of each checkpoint as the difference from its own minimum, with the
bootstrap 95 % interval of that difference (step 0 = base model is left out: 0.9 above the minimum).
Right: accent-changed tag minus correct tag on the span (validity check), with its 95 % interval.

Output: results/ckpt_select/steps.tsv (one row per step), losses.png
Usage: uv run python scripts/select/04_plot.py
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cv3lora.paths import REPO

OUT = REPO / "results/ckpt_select"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#888888"


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    sel = {(r["measure"], int(r["step"])): r for r in read(OUT / "selection.tsv")}
    val = {int(r["step"]): r for r in read(OUT / "validity.tsv")}
    dev_log = {int(r["step"]): r for r in read(REPO / "results/train/main/dev_log.tsv")}
    steps = sorted(val)
    rows = [{"step": s, "dev_loss": sel["full", s]["loss"], "dev_ci": f"{sel['full', s]['ci_lo']}..{sel['full', s]['ci_hi']}",
             "span_loss": sel["span", s]["loss"], "span_ci": f"{sel['span', s]['ci_lo']}..{sel['span', s]['ci_hi']}",
             "train_subset_loss": dev_log[s]["train_subset_loss"],
             "wrong_minus_correct": val[s]["wrong_minus_correct"]} for s in steps]
    with open(OUT / "steps.tsv", "w") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    ck = [s for s in steps if s > 0]
    x = [s / 1000 for s in ck]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for m, color, name in (("full", BLUE, "dev loss (whole utterance)"), ("span", ORANGE, "target-span loss")):
        y = [float(sel[m, s]["diff_from_min"]) for s in ck]
        lo = [y[i] - float(sel[m, s]["ci_lo"]) for i, s in enumerate(ck)]
        hi = [float(sel[m, s]["ci_hi"]) - y[i] for i, s in enumerate(ck)]
        a.errorbar(x, y, yerr=[lo, hi], color=color, lw=2, marker="o", ms=5, capsize=2, elinewidth=1, label=name)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_title("difference from each measure's minimum (95% bootstrap interval)", fontsize=10, color=INK)
    a.set_xlabel("step (x1000)")
    a.set_ylabel("nats / token")
    a.set_xticks(range(0, 21, 2))
    a.legend(frameon=False, fontsize=9)
    y = [float(val[s]["wrong_minus_correct"]) for s in steps]
    lo = [y[i] - float(val[s]["ci_lo"]) for i, s in enumerate(steps)]
    hi = [float(val[s]["ci_hi"]) - y[i] for i, s in enumerate(steps)]
    b.errorbar([s / 1000 for s in steps], y, yerr=[lo, hi], color=BLUE, lw=2, marker="o", ms=5, capsize=2, elinewidth=1)
    b.axhline(0, color=MUTED, lw=0.8)
    b.annotate("base model", (0, y[0]), xytext=(1.2, 0.02), fontsize=9, color=INK)
    b.set_title("span loss: accent-changed tag minus correct tag", fontsize=10, color=INK)
    b.set_xlabel("step (x1000)")
    b.set_ylabel("nats / token")
    b.set_xticks(range(0, 21, 2))
    for ax in (a, b):
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "losses.png", dpi=130)
    for r in rows:
        print("\t".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
