"""Plot the training records of 01_pilot_train.py (train_log.tsv, dev_log.tsv are the source; this only draws).

Panels: training loss (per step and moving average) with dev / training-subset teacher-forced loss,
gradient norm and learning rate, tag-row distance from the initial values, LoRA weight norm.
Output: results/pilot/train_log.png
Usage: uv run python scripts/train/02_plot_log.py
"""
import csv

import matplotlib
import numpy as np

from cv3lora.paths import REPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = REPO / "results/pilot"
WINDOW = 20


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main() -> None:
    t, d = read(RESULTS / "train_log.tsv"), read(RESULTS / "dev_log.tsv")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a = ax[0, 0]
    a.plot(t["step"], t["loss"], color="0.75", lw=0.8, label="train (batch)")
    ma = np.convolve(t["loss"], np.ones(WINDOW) / WINDOW, mode="valid")
    a.plot(t["step"][WINDOW - 1:], ma, color="C0", label=f"train ({WINDOW}-step mean)")
    a.plot(d["step"], d["dev_loss"], "o-", color="C3", label="dev 249 (teacher forcing)")
    a.plot(d["step"], d["train_subset_loss"], "s--", color="C2", label="train subset (teacher forcing)")
    a.set(xlabel="step", ylabel="CE (nats / token)", title="loss")
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.plot(t["step"], t["grad_norm"], color="C1", lw=0.8)
    a.set(xlabel="step", ylabel="grad norm (before clip)", title="gradient norm / learning rate")
    b = a.twinx()
    b.plot(t["step"], t["lr"], color="k", lw=0.8)
    b.set_ylabel("lr")
    a = ax[1, 0]
    a.plot(t["step"], t["tag0_dist"], label="<PHON_START>")
    a.plot(t["step"], t["tag1_dist"], label="<PHON_END>")
    a.set(xlabel="step", ylabel="L2 distance from initial row", title="tag rows")
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(t["step"], t["lora_norm"], color="C4")
    a.set(xlabel="step", ylabel="L2 norm of all LoRA weights", title="LoRA weights")
    fig.tight_layout()
    fig.savefig(RESULTS / "train_log.png", dpi=120)
    print(RESULTS / "train_log.png")


if __name__ == "__main__":
    main()
