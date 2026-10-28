"""Per-speaker loss vs speaking time on parallel100 with the base model (research plan §15.3).

All JVS parallel100 manifest rows without exclude.tsv; all speakers read the same sentences, so total
speaking time stands for speaking rate. Text = plain transcript after transcript_fixes.tsv (no tags:
the stock model has no tag tokens). JVS tokens trimmed of leading/trailing silence as in training;
speaking time = trimmed tokens / 25 Hz. Value per speaker = token-weighted NLL per speech token.

Output: results/speaker_diag/rate_base.tsv (speaker, group, n_utt, nll, seconds) and rate_base.md
        (Pearson and Spearman correlation over all speakers and over training speakers).
Usage: uv run python scripts/analysis/speaker_rate_corr.py
"""
import json
import time
from collections import defaultdict

import numpy as np
import torch

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.lm_loss import length_batches, sequence_loss, text_tokens
from cv3lora.manifest import load_excluded, load_manifest, original_texts
from cv3lora.paths import REPO, load_paths
from cv3lora.trim import TOKEN_HZ, load_token_spans

OUT = REPO / "results/speaker_diag"


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def corr(x: list[float], y: list[float]) -> tuple[float, float]:
    x, y = np.array(x), np.array(y)
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main() -> None:
    paths = load_paths()
    excluded = load_excluded()
    rows = [r for r in load_manifest() if r["subset"] == "parallel100" and r["id"] not in excluded]
    text = original_texts(paths, rows)
    spans = load_token_spans()

    cv = load_cosyvoice3()
    lm, fe = cv.model.llm, cv.frontend
    del cv.model.flow, cv.model.hift
    torch.cuda.empty_cache()
    lm.eval()

    items = []
    for r in rows:
        st = torch.load(paths["token_root"] / "jvs_parallel100" / f"{r['id']}.pt")
        a, b = spans[r["id"]]
        items.append({"speaker": r["speaker"], "tt": text_tokens(fe, text[r["id"]]), "st": st[a:b]})
    t0 = time.time()
    nll, n, sec, n_utt = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
    with torch.inference_mode():
        for batch in length_batches(items, lambda it: it["tt"].numel() + it["st"].numel() + 3, 12000):
            ce, k, _ = sequence_loss(lm, [it["tt"] for it in batch], [it["st"] for it in batch])
            for it, c, m in zip(batch, ce.tolist(), k.tolist()):
                s = it["speaker"]
                nll[s] += c
                n[s] += m
                sec[s] += it["st"].numel() / TOKEN_HZ
                n_utt[s] += 1

    split = json.loads((REPO / "data/splits/split.json").read_text())
    group = {s[:6]: g for g, ss in split["speakers"].items() for s in ss}  # "jvs029(F)" -> jvs029
    spk = sorted(nll)
    lines = ["speaker\tgroup\tn_utt\tnll\tseconds"]
    lines += [f"{s}\t{group[s]}\t{n_utt[s]}\t{nll[s] / n[s]:.5f}\t{sec[s]:.1f}" for s in spk]
    (OUT / "rate_base.tsv").write_text("\n".join(lines) + "\n")

    md = ["# Per-speaker NLL vs speaking time (parallel100, base)", "",
          f"- utterances {len(items)}, speakers {len(spk)}; JVS tokens trimmed (§9)", "",
          "| speakers | n | Pearson | Spearman |", "|---|---:|---:|---:|"]
    for name, ss in [("all", spk), ("train", [s for s in spk if group[s] == "train"])]:
        p, q = corr([nll[s] / n[s] for s in ss], [sec[s] for s in ss])
        md.append(f"| {name} | {len(ss)} | {p:+.2f} | {q:+.2f} |")
    (OUT / "rate_base.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
