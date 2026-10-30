"""F0 drop from the target word into ダ for the accent pair sentences (research plan §13.4, step 6 item 4).

Reads the wave tables of 02_token2wav.py (pairs_plain = condition A, pairs_tag = condition C) and measures,
per wave, the F0 change from the last mora of the word to ダ (accent_f0.mora_drop). Strongly negative =
falls before ダ (尾高, or 頭高 for a 2-mora word); about 0 or positive = flat. A listening aid, not the
accent judgment (§15.2 undecided). In step 6 this checks that the pipeline runs end to end and records the
base model's default pattern; whether the LoRA separates flat and final is judged after the main training.

Output: results/pilot/accent_pairs.tsv (one row per wave), accent_pairs_summary.tsv (per input form)
Usage: uv run python scripts/eval/01_accent_pairs.py --names pairs_base pairs_pilot
"""
import argparse
import csv

import numpy as np
import soundfile as sf

from cv3lora.accent_f0 import load_ctc, mora_drop
from cv3lora.paths import REPO

RESULTS = REPO / "results/pilot"
PRE, POST = "コレワ", "トオモイマス"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    args = ap.parse_args()
    model, vocab = load_ctc()
    rows = []
    for name in args.names:
        with open(RESULTS / f"wav_{name}.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                x, sr = sf.read(r["wav"], dtype="float32")
                out = {k: r[k] for k in ("key", "seed", "kana", "form", "model", "ref_lang", "ref_gender", "ref_speaker")}
                try:
                    m = mora_drop(model, vocab, x, sr, PRE, r["kana"], "ダ", POST)
                    out.update(status="ok", drop_st=f"{m['drop']:.2f}", mora_f0=" ".join(f"{v:.2f}" for v in m["mora_f0"]))
                except Exception as e:  # alignment failed (e.g. a different word was read)
                    out.update(status=f"fail:{type(e).__name__}", drop_st="", mora_f0="")
                rows.append(out)
                print(out["key"], out["seed"], out["ref_speaker"], out["status"], out["drop_st"], flush=True)
    with open(RESULTS / "accent_pairs.tsv", "w") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    summ = []
    for key in dict.fromkeys(r["key"] for r in rows):
        g = [r for r in rows if r["key"] == key]
        d = np.array([float(r["drop_st"]) for r in g if r["drop_st"] not in ("", "nan")])
        summ.append({"key": key, "model": g[0]["model"], "kana": g[0]["kana"], "form": g[0]["form"], "n": len(g),
                     "n_measured": len(d), "median_drop_st": f"{np.median(d):.2f}" if len(d) else "",
                     "q25": f"{np.percentile(d, 25):.2f}" if len(d) else "",
                     "q75": f"{np.percentile(d, 75):.2f}" if len(d) else ""})
    with open(RESULTS / "accent_pairs_summary.tsv", "w") as f:
        w = csv.DictWriter(f, list(summ[0]), delimiter="\t")
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
