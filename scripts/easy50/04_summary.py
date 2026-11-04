"""CER of CV3+LoRA cross_lingual on easy_50 (kana-whisper, JA dev references), next to UtterTune's Table 1.

Per file CER against results/easy50/inputs.tsv ref_kana, normalization of scripts/devgen/07_judge.py
(cv3lora.eval_audio.canon, known long-vowel bug kept for comparability with the dev numbers).
  capped    per-file CER capped at 1.0, no exclusion (this study's rule)
  excl>0.5  files with CER > 0.5 dropped (UtterTune's rule); kept share reported
Also target reading accuracy (no edit inside the target span) and CER outside the target.

Output: results/easy50/summary.json (_smoke with --smoke), printed table
Usage (eval env, CPU): PYTHONPATH=src uv run --project envs/eval python scripts/easy50/04_summary.py [--smoke]
"""
import argparse
import csv
import json

import numpy as np

from cv3lora.eval_audio import align_errors, canon
from cv3lora.paths import REPO

OUT = REPO / "results/easy50"
UTTERTUNE = {"CosyVoice 2": 0.0557, "UtterTune": 0.0639}  # arXiv 2508.09767 Table 1, Test Set 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    with open(OUT / "inputs.tsv") as f:
        inp = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    with open(OUT / f"asr_C{sfx}.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    per = []
    for a in rows:
        it = inp[a["id"]]
        e = align_errors(canon(it["ref_kana"]), canon(a["kw"]), (int(it["tgt_start"]), int(it["tgt_end"])))
        per.append({"id": a["id"], "seed": a["seed"], "ref_speaker": a["ref_speaker"], "cer": e["cer"],
                    "cer_outside": e["cer_outside"], "tgt_ok": float(e["target_ok"]), "hyp": canon(a["kw"])})
    v = np.array([p["cer"] for p in per])
    kept = v[v <= 0.5]
    res = {"n": len(v), "cer_capped": float(np.minimum(v, 1).mean()), "cer_median": float(np.median(v)),
           "kept_share": float(len(kept) / len(v)), "cer_excl>0.5": float(kept.mean()) if len(kept) else None,
           "cer_outside_capped": float(np.mean([min(p["cer_outside"], 1) for p in per])),
           "target_ok": float(np.mean([p["tgt_ok"] for p in per])), "uttertune_table1": UTTERTUNE}
    with open(OUT / f"summary{sfx}.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(OUT / f"cer_C{sfx}.tsv", "w") as f:
        w = csv.DictWriter(f, list(per[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(per)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
