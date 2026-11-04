"""CER on easy_50 (kana-whisper, JA dev references) per condition, next to UtterTune's Table 1.

Conditions: C (CV3+LoRA, tagged), A (CV3 base, plain), CV2A (CV2 base, plain), UT (CV2+UtterTune LoRA, tagged); all
cross_lingual. The 2x2 is read on LLM seed 0 (C also has seed 1, reported separately as "C, seeds 0+1").
Per file CER against results/easy50/inputs.tsv ref_kana, normalization of scripts/devgen/07_judge.py
(cv3lora.eval_audio.canon, known long-vowel bug kept for comparability with the dev numbers).
  capped    per-file CER capped at 1.0, no exclusion (this study's rule)
  excl>0.5  files with CER > 0.5 dropped (UtterTune's rule); kept share reported
Target reading accuracy and CER outside the target use the tag span (also defined for the plain-text conditions,
whose reference reading is the same).

Output: results/easy50/summary.json and cer_<cond>.tsv (_smoke with --smoke), printed table
Usage (eval env, CPU): PYTHONPATH=src uv run --project envs/eval python scripts/easy50/04_summary.py --cond C A CV2A UT
"""
import argparse
import csv
import json

import numpy as np

from cv3lora.eval_audio import align_errors, canon
from cv3lora.paths import REPO

OUT = REPO / "results/easy50"
UTTERTUNE = {"CosyVoice 2": 0.0557, "UtterTune": 0.0639}  # arXiv 2508.09767 Table 1, Test Set 1


def per_file(cond: str, inp: dict, sfx: str) -> list[dict]:
    with open(OUT / f"asr_{cond}{sfx}.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    per = []
    for a in rows:
        it = inp[a["id"]]
        e = align_errors(canon(it["ref_kana"]), canon(a["kw"]), (int(it["tgt_start"]), int(it["tgt_end"])))
        per.append({"cond": cond, "id": a["id"], "seed": a["seed"], "ref_speaker": a["ref_speaker"], "cer": e["cer"],
                    "cer_outside": e["cer_outside"], "tgt_ok": float(e["target_ok"]), "hyp": canon(a["kw"])})
    with open(OUT / f"cer_{cond}{sfx}.tsv", "w") as f:
        w = csv.DictWriter(f, list(per[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(per)
    return per


def agg(per: list[dict]) -> dict:
    v = np.array([p["cer"] for p in per])
    kept = v[v <= 0.5]
    return {"n": len(v), "cer_capped": float(np.minimum(v, 1).mean()), "cer_median": float(np.median(v)),
            "kept_share": float(len(kept) / len(v)), "cer_excl>0.5": float(kept.mean()) if len(kept) else None,
            "cer_outside_capped": float(np.mean([min(p["cer_outside"], 1) for p in per])),
            "target_ok": float(np.mean([p["tgt_ok"] for p in per]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=["C"], choices=["C", "A", "CV2A", "UT"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    with open(OUT / "inputs.tsv") as f:
        inp = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    res = {}
    for cond in args.cond:
        per = per_file(cond, inp, sfx)
        res[cond] = agg([p for p in per if p["seed"] == "0"])
        if cond == "C" and any(p["seed"] == "1" for p in per):
            res["C, seeds 0+1"] = agg(per)
    res["uttertune_table1"] = UTTERTUNE
    with open(OUT / f"summary{sfx}.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("cond          n    capped  median  kept  excl>0.5  outside  target_ok")
    for k, a in res.items():
        if k == "uttertune_table1":
            continue
        ex = f"{a['cer_excl>0.5']:.3f}" if a["cer_excl>0.5"] is not None else "-"
        print(f"{k:12s} {a['n']:4d}  {a['cer_capped']:.3f}   {a['cer_median']:.3f}  {a['kept_share']:.2f}  {ex:8s}  "
              f"{a['cer_outside_capped']:.3f}    {a['target_ok']:.2f}")


if __name__ == "__main__":
    main()
