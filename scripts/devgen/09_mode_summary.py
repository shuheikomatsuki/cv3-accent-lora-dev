"""Condition A in cross_lingual vs zero_shot (Az, 08_zero_shot.py): kana CER of the whole sentence.

Side check for the gap to UtterTune's CER; not part of the §12 judgement. Same files on both sides:
JA dev references (4), LLM seed 0, both recognizers of 05_asr.py, references and normalization of 07_judge.py
(cv3lora.eval_audio.canon, with its known long-vowel bug; it hits both sides alike). Also R (the recordings).

Two aggregations, per file then mean:
  capped     CER capped at 1.0 per file, no exclusion (this study's rule)
  excl>0.5   files with CER > 0.5 dropped (close to UtterTune's rule); kept share reported
Paired difference Az - A: per sentence mean over the 4 references (capped), bootstrap 95% interval over sentences.

Output: results/devgen/mode_summary.json, printed table
Usage (eval env, CPU): PYTHONPATH=src uv run --project envs/eval python scripts/devgen/09_mode_summary.py
"""
import csv
import json
from collections import defaultdict

import numpy as np

from cv3lora.eval_audio import align_errors, canon
from cv3lora.paths import REPO

OUT = REPO / "results/devgen"
N_BOOT, BOOT_SEED = 10_000, 20261104


def read(name: str) -> list[dict]:
    with open(OUT / name) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def cers(cond: str, inp: dict, asr: str) -> dict[tuple[str, str], float]:
    """(id, ref_speaker) -> CER for JA references and seed 0 (R: ref_speaker "")."""
    out = {}
    for a in read(f"asr_{cond}.tsv"):
        if cond != "R" and (a["ref_lang"] != "ja" or a["seed"] != "0"):
            continue
        it = inp[a["id"]]
        out[(a["id"], a["ref_speaker"])] = align_errors(canon(it["ref_kana"]), canon(a[asr]),
                                                         (int(it["tgt_start"]), int(it["tgt_end"])))["cer"]
    return out


def agg(v: list[float]) -> dict:
    v = np.array(v)
    kept = v[v <= 0.5]
    return {"n": len(v), "capped": float(np.minimum(v, 1.0).mean()), "median": float(np.median(v)),
            "kept_share": float(len(kept) / len(v)), "excl>0.5": float(kept.mean()) if len(kept) else None}


def paired(x: dict, y: dict) -> dict:
    """Mean of per-sentence (y - x), capped CER, sentences present on both sides."""
    sx, sy = defaultdict(list), defaultdict(list)
    for (i, _), c in x.items():
        sx[i].append(min(c, 1.0))
    for (i, _), c in y.items():
        sy[i].append(min(c, 1.0))
    ids = sorted(set(sx) & set(sy))
    d = np.array([np.mean(sy[i]) - np.mean(sx[i]) for i in ids])
    m = d[np.random.default_rng(BOOT_SEED).integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n_sentences": len(ids), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def main() -> None:
    inp = {r["id"]: r for r in read("inputs.tsv")}
    res = {}
    for asr in ("kw", "ctc"):
        c = {cond: cers(cond, inp, asr) for cond in ("A", "Az", "Ap", "R")}
        assert set(c["A"]) == set(c["Az"]), "A and Az must cover the same files"
        res[asr] = {cond: agg(list(v.values())) for cond, v in c.items()} | {"Az-A": paired(c["A"], c["Az"])}
    with open(OUT / "mode_summary.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for asr, r in res.items():
        print(f"[{asr}]  cond  n  capped  median  kept  excl>0.5")
        for cond in ("A", "Az", "Ap", "R"):
            a = r[cond]
            ex = f"{a['excl>0.5']:.3f}" if a["excl>0.5"] is not None else "-"
            print(f"  {cond:3s} {a['n']:5d}  {a['capped']:.3f}  {a['median']:.3f}  {a['kept_share']:.2f}  {ex}")
        p = r["Az-A"]
        print(f"  Az-A (per sentence, capped): {p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] n={p['n_sentences']}")


if __name__ == "__main__":
    main()
