"""Trim, loudness-normalize and resample the dev generation audio for the evaluation tools (research plan §12).

Conditions: A, Ap, C, D, O (wav tables of 03_token2wav.py) and R = the JSUT dev recordings themselves
(249, excluded recordings dropped; not synthesized, the lower bound of the ASR error). The 8 dev references are
processed too (refs). Processing: cv3lora.eval_audio.prep.

Output: <work_root>/devgen/eval16k/<cond>/<name>.wav, results/devgen/prep_<cond>.tsv
Usage (eval env): PYTHONPATH=src uv run --project envs/eval python scripts/devgen/04_prep.py --cond A Ap O R
"""
import argparse
import csv
from pathlib import Path

import soundfile as sf

from cv3lora.eval_audio import SR, prep
from cv3lora.manifest import load_excluded
from cv3lora.paths import REPO, load_paths

OUT = REPO / "results/devgen"


def items(cond: str, paths: dict) -> list[dict]:
    if cond == "refs":
        with open(REPO / "data/refs/refs.tsv") as f:
            return [{"id": "", "seed": "", "ref_lang": r["lang"], "ref_gender": r["gender"], "ref_speaker": r["speaker"],
                     "wav": r["path"], "name": f"{r['lang']}_{r['speaker']}"}
                    for r in csv.DictReader(f, delimiter="\t") if r["role"] == "dev" and r["use"] == "main"]
    if cond == "R":
        with open(OUT / "inputs.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        excluded = load_excluded()
        return [{"id": r["id"], "seed": "rec", "ref_lang": "", "ref_gender": "", "ref_speaker": "",
                 "wav": str(paths["jsut_root"] / "basic5000/wav" / f"{r['file_id']}.wav"), "name": r["id"]}
                for r in rows if r["id"] not in excluded]
    with open(OUT / f"wav_{cond}.tsv") as f:
        return [{k: r[k] for k in ("id", "seed", "ref_lang", "ref_gender", "ref_speaker", "wav")} | {"name": Path(r["wav"]).stem}
                for r in csv.DictReader(f, delimiter="\t")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=["A", "Ap", "C", "D", "Cp", "O", "R", "refs", "Az"])
    args = ap.parse_args()
    paths = load_paths()
    for cond in args.cond:
        out = paths["work_root"] / "devgen/eval16k" / cond
        out.mkdir(parents=True, exist_ok=True)
        table = []
        for it in items(cond, paths):
            y, info = prep(it["wav"])
            path = out / f"{it['name']}.wav"
            sf.write(path, y, SR, subtype="FLOAT")
            table.append({"cond": cond, **{k: v for k, v in it.items() if k != "name"}, "wav16": str(path),
                          **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in info.items()}})
        with open(OUT / f"prep_{cond}.tsv", "w") as f:
            w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(table)
        print(cond, len(table), "files;", sum(r["clip"] for r in table), "clip", flush=True)


if __name__ == "__main__":
    main()
