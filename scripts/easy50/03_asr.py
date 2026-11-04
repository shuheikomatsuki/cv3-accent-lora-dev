"""Trim/normalize (cv3lora.eval_audio.prep, as scripts/devgen/04_prep.py) and kana-whisper ASR (as 05_asr.py,
kana-whisper only) of the easy_50 audio.

Output: <work_root>/easy50/eval16k/C/<name>.wav, results/easy50/asr_C.tsv (_smoke with --smoke)
Usage (eval env, GPU): PYTHONPATH=src uv run --project envs/eval python scripts/easy50/03_asr.py [--smoke]
"""
import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
from transformers import pipeline

from cv3lora.eval_audio import SR, prep
from cv3lora.paths import REPO, load_paths

OUT = REPO / "results/easy50"
KW = "sbintuitions/kana-whisper"
BATCH = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    paths = load_paths()
    with open(OUT / f"wav_C{sfx}.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = paths["work_root"] / "easy50/eval16k/C"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        y, info = prep(r["wav"])
        r["wav16"] = str(out / f"{Path(r['wav']).stem}.wav")
        sf.write(r["wav16"], y, SR, subtype="FLOAT")
        r["clip"] = info["clip"]
    asr = pipeline("automatic-speech-recognition", model=KW, torch_dtype=torch.float16, device="cuda")
    table = []
    for s in range(0, len(rows), BATCH):
        chunk = rows[s: s + BATCH]
        audio = [sf.read(r["wav16"], dtype="float32")[0] for r in chunk]
        kw = asr(audio, batch_size=BATCH, generate_kwargs={"language": "ja", "task": "transcribe"})
        for r, k in zip(chunk, kw):
            table.append({k2: r[k2] for k2 in ("cond", "id", "seed", "ref_lang", "ref_speaker", "wav16", "clip")}
                         | {"kw": k["text"].strip()})
        print(s + len(chunk), "/", len(rows), flush=True)
    with open(OUT / f"asr_C{sfx}.tsv", "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
