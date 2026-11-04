"""Kana ASR of the processed dev generation audio (research plan §12, §15.1).

Two recognizers on every file of prep_<cond>.tsv:
  kw   sbintuitions/kana-whisper (katakana output), greedy, language ja, transcribe, fp16
  ctc  the hiragana wav2vec2 of cv3lora.accent_f0 (trained on JSUT), greedy CTC decoding
The reference reading is never given to either recognizer.

Output: results/devgen/asr_<cond>.tsv
Usage (eval env): PYTHONPATH=src uv run --project envs/eval python scripts/devgen/05_asr.py --cond A Ap O R
"""
import argparse
import csv

import soundfile as sf
import torch
from transformers import pipeline

from cv3lora.accent_f0 import load_ctc
from cv3lora.paths import REPO

OUT = REPO / "results/devgen"
KW = "sbintuitions/kana-whisper"
BATCH = 16


def ctc_greedy(model, inv, x) -> str:
    w = torch.from_numpy(x).float()
    with torch.inference_mode():
        ids = model(((w - w.mean()) / (w.std() + 1e-7))[None].cuda()).logits.argmax(-1)[0].tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and inv[i] != "[PAD]":
            out.append(inv[i])
        prev = i
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=["A", "Ap", "C", "D", "Cp", "O", "R", "Az"])
    args = ap.parse_args()
    asr = pipeline("automatic-speech-recognition", model=KW, torch_dtype=torch.float16, device="cuda")
    ctc, vocab = load_ctc()
    inv = {v: k for k, v in vocab.items()}
    for cond in args.cond:
        with open(OUT / f"prep_{cond}.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        table = []
        for s in range(0, len(rows), BATCH):
            chunk = rows[s: s + BATCH]
            audio = [sf.read(r["wav16"], dtype="float32")[0] for r in chunk]
            kw = asr(audio, batch_size=BATCH, generate_kwargs={"language": "ja", "task": "transcribe"})
            for r, x, k in zip(chunk, audio, kw):
                table.append({"cond": cond, "id": r["id"], "seed": r["seed"], "ref_lang": r["ref_lang"],
                              "ref_speaker": r["ref_speaker"], "wav16": r["wav16"], "kw": k["text"].strip(),
                              "ctc": ctc_greedy(ctc, inv, x)})
            print(cond, s + len(chunk), "/", len(rows), flush=True)
        with open(OUT / f"asr_{cond}.tsv", "w") as f:
            w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
