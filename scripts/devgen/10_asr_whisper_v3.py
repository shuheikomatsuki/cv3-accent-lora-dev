"""Kana-constrained Whisper large-v3 ASR (UtterTune's recognizer, re-implemented; cv3lora.whisper_kana).

For checking UtterTune's reported CER; not part of the §12 judgement. Greedy decoding by default (--beams, --prompt optional), language ja, transcribe,
no timestamps, fp16. The recognizer never sees the reference reading.

Input: results/devgen/prep_<cond>.tsv (column wav16) for --cond, or any TSV with a wav column via --tsv.
Output: results/devgen/asr_v3_<cond>.tsv (--cond) or --out; columns of the input id/seed/ref_* kept,
        v3_raw (decoded text) and v3 (katakana, symbols and spaces removed, paper's post-processing).
With --cond R the CER against the reference reading is printed next to kana-whisper (asr_R.tsv), using the
normalization of 07_judge.py, as the check that the constraint lets every kana through.

Usage (eval env, GPU): PYTHONPATH=src uv run --project envs/eval python scripts/devgen/10_asr_whisper_v3.py --cond R
"""
import argparse
import csv

import numpy as np
import soundfile as sf
import torch
from transformers import LogitsProcessorList, WhisperForConditionalGeneration, WhisperProcessor

from cv3lora.eval_audio import align_errors, canon
from cv3lora.paths import REPO
from cv3lora.whisper_kana import WHISPER_V3, WHISPER_V3_REV, KanaLogits, to_katakana

OUT = REPO / "results/devgen"
BATCH = 16  # --batch overrides
KEEP = ("cond", "id", "seed", "ref_lang", "ref_speaker")


def transcribe(rows: list[dict], wav_col: str, beams: int, prompt: str, batch: int = BATCH) -> list[dict]:
    proc = WhisperProcessor.from_pretrained(WHISPER_V3, revision=WHISPER_V3_REV)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_V3, revision=WHISPER_V3_REV,
                                                            torch_dtype=torch.float16).cuda().eval()
    kp = KanaLogits(proc.tokenizer, model.config.vocab_size, "cuda")
    extra = {"prompt_ids": proc.get_prompt_ids(prompt, return_tensors="pt").cuda()} if prompt else {}
    out = []
    for s in range(0, len(rows), batch):
        chunk = rows[s: s + batch]
        audio = []
        for r in chunk:
            x, sr = sf.read(r[wav_col], dtype="float32")
            assert sr == 16000, (r[wav_col], sr)
            audio.append(x)
        feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats.cuda().half(), language="ja", task="transcribe", return_timestamps=False,
                                 num_beams=beams, do_sample=False, logits_processor=LogitsProcessorList([kp]), **extra)
        for r, text in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
            out.append({k: r.get(k, "") for k in KEEP} | {"wav": r[wav_col], "v3_raw": text.strip(),
                                                          "v3": to_katakana(text)})
        print(s + len(chunk), "/", len(rows), flush=True)
    return out


def check_r(table: list[dict]) -> None:
    with open(OUT / "inputs.tsv") as f:
        inp = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    with open(OUT / "asr_R.tsv") as f:
        kw = {r["id"]: r["kw"] for r in csv.DictReader(f, delimiter="\t")}
    res = {"v3": [], "kw": []}
    for t in table:
        it = inp[t["id"]]
        ref, span = canon(it["ref_kana"]), (int(it["tgt_start"]), int(it["tgt_end"]))
        res["v3"].append(align_errors(ref, canon(t["v3"]), span)["cer"])
        res["kw"].append(align_errors(ref, canon(kw[t["id"]]), span)["cer"])
    for k, v in res.items():
        v = np.array(v)
        print(f"R {k}: n={len(v)} mean={v.mean():.4f} median={np.median(v):.4f} "
              f"share_cer0={np.mean(v == 0):.2f} max={v.max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond")
    ap.add_argument("--tsv")
    ap.add_argument("--out")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--prompt", default="", help="previous-text prompt (kana), empty = none")
    ap.add_argument("--tag", default="", help="suffix of the output table")
    args = ap.parse_args()
    assert bool(args.cond) != bool(args.tsv), "give --cond or --tsv"
    src = OUT / f"prep_{args.cond}.tsv" if args.cond else args.tsv
    with open(src) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    table = transcribe(rows, "wav16" if args.cond else "wav", args.beams, args.prompt, args.batch)
    dst = OUT / f"asr_v3{args.tag}_{args.cond}.tsv" if args.cond else args.out
    assert dst, "give --out with --tsv"
    with open(dst, "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    if args.cond == "R":
        check_r(table)


if __name__ == "__main__":
    main()
