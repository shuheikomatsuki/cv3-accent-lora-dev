"""Condition Az: condition A (base, plain text) in zero_shot mode, for the cause of the gap to UtterTune's CER.

Side check, not part of the §12 judgement. A in cross_lingual (the main mode, §2.1) misreads the kanji parts
widely (dev CER 0.52); UtterTune reports a low CER for CosyVoice 2 on plain text in zero_shot. This condition
changes only the mode: same inputs (inputs.tsv column A, no text_normalize, as 02_gen_tokens.py), base model,
JA dev references (refs.tsv role dev, use main, JA 4), LLM seed 0, flow seed 0.

In zero_shot the LLM sees the reference (PREFIX + transcript from refs.tsv, and its speech tokens), so the
tokens are generated once per sentence x reference (not shared across references as in cross_lingual). Each
token sequence goes through the flow with the same reference.

Output: <work_root>/devgen/tokens/Az/<id>_s0__ja_<speaker>.pt, <work_root>/devgen/wav/Az/<same>.wav,
        results/devgen/tokens_Az.tsv, results/devgen/wav_Az.tsv (same columns as 03_token2wav.py, so
        04_prep.py / 05_asr.py take --cond Az; 09_mode_summary.py compares with A)
Usage: uv run python scripts/devgen/08_zero_shot.py
"""
import argparse
import csv
import hashlib

import torch
import torchaudio

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens, token2wav, zero_shot_prompt
from cv3lora.trim import TOKEN_HZ

OUT = REPO / "results/devgen"
COND, SEED, FLOW_SEED = "Az", 0, 0


def write(name: str, table: list[dict]) -> None:
    with open(OUT / name, "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="first N sentences only (smoke test; tables get a _smoke suffix)")
    args = ap.parse_args()
    paths = load_paths()
    with open(OUT / "inputs.tsv") as f:
        items = list(csv.DictReader(f, delimiter="\t"))
    if args.limit:
        items = items[: args.limit]
    with open(REPO / "data/refs/refs.tsv") as f:
        refs = [r for r in csv.DictReader(f, delimiter="\t")
                if r["role"] == "dev" and r["use"] == "main" and r["lang"] == "ja"]
    assert len(refs) == 4, len(refs)

    cv = load_cosyvoice3()
    cv.model.llm.eval()
    prompts = {r["speaker"]: zero_shot_prompt(cv, r["path"], r["text"]) for r in refs}
    tok_dir, wav_dir = (paths["work_root"] / "devgen" / d / COND for d in ("tokens", "wav"))
    tok_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    tok_table, wav_table = [], []
    for it in items:
        rec = torch.load(paths["token_root"] / "jsut_basic5000" / f"{it['id']}.pt").numel()
        for r in refs:
            p = prompts[r["speaker"]]
            g = gen_tokens(cv, it["A"], SEED, prompt=p)
            name = f"{it['id']}_s{SEED}__{r['lang']}_{r['speaker']}"
            tokens, n = g["tokens"], g["tokens"].numel()
            torch.save(tokens, tok_dir / f"{name}.pt")
            tok_table.append({"cond": COND, "model": "base", "mode": "zero_shot", "id": it["id"], "seed": SEED,
                              "ref_speaker": r["speaker"], "n_text": g["n_text"],
                              "n_prompt_text": p["prompt_text"].shape[1],
                              "n_prompt_speech": p["llm_prompt_speech_token"].shape[1], "n_tokens": n,
                              "n_raw": g["n_raw"], "max_len": g["max_len"], "stop": g["stop"], "rec_tokens": rec,
                              "ratio_to_rec": f"{n / rec:.3f}",
                              "sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                              "path": str(tok_dir / f"{name}.pt")})
            wav = token2wav(cv, tokens, p, FLOW_SEED)
            path = wav_dir / f"{name}.wav"
            torchaudio.save(str(path), wav, cv.sample_rate, encoding="PCM_S", bits_per_sample=16)
            wav_table.append({"cond": COND, "id": it["id"], "seed": SEED, "ref_lang": r["lang"],
                              "ref_gender": r["gender"], "ref_speaker": r["speaker"], "ref_path": r["path"],
                              "n_tokens": n, "dur_s": f"{wav.shape[1] / cv.sample_rate:.3f}",
                              "expected_s": f"{n / TOKEN_HZ:.3f}", "flow_seed": FLOW_SEED, "wav": str(path)})
        print(COND, it["id"], flush=True)
    suffix = "_smoke" if args.limit else ""
    write(f"tokens_{COND}{suffix}.tsv", tok_table)
    write(f"wav_{COND}{suffix}.tsv", wav_table)


if __name__ == "__main__":
    main()
