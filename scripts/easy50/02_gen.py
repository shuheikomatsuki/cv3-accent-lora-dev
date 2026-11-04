"""CV3 side of the easy_50 checks, cross_lingual. --cond C (default): CV3+LoRA (step 10000) on phon_tagged
(UtterTune's Test Set 1 input), LLM seeds 0 and 1. --cond A: CV3 base on the plain text (vanilla), LLM seed 0 only
(the CV3 base cell of the {CV2, CV3} x {base, +LoRA} 2x2; the CV2 side is 05_gen_cv2.py).

Check of how this study's model scores on UtterTune's Test Set 1; decided by the user on 2026-11-04, before the
final-evaluation design is fixed, and not used to choose conditions (results/easy50/README.md).
Tokens once per sentence x seed (§13.3); flow seed 0 with each JA dev reference
(refs.tsv role dev, use main, JA 4). Same code path as scripts/devgen/02_gen_tokens.py and 03_token2wav.py.

Output: <work_root>/easy50/tokens/<cond>/<id>_s<seed>.pt, <work_root>/easy50/wav/<cond>/<id>_s<seed>__ja_<spk>.wav,
        results/easy50/tokens_<cond>.tsv, results/easy50/wav_<cond>.tsv (_smoke suffix with --limit)
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/easy50/02_gen.py [--cond A] [--limit 2]
"""
import argparse
import csv
import hashlib

import torch
import torchaudio

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens, prompt_inputs, token2wav
from cv3lora.trim import TOKEN_HZ

OUT = REPO / "results/easy50"
SEEDS = {"C": (0, 1), "A": (0,)}
FLOW_SEED = 0


def write(name: str, table: list[dict]) -> None:
    with open(OUT / name, "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="C", choices=list(SEEDS))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    paths = load_paths()
    with open(OUT / "inputs.tsv") as f:
        items = list(csv.DictReader(f, delimiter="\t"))[: args.limit]
    with open(REPO / "data/refs/refs.tsv") as f:
        refs = [r for r in csv.DictReader(f, delimiter="\t")
                if r["role"] == "dev" and r["use"] == "main" and r["lang"] == "ja"]
    assert len(refs) == 4, len(refs)

    COND = args.cond
    cv = load_cosyvoice3()
    model = "base"
    if COND == "C":
        from cv3lora.lora import load_adapter, register_tags  # needs peft on PYTHONPATH
        adapter = paths["work_root"] / "train/main/ckpt/step10000.pt"
        register_tags(cv.frontend.tokenizer)
        load_adapter(cv.model.llm, adapter)
        model = str(adapter)
    cv.model.llm.eval()
    prompts = {r["speaker"]: prompt_inputs(cv, r["path"]) for r in refs}
    tok_dir, wav_dir = (paths["work_root"] / "easy50" / d / COND for d in ("tokens", "wav"))
    tok_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    tok_table, wav_table = [], []
    for it in items:
        for seed in SEEDS[COND]:
            g = gen_tokens(cv, it[COND], seed)
            tokens, n = g["tokens"], g["tokens"].numel()
            tpath = tok_dir / f"{it['id']}_s{seed}.pt"
            torch.save(tokens, tpath)
            tok_table.append({"cond": COND, "model": model, "id": it["id"], "seed": seed,
                              "n_text": g["n_text"], "n_tokens": n, "n_raw": g["n_raw"], "max_len": g["max_len"],
                              "stop": g["stop"], "sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                              "path": str(tpath)})
            for r in refs:
                wav = token2wav(cv, tokens, prompts[r["speaker"]], FLOW_SEED)
                path = wav_dir / f"{it['id']}_s{seed}__{r['lang']}_{r['speaker']}.wav"
                torchaudio.save(str(path), wav, cv.sample_rate, encoding="PCM_S", bits_per_sample=16)
                wav_table.append({"cond": COND, "id": it["id"], "seed": seed, "ref_lang": r["lang"],
                                  "ref_gender": r["gender"], "ref_speaker": r["speaker"], "ref_path": r["path"],
                                  "n_tokens": n, "dur_s": f"{wav.shape[1] / cv.sample_rate:.3f}",
                                  "expected_s": f"{n / TOKEN_HZ:.3f}", "flow_seed": FLOW_SEED, "wav": str(path)})
        print(COND, it["id"], flush=True)
    suffix = "_smoke" if args.limit else ""
    write(f"tokens_{COND}{suffix}.tsv", tok_table)
    write(f"wav_{COND}{suffix}.tsv", wav_table)


if __name__ == "__main__":
    main()
