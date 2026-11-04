"""CV3+LoRA (step 10000), cross_lingual, on easy_50 phon_tagged (condition C; UtterTune's Test Set 1 input).

Check of how this study's model scores on UtterTune's Test Set 1; decided by the user on 2026-11-04, before the
final-evaluation design is fixed, and not used to choose conditions (results/easy50/README.md).
LLM seeds 0 and 1, tokens once per sentence x seed (§13.3); flow seed 0 with each JA dev reference
(refs.tsv role dev, use main, JA 4). Same code path as scripts/devgen/02_gen_tokens.py and 03_token2wav.py.

Output: <work_root>/easy50/tokens/C/<id>_s<seed>.pt, <work_root>/easy50/wav/C/<id>_s<seed>__ja_<spk>.wav,
        results/easy50/tokens_C.tsv, results/easy50/wav_C.tsv (_smoke suffix with --limit)
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/easy50/02_gen.py [--limit 2]
"""
import argparse
import csv
import hashlib

import torch
import torchaudio

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.lora import load_adapter, register_tags
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens, prompt_inputs, token2wav
from cv3lora.trim import TOKEN_HZ

OUT = REPO / "results/easy50"
COND, SEEDS, FLOW_SEED = "C", (0, 1), 0


def write(name: str, table: list[dict]) -> None:
    with open(OUT / name, "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    paths = load_paths()
    with open(OUT / "inputs.tsv") as f:
        items = list(csv.DictReader(f, delimiter="\t"))[: args.limit]
    with open(REPO / "data/refs/refs.tsv") as f:
        refs = [r for r in csv.DictReader(f, delimiter="\t")
                if r["role"] == "dev" and r["use"] == "main" and r["lang"] == "ja"]
    assert len(refs) == 4, len(refs)

    cv = load_cosyvoice3()
    adapter = paths["work_root"] / "train/main/ckpt/step10000.pt"
    register_tags(cv.frontend.tokenizer)
    load_adapter(cv.model.llm, adapter)
    cv.model.llm.eval()
    prompts = {r["speaker"]: prompt_inputs(cv, r["path"]) for r in refs}
    tok_dir, wav_dir = (paths["work_root"] / "easy50" / d / COND for d in ("tokens", "wav"))
    tok_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    tok_table, wav_table = [], []
    for it in items:
        for seed in SEEDS:
            g = gen_tokens(cv, it["C"], seed)
            tokens, n = g["tokens"], g["tokens"].numel()
            tpath = tok_dir / f"{it['id']}_s{seed}.pt"
            torch.save(tokens, tpath)
            tok_table.append({"cond": COND, "model": str(adapter), "id": it["id"], "seed": seed,
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
