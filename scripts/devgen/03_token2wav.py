"""Flow + HiFT for the dev generation tokens with each dev reference (research plan §12, §13.3, §15.4).

Conditions A, Ap, C, D read the token tables of 02_gen_tokens.py (sha256 re-checked). Condition O passes the
speech tokens extracted from the JSUT dev recordings (249: excluded recordings dropped, §8.1); its seed
column is "rec". References: refs.tsv role dev, use main (JA 4, EN 4). Flow seed 0.

Output: <work_root>/devgen/wav/<cond>/<id>_s<seed>__<lang>_<speaker>.wav, results/devgen/wav_<cond>.tsv
Usage: uv run python scripts/devgen/03_token2wav.py --cond A Ap O
"""
import argparse
import csv
import hashlib

import torch
import torchaudio

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.manifest import load_excluded
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import prompt_inputs, token2wav
from cv3lora.trim import TOKEN_HZ

OUT = REPO / "results/devgen"
FLOW_SEED = 0


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def token_rows(cond: str, paths: dict) -> list[dict]:
    if cond != "O":
        with open(OUT / f"tokens_{cond}.tsv") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    with open(OUT / "inputs.tsv") as f:
        ids = [r["id"] for r in csv.DictReader(f, delimiter="\t")]
    excluded = load_excluded()
    rows = []
    for i in ids:
        if i in excluded:
            continue
        path = paths["token_root"] / "jsut_basic5000" / f"{i}.pt"
        rows.append({"id": i, "seed": "rec", "path": str(path), "sha256": sha(torch.load(path))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=["A", "Ap", "C", "D", "Cp", "O"])
    args = ap.parse_args()
    paths = load_paths()
    with open(REPO / "data/refs/refs.tsv") as f:
        refs = [r for r in csv.DictReader(f, delimiter="\t") if r["role"] == "dev" and r["use"] == "main"]
    assert len(refs) == 8, len(refs)

    cv = load_cosyvoice3()
    del cv.model.llm
    torch.cuda.empty_cache()
    prompts = {r["speaker"]: prompt_inputs(cv, r["path"]) for r in refs}
    for cond in args.cond:
        out = paths["work_root"] / "devgen/wav" / cond
        out.mkdir(parents=True, exist_ok=True)
        table = []
        for t in token_rows(cond, paths):
            tokens = torch.load(t["path"])
            assert sha(tokens) == t["sha256"], t["path"]
            for r in refs:
                wav = token2wav(cv, tokens, prompts[r["speaker"]], FLOW_SEED)
                path = out / f"{t['id']}_s{t['seed']}__{r['lang']}_{r['speaker']}.wav"
                torchaudio.save(str(path), wav, cv.sample_rate, encoding="PCM_S", bits_per_sample=16)
                table.append({"cond": cond, "id": t["id"], "seed": t["seed"], "ref_lang": r["lang"],
                              "ref_gender": r["gender"], "ref_speaker": r["speaker"], "ref_path": r["path"],
                              "n_tokens": tokens.numel(), "dur_s": f"{wav.shape[1] / cv.sample_rate:.3f}",
                              "expected_s": f"{tokens.numel() / TOKEN_HZ:.3f}", "flow_seed": FLOW_SEED,
                              "wav": str(path)})
            print(cond, t["id"], t["seed"], flush=True)
        with open(OUT / f"wav_{cond}.tsv", "w") as f:
            w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
