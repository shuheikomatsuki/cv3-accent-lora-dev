"""Pass saved speech tokens through the flow + HiFT of each reference (research plan §13.3).

The token files from 01_gen_tokens.py are re-hashed and must match the sha256 recorded at generation,
so every reference gets the identical token sequence. References: refs.tsv use=main of --role
(default dev; eval references are not used during development, §12).
Checks (step 6 item 3): duration = tokens / 25 s; re-running with the same seed gives the same wave;
different references give different waves.

Output: <work_root>/pilot/wav/<name>/<key>_s<seed>__<ref>.wav, results/pilot/wav_<name>.tsv and .json
Usage: uv run python scripts/infer/02_token2wav.py --name dev20_pilot --n-keys 10 --langs ja en
"""
import argparse
import csv
import hashlib
import json

import torch
import torchaudio

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import prompt_inputs, token2wav
from cv3lora.trim import TOKEN_HZ

RESULTS = REPO / "results/pilot"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-keys", type=int, default=0, help="first N texts of the token table (0 = all)")
    ap.add_argument("--langs", nargs="+", default=["ja", "en"])
    ap.add_argument("--role", default="dev", choices=["dev"])
    ap.add_argument("--flow-seed", type=int, default=0)
    args = ap.parse_args()
    paths = load_paths()
    out = paths["work_root"] / "pilot/wav" / args.name
    out.mkdir(parents=True, exist_ok=True)

    with open(RESULTS / f"tokens_{args.name}.tsv") as f:
        toks = list(csv.DictReader(f, delimiter="\t"))
    keys = list(dict.fromkeys(t["key"] for t in toks))
    if args.n_keys:
        keys = keys[: args.n_keys]
    toks = [t for t in toks if t["key"] in keys]
    with open(REPO / "data/refs/refs.tsv") as f:
        refs = [r for r in csv.DictReader(f, delimiter="\t")
                if r["role"] == args.role and r["use"] == "main" and r["lang"] in args.langs]

    cv = load_cosyvoice3()
    del cv.model.llm
    torch.cuda.empty_cache()
    prompts = {r["speaker"]: prompt_inputs(cv, r["path"]) for r in refs}
    rows, waves = [], {}
    for t in toks:
        tokens = torch.load(t["path"])
        sha_ok = hashlib.sha256(tokens.numpy().tobytes()).hexdigest() == t["sha256"]
        for r in refs:
            wav = token2wav(cv, tokens, prompts[r["speaker"]], args.flow_seed)
            name = f"{t['key']}_s{t['seed']}__{r['lang']}_{r['speaker']}"
            torchaudio.save(str(out / f"{name}.wav"), wav, cv.sample_rate, encoding="PCM_S", bits_per_sample=16)
            dur, exp = wav.shape[1] / cv.sample_rate, tokens.numel() / TOKEN_HZ
            waves[(t["key"], t["seed"], r["speaker"])] = wav
            rows.append({"key": t["key"], "seed": t["seed"], "kana": t["kana"], "form": t["form"], "model": t["model"],
                         "ref_lang": r["lang"], "ref_gender": r["gender"], "ref_speaker": r["speaker"],
                         "n_tokens": tokens.numel(), "dur_s": f"{dur:.3f}", "expected_s": f"{exp:.3f}",
                         "token_sha_ok": sha_ok, "wav": str(out / f"{name}.wav")})
        print(t["key"], t["seed"], flush=True)

    # re-run the first text with every reference: same seed must give the same wave
    t0 = toks[0]
    repro = max((token2wav(cv, torch.load(t0["path"]), prompts[r["speaker"]], args.flow_seed)
                 - waves[(t0["key"], t0["seed"], r["speaker"])]).abs().max().item() for r in refs)
    # references must change the output: smallest difference between two references for the same tokens
    spk = [r["speaker"] for r in refs]
    min_ref_diff = min(
        (waves[(t["key"], t["seed"], a)][:, :n] - waves[(t["key"], t["seed"], b)][:, :n]).abs().mean().item()
        for t in toks for i, a in enumerate(spk) for b in spk[i + 1:]
        for n in [min(waves[(t["key"], t["seed"], a)].shape[1], waves[(t["key"], t["seed"], b)].shape[1])])
    with open(RESULTS / f"wav_{args.name}.tsv", "w") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    summary = {"n_wavs": len(rows), "n_texts": len(toks), "refs": spk, "flow_seed": args.flow_seed,
               "all_token_sha_ok": all(r["token_sha_ok"] for r in rows),
               "max_abs_dur_err_s": max(abs(float(r["dur_s"]) - float(r["expected_s"])) for r in rows),
               "rerun_max_abs_diff": repro, "min_mean_abs_diff_between_refs": min_ref_diff}
    (RESULTS / f"wav_{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
