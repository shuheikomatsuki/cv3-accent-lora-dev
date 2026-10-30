"""Generate speech tokens with the LLM only, once per text x seed, and save them (research plan §13.3).

Input sets:
  dev20       20 basic5000 dev sentences (seed 0 sample of the split, tagged manifest text); records the
              JSUT recording's token count for the length check (step 6 item 1)
  pairs_plain carrier sentence 「これは〇〇だと思います。」 with 端・橋・箸・鼻・花 (condition A, base model)
  pairs_tag   the same carrier with ハシ・ハシ'・ハ'シ・ハナ・ハナ' tags (condition C, needs --adapter)
The pair words are the §13.4 examples; their dictionary types are a guide, not a judgment.

Output: <work_root>/pilot/tokens/<name>/<key>_s<seed>.pt, results/pilot/tokens_<name>.tsv
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/infer/01_gen_tokens.py --set dev20 \
           --adapter <work_root>/pilot/adapter.pt --name dev20_pilot
"""
import argparse
import csv
import hashlib
import random

import torch

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.lora import load_adapter, register_tags
from cv3lora.manifest import load_manifest
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens

RESULTS = REPO / "results/pilot"
CARRIER = ("これは", "だと思います。")
PAIRS_PLAIN = [("ハシ", "端", "平板"), ("ハシ", "橋", "尾高"), ("ハシ", "箸", "頭高"), ("ハナ", "鼻", "平板"), ("ハナ", "花", "尾高")]
PAIRS_TAG = [("ハシ", "ハシ"), ("ハシ", "ハシ'"), ("ハシ", "ハ'シ"), ("ハナ", "ハナ"), ("ハナ", "ハナ'")]


def inputs(name: str, paths: dict) -> tuple[list[dict], tuple[int, ...]]:
    if name == "dev20":
        with open(REPO / "data/splits/utts.tsv") as f:
            dev = sorted(r["id"] for r in csv.DictReader(f, delimiter="\t") if r["set"] == "dev")
        text = {r["id"]: r for r in load_manifest()}
        rows = []
        for i in random.Random(0).sample(dev, 20):
            r = text[i]
            rec = torch.load(paths["token_root"] / f"{r['corpus']}_{r['subset']}" / f"{i}.pt").numel()
            rows.append({"key": i, "kana": "", "form": "", "text": r["text"], "rec_tokens": rec})
        return rows, (0,)
    if name == "pairs_plain":
        return [{"key": f"A_{k}_{s}", "kana": k, "form": f"{s}（{t}）", "text": CARRIER[0] + s + CARRIER[1], "rec_tokens": ""}
                for k, s, t in PAIRS_PLAIN], (0, 1, 2)
    if name == "pairs_tag":
        return [{"key": f"C_{tag}", "kana": k, "form": tag,
                 "text": f"{CARRIER[0]}<PHON_START>{tag}<PHON_END>{CARRIER[1]}", "rec_tokens": ""}
                for k, tag in PAIRS_TAG], (0, 1, 2)
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev20", "pairs_plain", "pairs_tag"])
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    paths = load_paths()
    out = paths["work_root"] / "pilot/tokens" / args.name
    out.mkdir(parents=True, exist_ok=True)

    cv = load_cosyvoice3()
    if args.adapter:
        register_tags(cv.frontend.tokenizer)
        load_adapter(cv.model.llm, args.adapter)
    cv.model.llm.eval()
    rows, seeds = inputs(args.set, paths)
    assert args.adapter or "<PHON_START>" not in rows[0]["text"], "tagged input needs --adapter"
    table = []
    for r in rows:
        for seed in seeds:
            g = gen_tokens(cv, r["text"], seed)
            path = out / f"{r['key']}_s{seed}.pt"
            torch.save(g["tokens"], path)
            sha = hashlib.sha256(g["tokens"].numpy().tobytes()).hexdigest()
            n = g["tokens"].numel()
            table.append({**r, "seed": seed, "model": "lora" if args.adapter else "base", "n_text": g["n_text"],
                          "n_tokens": n, "n_raw": g["n_raw"], "max_len": g["max_len"], "stop": g["stop"],
                          "ratio_to_rec": f"{n / r['rec_tokens']:.3f}" if r["rec_tokens"] else "",
                          "sha256": sha, "path": str(path)})
            print(r["key"], seed, n, g["stop"], flush=True)
    with open(RESULTS / f"tokens_{args.name}.tsv", "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t")
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
