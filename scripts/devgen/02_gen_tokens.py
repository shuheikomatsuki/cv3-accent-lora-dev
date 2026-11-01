"""Speech tokens of the dev generation, once per condition x sentence x LLM seed (research plan §12, §13.3).

Conditions (§14): A = base + plain text, Ap = A' (base + all katakana), C = LoRA + tagged text,
D = LoRA + plain text, Cp = C' (LoRA + all katakana, tag on the target only). Inputs from results/devgen/inputs.tsv (01_inputs.py). LLM seeds 0 and 1.
The LoRA is the adopted checkpoint (step 10000 of run main) unless --adapter is given.
Run stage 1 (A, Ap) before stage 2 (C, D) (§12).

Output: <work_root>/devgen/tokens/<cond>/<id>_s<seed>.pt, results/devgen/tokens_<cond>.tsv
Usage: uv run python scripts/devgen/02_gen_tokens.py --cond A Ap
       PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/devgen/02_gen_tokens.py --cond C D
"""
import argparse
import csv
import hashlib

import torch

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens

OUT = REPO / "results/devgen"
SEEDS = (0, 1)
TEXT = {"A": "A", "Ap": "Ap", "C": "C", "D": "A", "Cp": "Cp"}  # inputs.tsv column of each condition
LORA = {"A": False, "Ap": False, "C": True, "D": True, "Cp": True}


def run(cv, cond: str, items: list[dict], paths: dict, model: str) -> None:
    out = paths["work_root"] / "devgen/tokens" / cond
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for it in items:
        rec = torch.load(paths["token_root"] / "jsut_basic5000" / f"{it['id']}.pt").numel()
        for seed in SEEDS:
            g = gen_tokens(cv, it[TEXT[cond]], seed)
            path = out / f"{it['id']}_s{seed}.pt"
            torch.save(g["tokens"], path)
            n = g["tokens"].numel()
            table.append({"cond": cond, "model": model, "id": it["id"], "seed": seed, "n_text": g["n_text"],
                          "n_tokens": n, "n_raw": g["n_raw"], "max_len": g["max_len"], "stop": g["stop"],
                          "rec_tokens": rec, "ratio_to_rec": f"{n / rec:.3f}",
                          "sha256": hashlib.sha256(g["tokens"].numpy().tobytes()).hexdigest(), "path": str(path)})
        print(cond, it["id"], flush=True)
    with open(OUT / f"tokens_{cond}.tsv", "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=list(TEXT))
    ap.add_argument("--adapter")
    args = ap.parse_args()
    lora = {LORA[c] for c in args.cond}
    assert len(lora) == 1, "run base (A, Ap) and LoRA (C, D) conditions separately"
    paths = load_paths()
    with open(OUT / "inputs.tsv") as f:
        items = list(csv.DictReader(f, delimiter="\t"))

    cv = load_cosyvoice3()
    model = "base"
    if lora.pop():
        from cv3lora.lora import load_adapter, register_tags  # needs peft on PYTHONPATH
        adapter = args.adapter or paths["work_root"] / "train/main/ckpt/step10000.pt"
        register_tags(cv.frontend.tokenizer)
        load_adapter(cv.model.llm, adapter)
        model = str(adapter)
    cv.model.llm.eval()
    for cond in args.cond:
        run(cv, cond, items, paths, model)


if __name__ == "__main__":
    main()
