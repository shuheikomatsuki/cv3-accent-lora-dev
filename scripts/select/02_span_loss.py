"""Teacher-forced loss of the dev set on the whole utterance and on the target span (research plan §12).

For the base model (step 0) and every saved adapter, the per-token cross-entropy of the 249 dev utterances
is computed with the correct tags and with accent-changed tags (validity check, §12: a tag with a nucleus
loses all nuclei and becomes flat; a flat tag gets a nucleus after its first mora). Sums are taken over the
whole utterance (= dev_loss of the training log), the span of 01_target_spans.py, and the span widened by 1
and 2 tokens on each side (robustness; ±1 is the rule of §12, ±2 is for reference). Same precision as the
dev loss of the training (fp32, eval mode, no autocast).

Output: results/ckpt_select/utt_loss.tsv (one row per step x utterance: ce sums and token counts)
        <work_root>/ckpt_select/token_ce.pt (per-token cross-entropy, for later re-analysis without the GPU)
Usage: PYTHONPATH=<work_root>/pylib/peft017 uv run python scripts/select/02_span_loss.py [--run main]
"""
import argparse
import csv

import torch
import torch.nn.functional as F

from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.lm_loss import length_batches, sequence_logits, text_tokens
from cv3lora.lora import apply_lora, register_tags
from cv3lora.manifest import TAG, morae
from cv3lora.paths import REPO, load_paths
from cv3lora.train_data import load_set

OUT = REPO / "results/ckpt_select"
SPANS = {"span": 0, "span_w1": 1, "span_w2": 2}
COLS = ["step", "id", "full_ce", "full_n"] + [f"{k}_{v}" for k in SPANS for v in ("ce", "n")] + ["wrong_span_ce"]


def wrong_accent(text: str) -> str:
    """The same text with the accent of the tag changed (§12 validity check)."""
    m = TAG.search(text)
    span = m.group(1)
    if "'" in span:
        new = span.replace("'", "")
    else:
        first = morae(span)[0]
        new = first + "'" + span[len(first):]
    return text[: m.start(1)] + new + text[m.end(1):]


def token_ce(lm, items, key) -> dict[str, torch.Tensor]:
    """id -> cross-entropy of each speech token and EOS (len = tokens + 1), text tokens from it[key]."""
    out = {}
    lm.eval()
    with torch.inference_mode():
        for b in length_batches(items, lambda it: it[key].numel() + it["st"].numel() + 3, 12000):
            logits, y, row, n = sequence_logits(lm, [it[key] for it in b], [it["st"] for it in b])
            ce = F.cross_entropy(logits, y, reduction="none").cpu()
            for it, part in zip(b, ce.split(n.tolist())):
                out[it["id"]] = part
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    paths = load_paths()
    with open(OUT / "spans.tsv") as f:
        spans = {r["id"]: (int(r["tok_start"]), int(r["tok_end"]))
                 for r in csv.DictReader(f, delimiter="\t") if r["status"] == "ok"}
    cv = load_cosyvoice3()
    lm, fe = cv.model.llm, cv.frontend
    del cv.model.flow, cv.model.hift
    torch.cuda.empty_cache()
    register_tags(fe.tokenizer)
    dev = [it for it in load_set(paths, "dev") if it["id"] in spans]
    for it in dev:
        it["tt"] = text_tokens(fe, it["text"])
        it["tt_wrong"] = text_tokens(fe, wrong_accent(it["text"]))
        assert not torch.equal(it["tt"], it["tt_wrong"]), it["id"]
    print(f"{len(dev)} dev utterances with a span", flush=True)

    ckpts = sorted((paths["work_root"] / "train" / args.run / "ckpt").glob("step*.pt"))
    rows, saved = [], {}

    def measure(step):
        good, wrong = token_ce(lm, dev, "tt"), token_ce(lm, dev, "tt_wrong")
        saved[step] = {"correct": good, "wrong": wrong}
        for it in dev:
            ce, a, b = good[it["id"]], *spans[it["id"]]
            r = {"step": step, "id": it["id"], "full_ce": f"{ce.sum():.5f}", "full_n": ce.numel(),
                 "wrong_span_ce": f"{wrong[it['id']][a:b].sum():.5f}"}
            for k, wd in SPANS.items():
                lo, hi = max(0, a - wd), min(it["st"].numel(), b + wd)  # never into EOS
                r[f"{k}_ce"], r[f"{k}_n"] = f"{ce[lo:hi].sum():.5f}", hi - lo
            rows.append(r)
        full = sum(float(r["full_ce"]) for r in rows[-len(dev):]) / sum(r["full_n"] for r in rows[-len(dev):])
        span = sum(float(r["span_ce"]) for r in rows[-len(dev):]) / sum(r["span_n"] for r in rows[-len(dev):])
        print(f"step {step}: full {full:.5f} span {span:.5f}", flush=True)

    measure(0)  # base model (the tag rows are the untrained original rows)
    apply_lora(lm, r=16, alpha=64, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "o_proj"])
    params = dict(lm.named_parameters())
    for p in ckpts:
        ck = torch.load(p, map_location="cpu")
        assert set(ck["state"]) == {n for n, q in params.items() if q.requires_grad}
        with torch.no_grad():
            for n, v in ck["state"].items():
                params[n].copy_(v)
        measure(ck["config"]["step"])

    with open(OUT / "utt_loss.tsv", "w") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    work = paths["work_root"] / "ckpt_select"
    work.mkdir(parents=True, exist_ok=True)
    torch.save({"spans": spans, "ce": saved}, work / "token_ce.pt")


if __name__ == "__main__":
    main()
