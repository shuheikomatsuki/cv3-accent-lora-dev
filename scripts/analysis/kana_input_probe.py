"""Base model with all-katakana input (condition A' form, §14) vs the plain-text CER of short_text_probe.py.

Separates "the base model misreads kanji" from "the CTC is poor on synthesized speech": if the CER with
katakana input (reading given) is close to that of the recordings, the CTC is reliable on synthesized speech.
Criterion fixed before the run: median CER <= 0.2 -> CTC reliable, kanji misreading is real; >= 0.4 -> CTC is
the main cause. Input: the first N dev sentences (sorted by id) of probe.tsv, seed 0, the same reference.
Katakana = jsut-label kana_level0 (hiragana, as pronounced: particle は is わ) in katakana, plus 。.
Carriers: c0 and c3 in katakana.

Output: results/short_text/kana_input.tsv
Usage: uv run python scripts/analysis/kana_input_probe.py
"""
import csv
import sys

import yaml

from cv3lora.accent_f0 import load_ctc
from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens, prompt_inputs, token2wav

sys.path.insert(0, str(REPO / "scripts/analysis"))
from short_text_probe import CARRIERS, REF, WORDS, cer, greedy, norm  # noqa: E402

OUT = REPO / "results/short_text"
N = 60


def kata(h: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in h)


def main() -> None:
    paths = load_paths()
    probe = [r for r in csv.DictReader(open(OUT / "probe.tsv"), delimiter="\t") if r["patch"] == "0" and r["seed"] == "0"]
    dev = sorted((r for r in probe if r["set"] == "dev"), key=lambda r: r["key"])[:N]
    label = yaml.safe_load((paths["jsut_label_root"] / "text_kana/basic5000.yaml").read_text())
    items = [{"set": "dev", "key": r["key"], "kana": r["kana"], "plain_cer": r["cer"], "n_chars": r["n_chars"],
              "text": kata(norm(label[r["key"].split("_", 2)[2]]["kana_level0"])) + "。"} for r in dev]
    plain = {r["key"]: r["cer"] for r in probe}
    for cid, _, k in CARRIERS:
        if cid in ("c0", "c3"):
            for w, wk in WORDS:
                items.append({"set": cid, "key": f"{cid}_{w}", "kana": k.format(wk), "plain_cer": plain[f"{cid}_{w}"],
                              "n_chars": "", "text": kata(k.format(wk)) + "。"})
    cv = load_cosyvoice3()
    cv.model.llm.eval()
    ref = next(r for r in csv.DictReader(open(REPO / "data/refs/refs.tsv"), delimiter="\t")
               if r["speaker"] == REF and r["use"] == "main")
    prompt = prompt_inputs(cv, ref["path"])
    model, vocab = load_ctc()
    inv = {v: k for k, v in vocab.items()}
    with open(OUT / "kana_input.tsv", "w") as f:
        w = csv.DictWriter(f, ["set", "key", "n_chars", "n_tokens", "stop", "kana_cer", "plain_cer", "hyp", "text"],
                           delimiter="\t")
        w.writeheader()
        for it in items:
            g = gen_tokens(cv, it["text"], 0)
            hyp = greedy(model, inv, token2wav(cv, g["tokens"], prompt, 0)[0].numpy(), cv.sample_rate)
            w.writerow({"set": it["set"], "key": it["key"], "n_chars": it["n_chars"], "n_tokens": g["tokens"].numel(),
                        "stop": g["stop"], "kana_cer": f"{cer(hyp, it['kana']):.3f}", "plain_cer": it["plain_cer"],
                        "hyp": hyp, "text": it["text"]})
            f.flush()
            print(it["key"], flush=True)


if __name__ == "__main__":
    main()
