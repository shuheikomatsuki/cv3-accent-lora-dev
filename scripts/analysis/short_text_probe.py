"""Does the base model break down on short input? (step 6 follow-up; matters for §13.4 carrier and §13 eval)

In step 6 the base model gave 0.5-1.7 s of garbled speech for 「これは橋だと思います。」. Suspected cause: until
min_len, CosyVoice3LM.sampling_ids masks only speech_token_size (sos), not the eos token, so min_len does not
apply. This script measures, with the base model (cross_lingual, LLM tokens -> flow with one JA dev reference):
  dev      all basic5000 dev sentences (plain transcript, fixes applied) x seeds
  carrier  carrier sentences of 4 lengths x 端橋箸鼻花 x seeds
  and both again with --patch-eos for the short ones (all stop tokens masked until min_len).
Per wave: CTC (hiragana wav2vec2) greedy kana vs the expected kana -> CER; for dev also the token count over
the JSUT recording's. Collapse = CER > 0.5 or (dev) ratio < 0.6. Criteria were fixed before the run.
Baseline: CTC CER of the JSUT dev recordings themselves.

Output: results/short_text/probe.tsv (one row per wave), recordings.tsv (baseline)
Usage: uv run python scripts/analysis/short_text_probe.py
"""
import csv
import re
from contextlib import contextmanager

import soundfile as sf
import torch
import torchaudio
import yaml

from cv3lora.accent_f0 import load_ctc
from cv3lora.cosyvoice_loader import load_cosyvoice3
from cv3lora.manifest import load_manifest, original_texts
from cv3lora.paths import REPO, load_paths
from cv3lora.synth import gen_tokens, prompt_inputs, token2wav

OUT = REPO / "results/short_text"
SEEDS = (0, 1, 2)
REF = "jvs029"
SHORT = 20  # dev sentences shorter than this (characters) are also run with the eos patch
WORDS = [("端", "はし"), ("橋", "はし"), ("箸", "はし"), ("鼻", "はな"), ("花", "はな")]
# (id, text with {} for the word, expected kana with {} for the word's kana)
CARRIERS = [
    ("c0", "これは{}だと思います。", "これわ{}だとおもいます"),
    ("c1", "私は、これは{}だと思います。", "わたしわこれわ{}だとおもいます"),
    ("c2", "よく見てみると、これは{}だと思います。", "よくみてみるとこれわ{}だとおもいます"),
    ("c3", "写真をよく見てみると、これはたぶん{}だと思います。", "しゃしんをよくみてみるとこれわたぶん{}だとおもいます"),
]
HIRA = re.compile(r"[ぁ-ゖ]")


def norm(s: str) -> str:
    return "".join(HIRA.findall(s))


def cer(hyp: str, ref: str) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def greedy(model, inv, x, sr) -> str:
    w = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000)
    with torch.inference_mode():
        ids = model(((w - w.mean()) / (w.std() + 1e-7))[None].cuda()).logits.argmax(-1)[0].tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and inv[i] != "[PAD]":
            out.append(inv[i])
        prev = i
    return norm("".join(out))


@contextmanager
def patch_eos(lm):
    """Mask every stop token (not only sos) while ignore_eos is set, i.e. until min_len."""
    orig = lm.sampling_ids

    def sampling_ids(scores, decoded, sampling, ignore_eos=True):
        if ignore_eos:
            scores[lm.speech_token_size: lm.speech_token_size + 200] = -float("inf")
        return lm.sampling(scores, decoded, sampling)

    lm.sampling_ids = sampling_ids
    try:
        yield
    finally:
        lm.sampling_ids = orig


def main() -> None:
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(REPO / "data/splits/utts.tsv") as f:
        dev_ids = {r["id"] for r in csv.DictReader(f, delimiter="\t") if r["set"] == "dev"}
    rows = [r for r in load_manifest() if r["id"] in dev_ids]
    text = original_texts(paths, rows)
    label = yaml.safe_load((paths["jsut_label_root"] / "text_kana/basic5000.yaml").read_text())
    items = []
    for r in sorted(rows, key=lambda r: r["id"]):
        rec = torch.load(paths["token_root"] / "jsut_basic5000" / f"{r['id']}.pt").numel()
        items.append({"set": "dev", "key": r["id"], "text": text[r["id"]], "kana": norm(label[r["file_id"]]["kana_level0"]),
                      "rec_tokens": rec, "wav_rec": paths["jsut_root"] / "basic5000/wav" / f"{r['file_id']}.wav"})
    for cid, t, k in CARRIERS:
        for w, wk in WORDS:
            items.append({"set": cid, "key": f"{cid}_{w}", "text": t.format(w), "kana": k.format(wk), "rec_tokens": 0})

    model, vocab = load_ctc()
    inv = {v: k for k, v in vocab.items()}
    with open(OUT / "recordings.tsv", "w") as f:
        f.write("key\tn_chars\tcer\n")
        for it in items:
            if it["set"] == "dev":
                x, sr = sf.read(it["wav_rec"], dtype="float32")
                f.write(f"{it['key']}\t{len(it['text'])}\t{cer(greedy(model, inv, x, sr), it['kana']):.3f}\n")

    cv = load_cosyvoice3()
    lm = cv.model.llm
    lm.eval()
    ref = next(r for r in csv.DictReader(open(REPO / "data/refs/refs.tsv"), delimiter="\t")
               if r["speaker"] == REF and r["use"] == "main")
    prompt = prompt_inputs(cv, ref["path"])
    out = open(OUT / "probe.tsv", "w")
    cols = ["set", "key", "patch", "seed", "n_chars", "n_text", "n_tokens", "min_len", "stop", "ratio_to_rec", "cer",
            "collapse", "hyp", "kana", "text"]
    wr = csv.DictWriter(out, cols, delimiter="\t")
    wr.writeheader()
    for patch in (False, True):
        for it in items:
            if patch and it["set"] == "dev" and len(it["text"]) >= SHORT:
                continue
            for seed in SEEDS:
                if patch:
                    with patch_eos(lm):
                        g = gen_tokens(cv, it["text"], seed)
                else:
                    g = gen_tokens(cv, it["text"], seed)
                wav = token2wav(cv, g["tokens"], prompt, 0)[0].numpy()
                hyp = greedy(model, inv, wav, cv.sample_rate)
                c = cer(hyp, it["kana"])
                ratio = g["tokens"].numel() / it["rec_tokens"] if it["rec_tokens"] else None
                wr.writerow({"set": it["set"], "key": it["key"], "patch": int(patch), "seed": seed,
                             "n_chars": len(it["text"]), "n_text": g["n_text"], "n_tokens": g["tokens"].numel(),
                             "min_len": int(g["n_text"] * 2), "stop": g["stop"],
                             "ratio_to_rec": f"{ratio:.3f}" if ratio else "", "cer": f"{c:.3f}",
                             "collapse": int(c > 0.5 or (ratio is not None and ratio < 0.6)),
                             "hyp": hyp, "kana": it["kana"], "text": it["text"]})
            out.flush()
            print(patch, it["key"], flush=True)
    out.close()


if __name__ == "__main__":
    main()
