"""F0 around a target word located by CTC forced alignment (a listening aid; §15.2 is still undecided).

Ported from the review scripts (/workH/kmt/gr/review/spot_listen, spot_compare, uttertune_pairs) so the
same measurement runs inside the repo: hiragana CTC (wav2vec2) forced alignment of known kana gives the
onset of each mora of the word and of the next mora; F0 by pYIN in semitones relative to the
utterance median. drop = median F0 of the next mora minus that of the last mora of the word.
"""
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC

from cv3lora.manifest import morae

CTC = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
FRAME_S = 0.02  # wav2vec2 frame hop at 16 kHz
VOWEL = {**dict.fromkeys("あかさたなはまやらわがざだばぱぁゃ", "あ"), **dict.fromkeys("いきしちにひみりぎじぢびぴぃ", "い"),
         **dict.fromkeys("うくすつぬふむゆるぐずづぶぷぅゅゔ", "う"), **dict.fromkeys("えけせてねへめれげぜでべぺぇ", "え"),
         **dict.fromkeys("おこそとのほもよろをごぞどぼぽぉょ", "お")}


def hira(kana: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana.replace("'", "").replace("/", ""))


def long_as_vowel(h: str, ortho: bool) -> str:
    """ー -> the preceding vowel, or with ortho the kana spelling (おう, えい) the CTC labels may use."""
    out = ""
    for c in h:
        v = VOWEL.get(out[-1:], "ー")
        out += ({"お": "う", "え": "い"}.get(v, v) if ortho else v) if c == "ー" else c
    return out


def load_ctc():
    """(model on cuda, vocab) from the local Hugging Face cache."""
    snap = next((Path.home() / ".cache/huggingface/hub" / f"models--{CTC.replace('/', '--')}/snapshots").iterdir())
    return Wav2Vec2ForCTC.from_pretrained(snap).cuda().eval(), json.loads((snap / "vocab.json").read_text())


def align_morae(model, vocab, wav16: torch.Tensor, pre: str, span: str, post: str) -> list[float]:
    """Onset (s) of each mora of span, plus the onset of the following kana (end of span)."""
    with torch.inference_mode():
        x = (wav16 - wav16.mean()) / (wav16.std() + 1e-7)
        lp = model(x[None].cuda()).logits.log_softmax(-1)[0].float()
    best = None
    for f in (lambda h: h, lambda h: long_as_vowel(h, False), lambda h: long_as_vowel(h, True)):
        p, w, q = (f(hira(t)) for t in (pre, span, post))
        p, q = [c for c in p if c in vocab], [c for c in q if c in vocab]
        assert all(c in vocab for c in w), w  # mora onsets need every kana of the word
        ids = torch.tensor([[vocab[c] for c in p + list(w) + q]], device="cuda")
        path, score = torchaudio.functional.forced_align(lp[None], ids, blank=vocab["[PAD]"])
        if best is None or score.sum() > best[0]:
            best = (float(score.sum()), path[0].tolist(), len(p), len(w))
    _, path, n_pre, n_word = best
    onsets, prev = [], None
    for t, k in enumerate(path):
        if k != vocab["[PAD]"] and k != prev:
            onsets.append(t)
        prev = k if k != vocab["[PAD]"] else None
    char_on = onsets[n_pre: n_pre + n_word] + [onsets[n_pre + n_word] if n_pre + n_word < len(onsets) else len(path)]
    idx = np.cumsum([0] + [len(m) for m in morae(span.replace("'", "").replace("/", ""))])
    return [char_on[i] * FRAME_S for i in idx]


def f0_semitones(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    hop = sr // 200  # 5 ms
    f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=600, sr=sr, frame_length=2048, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)
    st = 12 * np.log2(f0 / np.nanmedian(f0)) if np.any(voiced) else f0
    return np.arange(len(f0)) * hop / sr, st


def mora_drop(model, vocab, x: np.ndarray, sr: int, pre: str, word: str, nxt: str, post: str) -> dict:
    """Median F0 (semitones) per mora of word + nxt and the drop into nxt. Raises if alignment fails."""
    b = align_morae(model, vocab, torchaudio.functional.resample(torch.from_numpy(x), sr, 16000), pre, word + nxt, post)
    t, st = f0_semitones(x, sr)
    med = [float(np.nanmedian(st[(t >= p) & (t < q)])) if np.any(np.isfinite(st[(t >= p) & (t < q)])) else float("nan")
           for p, q in zip(b, b[1:])]
    return {"bounds": b, "mora_f0": med, "drop": med[-1] - med[-2]}
