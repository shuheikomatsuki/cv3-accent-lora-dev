"""Shared pieces of the objective evaluation (research plan §12 dev generation, §15).

prep: leading/trailing silence trimmed with the §9 rule (frames TOP_DB below the loudest, MARGIN_S kept),
loudness set to TARGET_LUFS (ITU-R BS.1770, §15; value provisional) by one gain, then 16 kHz mono.
Failure checks use the unprocessed audio.

Kana: both sides in katakana without punctuation, then character-wise (1:1, so indices are kept)
  - a vowel kana that only lengthens the preceding mora becomes ー (セイ/セー, オウ/オー, ユウ/ユー, ...)
  - ハ->ワ, ヘ->エ, ヲ->オ, ヂ->ジ, ヅ->ズ everywhere (the position of particles is not identified)
CER and target errors come from one Levenshtein alignment against the reference kana.
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio

from cv3lora.trim import FRAME_S, MARGIN_S, TOP_DB

TARGET_LUFS = -23.0
SR = 16000
SAME = {"ハ": "ワ", "ヘ": "エ", "ヲ": "オ", "ヂ": "ジ", "ヅ": "ズ"}
ROW = {**dict.fromkeys("アカサタナハマヤラワガザダバパァャ", "ア"), **dict.fromkeys("イキシチニヒミリギジヂビピィ", "イ"),
       **dict.fromkeys("ウクスツヌフムユルグズヅブプゥュヴ", "ウ"), **dict.fromkeys("エケセテネヘメレゲゼデベペェ", "エ"),
       **dict.fromkeys("オコソトノホモヨロヲゴゾドボポォョ", "オ")}
LENGTHEN = {"ア": "ア", "イ": "イエ", "ウ": "ウオ", "エ": "エ", "オ": "オ"}  # long-vowel kana -> rows they lengthen


def prep(path: str) -> tuple[np.ndarray, dict]:
    """(16 kHz float32 audio, info) after trim and loudness normalization."""
    import pyloudnorm

    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    hop = int(sr * FRAME_S)
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    v = np.flatnonzero(db > db.max() - TOP_DB)
    a, b = max(0, int((v[0] * FRAME_S - MARGIN_S) * sr)), min(len(x), int(((v[-1] + 1) * FRAME_S + MARGIN_S) * sr))
    y = x[a:b]
    loud = pyloudnorm.Meter(sr).integrated_loudness(y) if len(y) > 0.4 * sr else float("nan")
    gain = TARGET_LUFS - loud if np.isfinite(loud) else 0.0
    y = y * 10 ** (gain / 20)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    y16 = torchaudio.functional.resample(torch.from_numpy(y).float(), sr, SR).numpy()
    return y16, {"dur_raw": len(x) / sr, "dur_trim": len(y) / sr, "lufs_in": loud, "gain_db": gain, "peak": peak,
                 "clip": peak > 1.0}


def kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def canon(s: str) -> str:
    """Katakana-only string normalized character-wise (same length as the katakana input)."""
    out = []
    for c in kata(s):
        if not ("ァ" <= c <= "ヺ" or c == "ー"):
            continue
        c = SAME.get(c, c)
        if c in LENGTHEN and out and ROW.get(out[-1], "") in LENGTHEN[c]:
            c = "ー"
        out.append(c)
    return "".join(out)


def align_errors(ref: str, hyp: str, span: tuple[int, int]) -> dict:
    """Edit counts of hyp against ref, split into inside / outside the reference span [a, b).

    A substitution or deletion belongs to its reference character; an insertion before reference index i is
    inside when a < i < b. target_ok = no edit inside the span.
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    a, b = span
    inside = outside = 0
    i, j = n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            e, pos, ins = int(ref[i - 1] != hyp[j - 1]), i - 1, False
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            e, pos, ins = 1, i - 1, False
            i -= 1
        else:
            e, pos, ins = 1, i, True
            j -= 1
        if e:
            if (a < pos < b) if ins else (a <= pos < b):
                inside += 1
            else:
                outside += 1
    return {"edits": inside + outside, "inside": inside, "outside": outside, "cer": (inside + outside) / max(1, n),
            "cer_outside": outside / max(1, n - (b - a)), "target_ok": inside == 0}
