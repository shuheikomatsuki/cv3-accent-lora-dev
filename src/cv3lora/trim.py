"""Leading/trailing silence trim for JVS (research plan §9).

JVS recordings have about 0.5 s of silence at each end; JSUT is already trimmed and is left as is.
Trimming is applied to the canonical speech tokens (25 Hz, one token per 40 ms), not by
re-extracting them from trimmed audio (§10).
"""
from pathlib import Path

import numpy as np
import soundfile as sf

TOKEN_HZ = 25
FRAME_S = 0.01
TOP_DB = 40.0  # frames this far below the loudest frame of the utterance count as silence
MARGIN_S = 0.1  # silence kept at each end, so weak onsets and devoiced endings are not cut


def speech_bounds(wav: Path) -> tuple[float, float, float]:
    """(start, end, duration) in seconds of the part louder than peak - TOP_DB."""
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x.mean(1)
    hop = int(sr * FRAME_S)
    n = len(x) // hop
    db = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    voiced = np.flatnonzero(db > db.max() - TOP_DB)
    return voiced[0] * FRAME_S, (voiced[-1] + 1) * FRAME_S, len(x) / sr


def token_span(start: float, end: float, n_tokens: int) -> tuple[int, int]:
    """Token slice [a, b) that keeps MARGIN_S of silence at each end; never widens past the tokens."""
    a = max(0, int(np.floor((start - MARGIN_S) * TOKEN_HZ)))
    b = min(n_tokens, int(np.ceil((end + MARGIN_S) * TOKEN_HZ)))
    return a, b
