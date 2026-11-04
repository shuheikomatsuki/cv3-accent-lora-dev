"""Whisper large-v3 constrained to kana and CJK symbols, re-implemented after UtterTune (arXiv 2508.09767).

The paper forces Whisper large-v3 to output "only kana and a limited set of symbols"; neither the paper nor its
public code gives the implementation, so this is our own. Allowed output: characters U+3000-U+30FF (CJK symbols
and punctuation, hiragana, katakana) and the ASCII space; nothing else, no timestamps.

Whisper's tokenizer is byte-level BPE. 43 of the 177 kana have no single token and come out as two byte-fragment
tokens (e.g. ぃ = E3 81 83 -> tokens "E3 81" + "83"), so a filter on decoded tokens would make them impossible.
KanaLogits tracks the incomplete UTF-8 bytes at the end of each hypothesis and allows a token only if the bytes
so far plus the token's bytes are valid characters in range, possibly ending in the prefix of one. Every
character in range is E3 [80-83] [80-BF], so the pending state is one of: none, E3, E3 80..E3 83 (6 states);
the allowed-token mask of each state is precomputed. End of text is allowed only with nothing pending.
"""
import torch
from transformers import LogitsProcessor

WHISPER_V3 = "openai/whisper-large-v3"
WHISPER_V3_REV = "06f233fe06e710322aca913c1bc4249a0d71fce1"  # local snapshot, no refs/main
LO, HI = 0x3000, 0x30FF


def _byte_decoder() -> dict[str, int]:
    """GPT-2 byte-to-unicode table, inverted (token strings -> bytes)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tokenizer) -> dict[int, bytes]:
    """Bytes of every ordinary (non-special) token."""
    dec = _byte_decoder()
    out = {}
    for tok, i in tokenizer.get_vocab().items():
        if i in tokenizer.all_special_ids or tok.startswith("<|"):
            continue
        out[i] = bytes(dec[c] for c in tok)
    return out


def step(state: bytes, b: bytes) -> bytes | None:
    """Pending bytes after appending b to state, or None if the result leaves the allowed set."""
    s = state + b
    i = 0
    while i < len(s):
        if s[i] == 0x20:
            i += 1
            continue
        rest = s[i:i + 3]
        if rest[0] != 0xE3 or (len(rest) > 1 and not 0x80 <= rest[1] <= 0x83) or (len(rest) > 2 and not 0x80 <= rest[2] <= 0xBF):
            return None
        if len(rest) < 3:
            return rest  # prefix of an allowed character
        i += 3
    return b""


STATES = [b"", b"\xe3"] + [bytes([0xE3, x]) for x in range(0x80, 0x84)]


class KanaLogits(LogitsProcessor):
    def __init__(self, tokenizer, vocab_size: int, device):
        self.tb = token_bytes(tokenizer)
        self.eos = tokenizer.eos_token_id
        self.masks = {}
        for st in STATES:
            m = torch.zeros(vocab_size, dtype=torch.bool)
            for i, b in self.tb.items():
                if b and step(st, b) is not None:
                    m[i] = True
            if st == b"":
                m[self.eos] = True
            self.masks[st] = m.to(device)

    def pending(self, ids: list[int]) -> bytes:
        st = b""
        for i in ids:
            b = self.tb.get(i)
            if b is None:  # special tokens of the decoder prompt
                continue
            st = step(st, b)
            assert st is not None, "hypothesis left the allowed set"
        return st

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for r in range(input_ids.shape[0]):
            m = self.masks[self.pending(input_ids[r].tolist())]
            scores[r, : m.shape[0]] = scores[r, : m.shape[0]].masked_fill(~m, float("-inf"))
            scores[r, m.shape[0]:] = float("-inf")
        return scores


def to_katakana(text: str) -> str:
    """Paper's post-processing: keep characters in range, drop spaces and symbols, hiragana -> katakana."""
    out = []
    for c in text:
        if "ぁ" <= c <= "ゖ":
            out.append(chr(ord(c) + 0x60))
        elif "ァ" <= c <= "ヺ" or c == "ー":
            out.append(c)
    return "".join(out)
