"""Training manifest, original transcripts and tag parsing (research plan §7).

data/uttertune/jsutjvs_manifest.tsv is the UtterTune manifest, committed unmodified first and then
corrected in later commits (git log/diff show each change; fixes.tsv lists the reasons).
"""
import csv
import re
from pathlib import Path

from cv3lora.paths import REPO

MANIFEST = REPO / "data/uttertune/jsutjvs_manifest.tsv"
TAG = re.compile(r"<PHON_START>(.*?)<PHON_END>")
SMALL = set("ァィゥェォャュョヮ")


def utt_id(r: dict) -> str:
    """Matches the speech token file name: <speaker or jsut>_<subset>_<file_id>."""
    return f"{r['speaker'] or 'jsut'}_{r['subset']}_{r['file_id']}"


def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["id"] = utt_id(r)
    return rows


def read_transcript(path: Path) -> dict[str, str]:
    return dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)


def original_texts(paths: dict, rows: list[dict]) -> dict[str, str]:
    """Utterance id -> corpus transcript (plain notation) for manifest rows."""
    cache, out = {}, {}
    for r in rows:
        if r["corpus"] == "jsut":
            p = paths["jsut_root"] / r["subset"] / "transcript_utf8.txt"
        else:
            p = paths["jvs_root"] / r["speaker"] / r["subset"] / "transcripts_utf8.txt"
        if p not in cache:
            cache[p] = read_transcript(p)
        out[r["id"]] = cache[p][r["file_id"]]
    return out


def target_kana(span: str) -> str:
    """Tag content without accent marks (the unit for seen/unseen matching, §8.4)."""
    return span.replace("'", "").replace("/", "")


def surface_of_tag(tagged: str, plain: str) -> str | None:
    """Plain-notation string the single tag replaces, or None if the text outside the tag differs."""
    m = TAG.search(tagged)
    pre, post = tagged[: m.start()], tagged[m.end():]
    if not (plain.startswith(pre) and plain.endswith(post)) or len(pre) + len(post) > len(plain):
        return None
    return plain[len(pre): len(plain) - len(post)]


def morae(kana: str) -> list[str]:
    """Split katakana into morae (small ャュョ etc. join the preceding kana)."""
    out = []
    for ch in kana:
        if ch in SMALL and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out
