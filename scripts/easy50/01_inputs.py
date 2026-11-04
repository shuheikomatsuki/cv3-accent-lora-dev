"""Inputs and reference readings of easy_50 (UtterTune Test Set 1) for the CV3+LoRA cross_lingual check.

Input C = the phon_tagged column as is (UtterTune's input form). Reference reading (katakana only, no symbols):
pyopenjtalk reading of the plain text before and after the tag, joined with the tag's kana (accent marks and
phrase breaks removed), so the target span is known exactly. Readings checked by hand: overrides with reasons in
data/eval/easy50_reading_fixes.tsv (id, ref_kana, reason) replace the automatic reading.

Output: results/easy50/inputs.tsv (id, C, A, target, ref_kana_auto, ref_kana, tgt_start, tgt_end, fixed)
Usage (eval env, CPU): PYTHONPATH=src uv run --project envs/eval python scripts/easy50/01_inputs.py
"""
import csv
import re

import pyopenjtalk

from cv3lora.paths import REPO

SRC = REPO / "data/eval/easy_50.tsv"
FIXES = REPO / "data/eval/easy50_reading_fixes.tsv"
OUT = REPO / "results/easy50"
TAG = re.compile(r"<PHON_START>(.*?)<PHON_END>")


def kata_only(s: str) -> str:
    return "".join(c for c in s if "ァ" <= c <= "ヺ" or c == "ー")


def g2p(s: str) -> str:
    return kata_only(pyopenjtalk.g2p(s, kana=True)) if s else ""


def main() -> None:
    with open(SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fixes = {}
    if FIXES.exists():
        with open(FIXES) as f:
            fixes = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for r in rows:
        m = TAG.search(r["phon_tagged"])
        assert m and len(TAG.findall(r["phon_tagged"])) == 1, r["id"]
        before, after = r["phon_tagged"][: m.start()], r["phon_tagged"][m.end():]
        tgt = kata_only(m.group(1).replace("'", "").replace("/", ""))
        auto = g2p(before) + tgt + g2p(after)
        a = len(g2p(before))
        ref = auto
        if r["id"] in fixes:
            ref = fixes[r["id"]]["ref_kana"]
            assert tgt in ref, r["id"]
            a = ref.index(tgt) if ref.count(tgt) == 1 else int(fixes[r["id"]]["tgt_start"])
        assert ref[a: a + len(tgt)] == tgt, r["id"]
        table.append({"id": f"easy50_{int(r['id']):03d}", "C": r["phon_tagged"], "A": r["vanilla"], "target": tgt,
                      "ref_kana_auto": auto, "ref_kana": ref, "tgt_start": a, "tgt_end": a + len(tgt),
                      "fixed": int(r["id"] in fixes)})
    with open(OUT / "inputs.tsv", "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    print(len(table), "sentences,", sum(t["fixed"] for t in table), "fixed readings")


if __name__ == "__main__":
    main()
