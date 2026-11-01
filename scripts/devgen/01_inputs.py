"""Input texts of the dev generation for conditions A, A', C, D, and the kana references (research plan §12).

All 250 dev sentences (excluded recordings included: synthesis from text keeps them, §8.1).
  C       tagged manifest text (fixes applied)
  A, D    plain text: the corpus transcript with transcript_fixes.tsv (the tag replaced by its surface)
  A'      jsut-label kana_level2 (correct reading of the text as spoken: particle は is わ, long vowels ー) in
          katakana, commas kept, 。 at the end, the target replaced by the tag kana without accent marks.
          kana_level3 is not used: it follows the recording, which misreads 3 dev sentences (§12).
  Cp      C' (candidate, not yet in plan §14): the A' text with the target replaced by the tag of C
          (LoRA + all katakana + tag only on the target); readings outside the target equal A'.
The target is located in the kana_level2 morae as in scripts/select/01_target_spans.py (a vowel kana and ー match;
several hits: relative position closest to the tag's). ref_kana = the A' text without punctuation (the CER
reference for every condition); tgt_start/tgt_end = character span of the target in ref_kana.

Output: results/devgen/inputs.tsv
Usage: uv run python scripts/devgen/01_inputs.py
"""
import csv
import re
import sys
from importlib import import_module

import yaml

from cv3lora.manifest import TAG, load_manifest, morae, original_texts, surface_of_tag, target_kana
from cv3lora.paths import REPO, load_paths

sys.path.insert(0, str(REPO / "scripts/select"))
spans = import_module("01_target_spans")  # kata, same_mora (the selection rules)

OUT = REPO / "results/devgen"
PUNCT = re.compile(r"[、。，．？！「」 ]")


def a_prime(text: str, reading: str) -> tuple[str, str, int, int, int]:
    """(A' text, C' text, target start, end in ref_kana, number of hits) for one tagged sentence."""
    m = TAG.search(text)
    tgt = morae(target_kana(m.group(1)))
    parts = re.split(r"(、)", spans.kata(reading.replace("，", "、")))
    seq = [x for p in parts for x in (["、"] if p == "、" else morae(PUNCT.sub("", p)))]
    sp = [i for i, x in enumerate(seq) if x != "、"]
    read = [seq[i] for i in sp]
    hits = [s for s in range(len(read) - len(tgt) + 1) if all(map(spans.same_mora, tgt, read[s: s + len(tgt)]))]
    if not hits:
        raise ValueError(f"target {''.join(tgt)} not found in {''.join(read)}")
    pre, post = text[: m.start()], text[m.end():]
    s = min(hits, key=lambda h: abs(h / len(read) - len(pre) / (len(pre) + len(post))))
    a, b = sp[s], sp[s + len(tgt) - 1] + 1
    cp = "".join(seq[:a]) + m.group(0) + "".join(seq[b:]) + "。"
    seq = seq[:a] + tgt + seq[b:]
    ref_pre = "".join(x for x in seq[:a] if x != "、")
    return "".join(seq) + "。", cp, len(ref_pre), len(ref_pre) + len("".join(tgt)), len(hits)


def main() -> None:
    paths = load_paths()
    label = yaml.safe_load((paths["jsut_label_root"] / "text_kana/basic5000.yaml").read_text())
    with open(REPO / "data/splits/utts.tsv") as f:
        dev = {r["id"] for r in csv.DictReader(f, delimiter="\t") if r["set"] == "dev"}
    rows = sorted((r for r in load_manifest() if r["id"] in dev), key=lambda r: r["id"])
    plain = original_texts(paths, rows)
    out = []
    for r in rows:
        assert surface_of_tag(r["text"], plain[r["id"]]) is not None, r["id"]
        kana, cp, ts, te, hits = a_prime(r["text"], label[r["file_id"]]["kana_level2"])
        ref = PUNCT.sub("", kana)
        out.append({"id": r["id"], "file_id": r["file_id"], "target": ref[ts:te], "n_hits": hits,
                    "tgt_start": ts, "tgt_end": te, "C": r["text"], "A": plain[r["id"]], "Ap": kana, "ref_kana": ref, "Cp": cp})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "inputs.tsv", "w") as f:
        w = csv.DictWriter(f, list(out[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} sentences; several hits {sum(r['n_hits'] > 1 for r in out)}")


if __name__ == "__main__":
    main()
