"""Token span of the target word + next mora in each dev recording (research plan §12, selection rules).

The times come from jsut-label's phoneme labels (labels/basic5000/*.lab: phonemes annotated by hand, times by
Julius forced alignment). The labels are grouped into morae (a mora ends at a vowel, N or cl) and must have as
many morae as the label reading kana_level3; the tag kana are then searched in that mora list (a vowel kana
and ー match each other). Several hits: the one whose relative position (start / morae) is closest to the
tag's (characters before the tag / characters outside it). The span is the target morae plus the next mora,
or the target only when a pause or the end follows. Tokens are 25 Hz and JSUT tokens are untrimmed (§9), so
the span is tokens [floor(start * 25), ceil(end * 25)).

Output: results/ckpt_select/spans.tsv (one row per dev utterance; status != ok rows are not used)
        with --clips N: N random spans (seed 20261031) as wavs with 0.3 s context on each side and the span
        alone, for listening (outside the repo: /workH/kmt/gr/review/span_check/)
Usage: uv run python scripts/select/01_target_spans.py [--clips 12]
"""
import argparse
import csv
import math
import random
import re
from pathlib import Path

import soundfile as sf
import torch
import yaml

from cv3lora.manifest import TAG, load_manifest, morae, target_kana
from cv3lora.paths import REPO, load_paths
from cv3lora.train_data import load_set

OUT = REPO / "results/ckpt_select"
CLIPS = Path("/workH/kmt/gr/review/span_check")
TOKEN_HZ = 25
END_MORA = set("aiueoAIUEO") | {"N", "cl"}
VOWELS = set("アイウエオー")
SAME = {"ヅ": "ズ", "ヂ": "ジ", "ヲ": "オ", "ハ": "ワ", "ヘ": "エ"}
COLS = ["id", "status", "target", "n_hits", "next_mora", "t_start", "t_end", "tok_start", "tok_end", "n_tok_utt",
        "label_morae"]


def label_morae(path: Path) -> list[tuple[str, float, float]]:
    """(phonemes of one mora or sil/pau, start s, end s) from an HTS full-context label file."""
    out, cur = [], []
    for line in path.read_text().splitlines():
        s, e, ctx = line.split()
        ph = ctx.split("-")[1].split("+")[0]
        seg = (ph, int(s) / 1e7, int(e) / 1e7)
        if ph in ("sil", "pau"):
            assert not cur, path
            out.append(seg)
            continue
        cur.append(seg)
        if ph in END_MORA:
            out.append(("".join(p for p, _, _ in cur), cur[0][1], cur[-1][2]))
            cur = []
    assert not cur, path
    return out


def kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def same_mora(tag: str, read: str) -> bool:
    return tag == read or (tag in VOWELS and read in VOWELS) or SAME.get(tag) == read


def find_span(text: str, reading: str, lab: list) -> dict:
    """Locate the tag of one dev sentence in its label morae; returns the spans.tsv fields."""
    m = TAG.search(text)
    tgt = morae(target_kana(m.group(1)))
    read = morae(kata(re.sub(r"[、。，．？！「」 ]", "", reading)))
    speech = [i for i, x in enumerate(lab) if x[0] not in ("sil", "pau")]
    row = {"target": "".join(tgt), "label_morae": len(speech)}
    if len(read) != len(speech):
        return {**row, "status": "mora_count_mismatch", "n_hits": 0}
    hits = [s for s in range(len(read) - len(tgt) + 1) if all(map(same_mora, tgt, read[s: s + len(tgt)]))]
    row["n_hits"] = len(hits)
    if not hits:
        return {**row, "status": "not_found"}
    pre, post = text[: m.start()], text[m.end():]
    s = min(hits, key=lambda h: abs(h / len(read) - len(pre) / (len(pre) + len(post))))
    a, b = speech[s], speech[s + len(tgt) - 1]
    nxt = lab[b + 1] if b + 1 < len(lab) else ("sil", 0, 0)
    if nxt[0] not in ("sil", "pau"):
        b += 1
    return {**row, "status": "ok", "next_mora": "" if nxt[0] in ("sil", "pau") else nxt[0],
            "t_start": lab[a][1], "t_end": lab[b][2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0)
    args = ap.parse_args()
    paths = load_paths()
    root = paths["jsut_label_root"]
    reading = yaml.safe_load((root / "text_kana/basic5000.yaml").read_text())
    file_id = {r["id"]: r["file_id"] for r in load_manifest()}
    rows = []
    for it in load_set(paths, "dev"):  # the 249 dev-loss utterances (excluded recordings dropped)
        fid = file_id[it["id"]]
        row = {"id": it["id"], "n_tok_utt": it["st"].numel(),
               **find_span(it["text"], reading[fid]["kana_level3"], label_morae(root / f"labels/basic5000/{fid}.lab"))}
        if row["status"] == "ok":
            row["tok_start"] = math.floor(row["t_start"] * TOKEN_HZ)
            row["tok_end"] = min(math.ceil(row["t_end"] * TOKEN_HZ), row["n_tok_utt"])
            row["t_start"], row["t_end"] = f"{row['t_start']:.3f}", f"{row['t_end']:.3f}"
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "spans.tsv", "w") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)}/{len(rows)} ok; several hits {sum(r['n_hits'] > 1 for r in ok)}; "
          f"pause/end after the target {sum(not r['next_mora'] for r in ok)}; "
          f"span tokens {sum(r['tok_end'] - r['tok_start'] for r in ok)} of {sum(r['n_tok_utt'] for r in ok)}")
    for r in rows:
        if r["status"] != "ok":
            print("not used:", r["id"], r["status"], r["target"])

    if args.clips:
        CLIPS.mkdir(parents=True, exist_ok=True)
        lines = ["id\ttarget\tnext_mora\tt_start\tt_end\tclip\tspan_only"]
        for r in sorted(random.Random(20261031).sample(ok, args.clips), key=lambda r: r["id"]):
            x, sr = sf.read(paths["jsut_root"] / f"basic5000/wav/{file_id[r['id']]}.wav")
            a, b = float(r["t_start"]), float(r["t_end"])
            ctx = x[max(0, int((a - 0.3) * sr)): int((b + 0.3) * sr)]
            span = x[int(r["tok_start"] / TOKEN_HZ * sr): int(r["tok_end"] / TOKEN_HZ * sr)]
            sf.write(CLIPS / f"{r['id']}_context.wav", ctx, sr)
            sf.write(CLIPS / f"{r['id']}_span.wav", span, sr)
            lines.append(f"{r['id']}\t{r['target']}\t{r['next_mora']}\t{a}\t{b}\t{r['id']}_context.wav\t{r['id']}_span.wav")
        (CLIPS / "clips.tsv").write_text("\n".join(lines) + "\n")
        print(f"{args.clips} clips -> {CLIPS}")


if __name__ == "__main__":
    main()
