"""Seen/unseen flags of target words (research plan §8.4).

Target = kana inside <PHON_START>…<PHON_END> without accent marks. Training side = tags of the
distinct training sentences (split set "train": basic5000 train + parallel 95 sentences), without
the recordings in exclude.tsv. Dev and aux_eval targets keep excluded recordings (synthesis uses the text).
seen = a training tag has the same kana. The model only receives the kana inside the tag, so the
surface it replaces (the plain-notation string) does not matter: 一番 / いちばん and 好意 / 行為 are seen.
surface_seen (reference, the earlier rule) = a training tag has the same kana and replaced the same surface.

Output data/splits/targets.tsv, one row per distinct evaluated sentence:
  set (dev / aux_eval / easy_50 / difficult_50), id, span (tag content), kana, surface,
  seen, surface_seen,
  n_train (training sentences with the kana in a tag), train_surfaces,
  n_untagged (training sentences containing the surface outside the tag; not used for seen)
Usage: uv run python scripts/data/05_seen_unseen.py
"""
import csv
from collections import Counter, defaultdict

from cv3lora.manifest import (
    EVAL,
    TAG,
    load_excluded,
    load_manifest,
    original_texts,
    surface_of_tag,
    target_kana,
)
from cv3lora.paths import REPO, load_paths


def main() -> None:
    paths = load_paths()
    rows = load_manifest()
    orig = original_texts(paths, rows)
    with open(REPO / "data/splits/utts.tsv") as f:
        set_of = {r["id"]: r["set"] for r in csv.DictReader(f, delimiter="\t")}

    # Distinct sentences per set (first id in id order represents a sentence)
    excluded = load_excluded()
    sents = defaultdict(dict)
    for r in sorted(rows, key=lambda r: r["id"]):
        s = set_of[r["id"]]
        if s == "train" and r["id"] in excluded:
            continue
        if s in ("train", "dev", "aux_eval"):
            sents[s].setdefault(r["text"], (r["file_id"], orig[r["id"]]))

    train_surf, untagged = defaultdict(list), []
    for text, (_, plain) in sents["train"].items():
        if m := TAG.search(text):
            train_surf[target_kana(m.group(1))].append(surface_of_tag(text, plain) or "")
            untagged.append(text[: m.start()] + "\n" + text[m.end():])
        else:
            untagged.append(text)

    items = [(s, fid, text, plain) for s in ("dev", "aux_eval")
             for text, (fid, plain) in sorted(sents[s].items(), key=lambda kv: kv[1][0])]
    for name in ("easy_50", "difficult_50"):
        with open(EVAL / f"{name}.tsv") as f:
            items += [(name, r["id"], r["phon_tagged"], r["vanilla"])
                      for r in csv.DictReader(f, delimiter="\t")]

    out = []
    for s, uid, text, plain in items:
        span = TAG.search(text).group(1)
        kana, surface = target_kana(span), surface_of_tag(text, plain) or ""
        tr = train_surf.get(kana, [])
        n_untagged = sum(surface in u for u in untagged) if surface else 0
        out.append([s, uid, span, kana, surface, int(bool(tr)), int(surface in tr), len(tr),
                    " ".join(sorted(set(tr))), n_untagged])

    with open(REPO / "data/splits/targets.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["set", "id", "span", "kana", "surface", "seen", "surface_seen", "n_train",
                    "train_surfaces", "n_untagged"])
        w.writerows(out)
    n, seen = Counter(o[0] for o in out), Counter(o[0] for o in out if o[5])
    surface_seen = Counter(o[0] for o in out if o[6])
    for s in n:
        print(f"{s}: {n[s]} sentences, seen {seen[s]} ({seen[s] / n[s]:.0%}), "
              f"surface_seen (earlier rule) {surface_seen[s]}")


if __name__ == "__main__":
    main()
