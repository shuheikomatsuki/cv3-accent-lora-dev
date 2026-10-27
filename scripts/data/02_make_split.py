"""Final speaker and sentence split (research plan §8).

Speakers: per gender, draw 12 JVS speakers at random; 2 of them are dev references, 10 eval
references. The other 76 are training speakers. No other stratification.
basic5000: group identical sentences, keep untagged ones in training, then draw 250 dev and 250
auxiliary-eval groups at random from the tagged groups.
parallel (voiceactress100 = parallel100 text): draw 5 sentences at random for diagnostic set B
from the 97 sentences recorded by all 100 JVS speakers; their recordings never enter training.

Outputs (data/splits/):
  split.json  seed, speaker roles, sentence lists, counts
  utts.tsv    one row per manifest utterance: id, corpus, subset, speaker, file_id, set, diag_b
              id matches the speech token file name; set is train / dev / aux_eval / diag_b /
              heldout_spk (held-out speakers' parallel100, never used for training or dev loss)

Usage: uv run python scripts/data/02_make_split.py [--seed N]
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from cv3lora.manifest import load_manifest
from cv3lora.paths import REPO, load_paths

SEED = 20261027
N_PER_GENDER, N_DEV_PER_GENDER = 12, 2
N_DEV, N_AUX = 250, 250
N_DIAG = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    paths = load_paths()
    rows = load_manifest()

    # Speakers
    gender = {}
    for line in (paths["jvs_root"] / "gender_f0range.txt").read_text().splitlines()[1:]:
        spk, g, *_ = line.split()
        gender[spk] = g
    rng = random.Random(f"{args.seed}-speakers")
    role = {spk: "train" for spk in gender}
    for g in ("M", "F"):
        picked = rng.sample(sorted(s for s in gender if gender[s] == g), N_PER_GENDER)
        dev = set(rng.sample(picked, N_DEV_PER_GENDER))
        for spk in picked:
            role[spk] = "dev" if spk in dev else "eval"

    # basic5000 sentences
    groups = defaultdict(list)
    for r in rows:
        if r["subset"] == "basic5000":
            groups[r["text"]].append(r["file_id"])
    tagged = sorted((ids for text, ids in groups.items() if "<PHON_START>" in text), key=min)
    rng = random.Random(f"{args.seed}-basic5000")
    rng.shuffle(tagged)
    basic_set = {}
    for i, ids in enumerate(tagged):
        s = "dev" if i < N_DEV else "aux_eval" if i < N_DEV + N_AUX else "train"
        basic_set.update(dict.fromkeys(ids, s))
    for ids in groups.values():
        for fid in ids:
            basic_set.setdefault(fid, "train")  # untagged sentences

    # parallel sentences for set B
    complete = sorted(fid for fid, n in Counter(r["file_id"] for r in rows
                                                if r["subset"] == "parallel100").items() if n == 100)
    rng = random.Random(f"{args.seed}-parallel")
    diag = sorted(rng.sample(complete, N_DIAG))

    # Utterance table
    out = []
    for r in rows:
        if r["subset"] == "basic5000":
            s, b = basic_set[r["file_id"]], 0
        else:
            b = int(r["file_id"] in diag)
            held = r["speaker"] and role[r["speaker"]] != "train"
            s = "heldout_spk" if held else "diag_b" if b else "train"
        out.append([r["id"], r["corpus"], r["subset"], r["speaker"], r["file_id"], s, b])
    counts = Counter(o[5] for o in out)

    d = REPO / "data/splits"
    with open(d / "utts.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "corpus", "subset", "speaker", "file_id", "set", "diag_b"])
        w.writerows(out)
    by_role = {k: sorted(s for s in role if role[s] == k) for k in ("train", "dev", "eval")}
    split = {
        "seed": args.seed,
        "speakers": {k: [f"{s}({gender[s]})" for s in v] for k, v in by_role.items()},
        "basic5000": {k: sorted(f for f, s in basic_set.items() if s == k)
                      for k in ("dev", "aux_eval")},
        "parallel_diag_b": diag,
        "counts": {
            "speakers": {k: dict(Counter(gender[s] for s in v)) for k, v in by_role.items()},
            "basic5000_sentences": {k: sum(1 for ids in groups.values() if basic_set[ids[0]] == k)
                                    for k in ("train", "dev", "aux_eval")},
            "utterances": dict(sorted(counts.items())),
            "diag_b_all_speakers": sum(o[6] for o in out),
        },
    }
    (d / "split.json").write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(split["counts"], indent=1))


if __name__ == "__main__":
    main()
