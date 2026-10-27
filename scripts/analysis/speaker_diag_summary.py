"""Speaker-information diagnostic, step 2: per-speaker summary (research plan §15.3).

Value per speaker = token-weighted mean NLL per speech token over the set (nats).
With a reference tag, value = NLL(tag) - NLL(ref) per utterance first (e.g. lora - base): this change
is what the diagnostic judges by, since it cancels per-speaker differences already in the base model.
JVS speakers are grouped train / dev / eval by data/splits/split.json; JSUT (set B only) is listed
separately and not included in the train statistics.

Output: results/speaker_diag/speakers_<name>.tsv, summary_<name>.md
Usage: uv run python scripts/analysis/speaker_diag_summary.py base [ref]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from cv3lora.paths import REPO

OUT = REPO / "results/speaker_diag"


def load(tag: str) -> dict[str, dict]:
    lines = (OUT / f"loss_{tag}.jsonl").read_text().splitlines()
    return {r["id"]: r for r in map(json.loads, lines)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("ref", nargs="?")
    args = ap.parse_args()
    name = args.tag if args.ref is None else f"{args.tag}_minus_{args.ref}"
    split = json.loads((REPO / "data/splits/split.json").read_text())
    group, gender = {"jsut": "jsut"}, {"jsut": "F"}
    for g, spks in split["speakers"].items():
        for s in spks:
            group[s[:6]], gender[s[:6]] = g, s[7]
    cur = load(args.tag)
    ref = load(args.ref) if args.ref else None

    rows, lines = [], []
    for s in ("A", "B"):
        per = defaultdict(list)  # speaker -> [(value per token, n tokens)]
        for i, r in cur.items():
            if r["set"] == s:
                v = r["nll_sum"] / r["n"] - (ref[i]["nll_sum"] / ref[i]["n"] if ref else 0.0)
                per[r["speaker"]].append((v, r["n"]))
        spk = sorted(per)
        val = {k: np.average([v for v, _ in per[k]], weights=[n for _, n in per[k]]) for k in spk}
        tr = np.array([val[k] for k in spk if group[k] == "train"])
        mu, sd = tr.mean(), tr.std(ddof=1)
        # split-half reliability (odd / even utterances), Spearman-Brown corrected; JVS only
        h = np.array([(np.mean([v for v, _ in per[k][0::2]]), np.mean([v for v, _ in per[k][1::2]]))
                      for k in spk if k != "jsut"])
        rh = np.corrcoef(h.T)[0, 1]
        rel = 2 * rh / (1 + rh)
        for k in spk:
            rows.append({"set": s, "speaker": k, "gender": gender[k], "group": group[k],
                         "n_utt": len(per[k]), "value": val[k], "z_vs_train": (val[k] - mu) / sd,
                         "pct_in_train": 100 * (tr < val[k]).mean()})
        jvs = [k for k in spk if k != "jsut"]
        lines += [f"## Set {s}", "",
                  (f"- JVS speakers {len(jvs)}, utterances {sum(len(per[k]) for k in jvs)} "
                  f"(per speaker {min(len(per[k]) for k in jvs)}–{max(len(per[k]) for k in jvs)})"),
                  f"- train speakers: mean {mu:.4f}, SD {sd:.4f} (nats per speech token)",
                  f"- split-half reliability of the per-speaker value (Spearman–Brown): {rel:.2f}", "",
                  "| group | n | mean | mean z vs train |", "|---|---:|---:|---:|"]
        for g in ("train", "dev", "eval", "jsut"):
            sel = [r for r in rows if r["set"] == s and r["group"] == g]
            if sel:
                lines.append(f"| {g} | {len(sel)} | {np.mean([r['value'] for r in sel]):.4f} | "
                             f"{np.mean([r['z_vs_train'] for r in sel]):+.2f} |")
        lines += ["", "| dev speaker | gender | utts | value | z vs train | percentile in train |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in rows:
            if r["set"] == s and r["group"] == "dev":
                lines.append(f"| {r['speaker']} | {r['gender']} | {r['n_utt']} | {r['value']:.4f} | "
                             f"{r['z_vs_train']:+.2f} | {r['pct_in_train']:.0f} |")
        lines.append("")

    with open(OUT / f"speakers_{name}.tsv", "w") as f:
        f.write("\t".join(rows[0]) + "\n")
        for r in rows:
            f.write("\t".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
    what = "NLL per speech token" if args.ref is None else f"NLL({args.tag}) − NLL({args.ref}) per speech token"
    head = [f"# Speaker-information loss diagnostic: {name}", "",
            f"value = {what}; split seed {split['seed']}", ""]
    (OUT / f"summary_{name}.md").write_text("\n".join(head + lines))
    print("\n".join(head + lines))


if __name__ == "__main__":
    main()
