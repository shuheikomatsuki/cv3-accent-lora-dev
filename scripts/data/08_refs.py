"""Adopted and spare reference utterances from the listening verdicts (research plan §13.2).

Inputs: data/refs/candidates.tsv (07_ref_candidates.py), data/refs/listening.tsv (id = <speaker>:<utt id>,
verdict adopt / spare / ng, note; JECS verdicts are on the JA row and apply to the JA/EN pair).
Checks the listening rule of 07_ref_candidates.py: per speaker, candidates were heard in rank order, so
every rank before the adopted one is ng, every rank between adopt and spare is ng, and nothing after the
spare was judged. Exactly one adopt per speaker; one spare unless the candidates ran out.
Output: data/refs/refs.tsv, one row per reference utterance:
  lang, role (dev / eval / aux), use (main / spare), gender, speaker, rank, id, path, text, dur, level_db,
  f0_med, f0_range_st, note
Usage: uv run python scripts/data/08_refs.py
"""
import csv
import sys

from cv3lora.paths import REPO

D = REPO / "data/refs"


def main():
    with open(D / "candidates.tsv") as f:
        cand = list(csv.DictReader(f, delimiter="\t"))
    with open(D / "listening.tsv") as f:
        verdict = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    by = {}
    for r in cand:
        if r["speaker"] == "jecs" and r["lang"] == "en":
            continue
        by.setdefault(r["speaker"], []).append(r)
    known = {f"{r['speaker']}:{r['id']}" for rs in by.values() for r in rs}
    errors = [f"unknown id {k}" for k in verdict if k not in known]
    out = []
    for spk, rs in by.items():
        rs.sort(key=lambda r: int(r["rank"]))
        vs = [verdict.get(f"{spk}:{r['id']}", {}).get("verdict", "") for r in rs]
        heard = [v for v in vs if v]
        if vs[: len(heard)] != heard:
            errors.append(f"{spk}: verdicts not in rank order {vs}")
        if heard.count("adopt") != 1 or heard.count("spare") > 1:
            errors.append(f"{spk}: needs one adopt and at most one spare {vs}")
            continue
        a = heard.index("adopt")
        s = heard.index("spare") if "spare" in heard else None
        if any(v != "ng" for v in heard[:a]) or (s is not None and (s < a or any(v != "ng" for v in heard[a + 1:s]))):
            errors.append(f"{spk}: an earlier passing candidate was skipped {vs}")
        if s is None and len(heard) < len(rs):
            errors.append(f"{spk}: no spare although candidates remain {vs}")
        for i, use in ((a, "main"), (s, "spare")):
            if i is None:
                continue
            rows = [rs[i]]
            if spk == "jecs":
                rows += [c for c in cand if c["speaker"] == "jecs" and c["id"] == rs[i]["pair"]]
            for r in rows:
                out.append({"lang": r["lang"], "role": r["role"], "use": use, "gender": r["gender"], "speaker": spk,
                            "rank": r["rank"], "id": r["id"], "path": r["path"], "text": r["text"], "dur": r["dur"],
                            "level_db": r["level_db"], "f0_med": r["f0_med"], "f0_range_st": r["f0_range_st"],
                            "note": verdict[f"{spk}:{rs[i]['id']}"]["note"]})
    if errors:
        sys.exit("\n".join(errors))
    with open(D / "refs.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)
    print(D / "refs.tsv", len(out), "rows,", len(by), "speakers,", sum(r["use"] == "main" for r in out), "main")


if __name__ == "__main__":
    main()
