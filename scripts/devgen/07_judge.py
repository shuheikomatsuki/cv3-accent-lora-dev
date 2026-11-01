"""Metrics, bootstrap and the dev generation rules of research plan §12.

Per file: kana CER (whole and outside the target), target reading correct, length ratio of the kana (each for
kw and ctc, cv3lora.eval_audio), speaker cos, UTMOS. Per sentence: mean over the 4 references of a language
x 2 LLM seeds. Paired bootstrap over sentences (10,000, seed 20261101), 95% percentile interval.

--stage 1 (A, Ap, O, R): ASR choice (kana-whisper if the median per-sentence CER of R and of Ap in each language
  is <= 0.15, else ctc) and the final tolerances (delta = max(preset, 2 x half width of the 95% interval of the
  seed 0 - seed 1 difference of the base condition: Ap for target reading, A otherwise; larger of JA/EN))
  -> results/devgen/stage1.json
--stage 2 (adds C, D): the gates with stage1.json -> results/devgen/stage2.json
Both write per-file metrics (metrics.tsv) and the files to listen to (listen.tsv: failures and target-reading
disagreements between kw and ctc).
Failure of (sentence, seed) in a language: stopped at the length limit, tokens outside 0.5-2 x the recording, or
the kana length outside 0.7-1.3 x the reference in 2 or more of the language's 4 references.

Usage (eval env): PYTHONPATH=src uv run --project envs/eval python scripts/devgen/07_judge.py --stage 1
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from cv3lora.eval_audio import align_errors, canon
from cv3lora.paths import REPO

OUT = REPO / "results/devgen"
N_BOOT, BOOT_SEED = 10000, 20261101
ASR_MAX_MEDIAN_CER = 0.15
PRESET = {"spk_cos": 0.02, "utmos": 0.1, "tgt_ok": 0.05, "cer_outside": 0.02, "cer": 0.02}
FAIL_MARGIN = 5
# (metric, higher is better, comparisons (x, y) judged on x - y)
GATES = [("spk_cos", True, [("D", "A"), ("C", "D")]), ("utmos", True, [("D", "A"), ("C", "D")]),
         ("tgt_ok", True, [("C", "Ap")]), ("cer_outside", False, [("C", "D")]), ("cer", False, [("D", "A")])]
RECORD = [(m, x, y) for m in ("spk_cos", "utmos", "cer", "tgt_ok") for x, y in [("C", "Ap"), ("A", "Ap")]]
LANGS = ("ja", "en")


def read(name: str) -> list[dict]:
    with open(OUT / name) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def file_metrics(conds: list[str]) -> list[dict]:
    inp = {r["id"]: r for r in read("inputs.tsv")}
    rows = []
    for cond in conds:
        qual = {(r["id"], r["seed"], r["ref_speaker"]): r for r in read(f"quality_{cond}.tsv")}
        for a in read(f"asr_{cond}.tsv"):
            it = inp[a["id"]]
            ref, span = canon(it["ref_kana"]), (int(it["tgt_start"]), int(it["tgt_end"]))
            q = qual[(a["id"], a["seed"], a["ref_speaker"])]
            row = {"cond": cond, "id": a["id"], "seed": a["seed"], "lang": a["ref_lang"] or "rec",
                   "ref_speaker": a["ref_speaker"], "ref_gender": q["ref_gender"], "spk_cos": float(q["spk_cos"]),
                   "utmos": float(q["utmos"])}
            for asr in ("kw", "ctc"):
                hyp = canon(a[asr])
                e = align_errors(ref, hyp, span)
                row |= {f"cer_{asr}": e["cer"], f"cer_outside_{asr}": e["cer_outside"],
                        f"tgt_ok_{asr}": float(e["target_ok"]), f"len_ratio_{asr}": len(hyp) / len(ref), f"hyp_{asr}": hyp}
            rows.append(row)
    return rows


def per_sentence(rows, cond, lang, metric, seed=None) -> dict[str, float]:
    acc = defaultdict(list)
    for r in rows:
        if r["cond"] == cond and r["lang"] == lang and (seed is None or r["seed"] == seed):
            acc[r["id"]].append(r[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def boot(diff: dict[str, float]) -> dict:
    v = np.array([diff[k] for k in sorted(diff)])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(v), (N_BOOT, len(v)))
    m = v[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": len(v), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def paired(rows, metric, x, y, lang) -> dict:
    a, b = per_sentence(rows, x, lang, metric), per_sentence(rows, y, lang, metric)
    return boot({k: a[k] - b[k] for k in a.keys() & b.keys()})


def failures(rows, cond, lang, asr) -> list[tuple[str, str]]:
    tok = {(t["id"], t["seed"]): t for t in read(f"tokens_{cond}.tsv")}
    bad_len = defaultdict(int)
    for r in rows:
        if r["cond"] == cond and r["lang"] == lang and not 0.7 <= r[f"len_ratio_{asr}"] <= 1.3:
            bad_len[(r["id"], r["seed"])] += 1
    out = []
    for key, t in tok.items():
        reasons = [s for s, bad in [("max_len", t["stop"] == "max_len"),
                                    ("tokens", not 0.5 <= float(t["ratio_to_rec"]) <= 2.0),
                                    ("kana_len", bad_len[key] >= 2)] if bad]
        if reasons:
            out.append((*key, "+".join(reasons)))
    return out


def write_rows(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def listen_rows(rows, conds, asr) -> list[dict]:
    out = []
    for cond in conds:
        if cond in ("O", "R"):
            continue
        for lang in LANGS:
            for i, s, why in failures(rows, cond, lang, asr):
                out.append({"cond": cond, "lang": lang, "id": i, "seed": s, "ref_speaker": "", "why": why})
    for r in rows:
        if r["cond"] in ("Ap", "C") and r["tgt_ok_kw"] != r["tgt_ok_ctc"]:
            out.append({"cond": r["cond"], "lang": r["lang"], "id": r["id"], "seed": r["seed"],
                        "ref_speaker": r["ref_speaker"], "why": f"target kw={int(r['tgt_ok_kw'])} ctc={int(r['tgt_ok_ctc'])}"})
    return out


def stage1(rows) -> dict:
    med = {"R": float(np.median(list(per_sentence(rows, "R", "rec", "cer_kw").values())))}
    med |= {f"Ap_{l}": float(np.median(list(per_sentence(rows, "Ap", l, "cer_kw").values()))) for l in LANGS}
    med_ctc = {"R": float(np.median(list(per_sentence(rows, "R", "rec", "cer_ctc").values())))}
    med_ctc |= {f"Ap_{l}": float(np.median(list(per_sentence(rows, "Ap", l, "cer_ctc").values()))) for l in LANGS}
    asr = "kw" if max(med.values()) <= ASR_MAX_MEDIAN_CER else "ctc"
    delta, noise = {}, {}
    for metric, preset in PRESET.items():
        col = metric if metric in ("spk_cos", "utmos") else f"{metric}_{asr}"
        base = "Ap" if metric == "tgt_ok" else "A"
        hw = {}
        for l in LANGS:
            s0, s1 = per_sentence(rows, base, l, col, "0"), per_sentence(rows, base, l, col, "1")
            b = boot({k: s0[k] - s1[k] for k in s0})
            hw[l] = (b["hi"] - b["lo"]) / 2
            noise[f"{metric}_{l}"] = b
        delta[metric] = max(preset, 2 * max(hw.values()))
    fails = {f"{c}_{l}": len(failures(rows, c, l, asr)) for c in ("A", "Ap") for l in LANGS}
    means = {f"{c}_{l}_{m}": float(np.mean(list(per_sentence(rows, c, l, m).values())))
             for c in ("A", "Ap", "O") for l in LANGS for m in ("spk_cos", "utmos", "cer_kw", "cer_ctc", "tgt_ok_kw")}
    means |= {f"R_{m}": float(np.mean(list(per_sentence(rows, "R", "rec", m).values())))
              for m in ("utmos", "cer_kw", "cer_ctc", "tgt_ok_kw")}
    return {"asr": asr, "median_cer_kw": med, "median_cer_ctc": med_ctc, "asr_max_median_cer": ASR_MAX_MEDIAN_CER,
            "preset": PRESET, "delta": delta, "seed_noise": noise, "failures": fails, "means": means}


def stage2(rows, s1) -> dict:
    asr, delta = s1["asr"], s1["delta"]
    gates, ok = [], True
    for metric, higher, comps in GATES:
        col = metric if metric in ("spk_cos", "utmos") else f"{metric}_{asr}"
        for x, y in comps:
            for l in LANGS:
                b = paired(rows, col, x, y, l)
                passed = b["lo"] > -delta[metric] if higher else b["hi"] < delta[metric]
                ok &= passed
                gates.append({"metric": metric, "diff": f"{x}-{y}", "lang": l, **b, "delta": delta[metric], "pass": passed})
    for x, y in [("C", "Ap"), ("D", "A")]:
        for l in LANGS:
            nx, ny = len(failures(rows, x, l, asr)), len(failures(rows, y, l, asr))
            passed = nx <= ny + FAIL_MARGIN
            ok &= passed
            gates.append({"metric": "failures", "diff": f"{x} vs {y}", "lang": l, "n_x": nx, "n_y": ny,
                          "margin": FAIL_MARGIN, "pass": passed})
    record = []
    for metric, x, y in RECORD:
        col = metric if metric in ("spk_cos", "utmos") else f"{metric}_{asr}"
        for l in LANGS:
            record.append({"metric": metric, "diff": f"{x}-{y}", "lang": l, **paired(rows, col, x, y, l)})
    by_gender = {f"{c}_{l}_{g}_{m}": float(np.mean([r[m] for r in rows if r["cond"] == c and r["lang"] == l
                                                     and r["ref_gender"] == g]))
                 for c in ("A", "D", "C") for l in LANGS for g in ("F", "M") for m in ("spk_cos", "utmos")}
    return {"asr": asr, "delta": delta, "pass": ok, "gates": gates, "record": record, "by_gender": by_gender}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()
    conds = ["A", "Ap", "O", "R"] + (["C", "D"] if args.stage == 2 else [])
    rows = file_metrics(conds)
    write_rows(f"metrics_stage{args.stage}.tsv", [{k: f"{v:.5f}" if isinstance(v, float) else v for k, v in r.items()}
                                                   for r in rows])
    if args.stage == 1:
        res = stage1(rows)
    else:
        res = stage2(rows, json.loads((OUT / "stage1.json").read_text()))
    write_rows(f"listen_stage{args.stage}.tsv", listen_rows(rows, conds, res["asr"]) or
               [{"cond": "", "lang": "", "id": "", "seed": "", "ref_speaker": "", "why": ""}])
    (OUT / f"stage{args.stage}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("seed_noise", "means", "record", "by_gender")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
