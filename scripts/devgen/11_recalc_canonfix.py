"""Rebuild the dev tables with the fixed canon() (2026-11-06) from the saved ASR and quality tables, on CPU.

Runs 07_judge.py (stages 1, 2), 09_mode_summary.py and the C' judge (review/devgen_cp/judge_cp.py) unchanged, reading
from results/devgen and writing to results/devgen_canonfix, so the tables and judgements of results/devgen stay as
recorded. Tolerances in stage1.json are recomputed with the fixed canon(). Also writes summary.tsv: mean CER over all
files, mean with CER > 0.5 dropped (plan §15.1) and target reading, per condition and reference language.

Usage (eval env): PYTHONPATH=src envs/eval/.venv/bin/python scripts/devgen/11_recalc_canonfix.py
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from cv3lora.paths import REPO

SRC, OUT = REPO / "results/devgen", REPO / "results/devgen_canonfix"
CP_JUDGE = Path("/workH/kmt/gr/review/devgen_cp/judge_cp.py")
PATCH = 'J.OUT = Path(OUT_DIR)\nJ.read = lambda name: list(csv.DictReader(open(Path(SRC_DIR) / name), delimiter="\\t"))\n'


def read_src(name: str) -> list[dict]:
    with open(SRC / name) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load(path: Path):
    """Import a numbered script with its reads from SRC and its writes to OUT."""
    spec = importlib.util.spec_from_file_location(path.stem[3:], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT, mod.read = OUT, read_src
    return mod


def run_cp_judge() -> None:
    src = CP_JUDGE.read_text()
    src = src.replace('HERE = Path(__file__).parent', 'HERE = Path(OUT_DIR)')
    src = src.replace('(HERE / "criteria.md.sha256")', f'Path("{CP_JUDGE.parent}/criteria.md.sha256")')
    anchor = "spec.loader.exec_module(J)\n"
    assert anchor in src and "HERE = Path(OUT_DIR)" in src
    exec(compile(src.replace(anchor, anchor + PATCH), str(CP_JUDGE), "exec"),
         {"__name__": "judge_cp", "OUT_DIR": str(OUT), "SRC_DIR": str(SRC)})


def summary(rows: list[dict]) -> None:
    out = []
    for cond in sorted({r["cond"] for r in rows}):
        for lang in sorted({r["lang"] for r in rows if r["cond"] == cond}):
            sel = [r for r in rows if r["cond"] == cond and r["lang"] == lang]
            cer = np.array([float(r["cer_kw"]) for r in sel])
            kept = cer[cer <= 0.5]
            out.append({"cond": cond, "lang": lang, "n": len(cer), "cer_all": f"{cer.mean():.4f}",
                        "cer_excl_gt0.5": f"{kept.mean():.4f}", "n_excluded": int((cer > 0.5).sum()),
                        "tgt_ok_kw": f"{np.mean([float(r['tgt_ok_kw']) for r in sel]):.4f}",
                        "tgt_ok_ctc": f"{np.mean([float(r['tgt_ok_ctc']) for r in sel]):.4f}",
                        "cer_all_ctc": f"{np.mean([float(r['cer_ctc']) for r in sel]):.4f}"})
    with open(OUT / "summary.tsv", "w") as f:
        w = csv.DictWriter(f, list(out[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    judge = load(REPO / "scripts/devgen/07_judge.py")
    for stage, conds in ((1, ["A", "Ap", "O", "R"]), (2, ["A", "Ap", "O", "R", "C", "D"])):
        rows = judge.file_metrics(conds)
        judge.write_rows(f"metrics_stage{stage}.tsv",
                         [{k: f"{v:.5f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows])
        res = judge.stage1(rows) if stage == 1 else judge.stage2(rows, json.loads((OUT / "stage1.json").read_text()))
        judge.write_rows(f"listen_stage{stage}.tsv", judge.listen_rows(rows, conds, res["asr"]) or
                         [{"cond": "", "lang": "", "id": "", "seed": "", "ref_speaker": "", "why": ""}])
        (OUT / f"stage{stage}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    load(REPO / "scripts/devgen/09_mode_summary.py").main()
    run_cp_judge()
    with open(OUT / "metrics_cp.tsv") as f:
        cp_rows = [r for r in csv.DictReader(f, delimiter="\t") if r["cond"] == "Cp"]
    with open(OUT / "metrics_stage2.tsv") as f:
        summary(list(csv.DictReader(f, delimiter="\t")) + cp_rows)


if __name__ == "__main__":
    main()
