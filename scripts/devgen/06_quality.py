"""Speaker cos similarity (ERes2Net) and UTMOSv2 of the processed dev generation audio (research plan §12, §15).

spk_cos: ERes2Net (iic/speech_eres2net_sv_zh-cn_16k-common, as UtterTune) embeddings of the processed reference
  (prep_refs.tsv) and of each file; not for R (no reference). CAM++ is not used: it is the flow's conditioning.
utmos: UTMOSv2 v1.3.0 with its defaults (predict_dataset "sarulab", num_repetitions 1, remove_silent_section).

Output: results/devgen/quality_<cond>.tsv
Usage (eval env): PYTHONPATH=src uv run --project envs/eval python scripts/devgen/06_quality.py --cond A Ap O R
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from cv3lora.paths import REPO

OUT = REPO / "results/devgen"
SV = "iic/speech_eres2net_sv_zh-cn_16k-common"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=["A", "Ap", "C", "D", "Cp", "O", "R"])
    args = ap.parse_args()
    import utmosv2
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    sv = pipeline(task=Tasks.speaker_verification, model=SV)

    def emb(path: str) -> np.ndarray:
        e = np.asarray(sv([sf.read(path, dtype="float32")[0]], output_emb=True)["embs"])[0]
        return e / np.linalg.norm(e)

    with open(OUT / "prep_refs.tsv") as f:
        ref_emb = {r["ref_speaker"]: emb(r["wav16"]) for r in csv.DictReader(f, delimiter="\t")}
    mos_model = utmosv2.create_model(pretrained=True, device="cuda")
    for cond in args.cond:
        with open(OUT / f"prep_{cond}.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        mos = {Path(p["file_path"]).name: p["predicted_mos"]
               for p in mos_model.predict(input_dir=str(Path(rows[0]["wav16"]).parent), batch_size=32, verbose=False)}
        table = []
        for i, r in enumerate(rows):
            cos = float(emb(r["wav16"]) @ ref_emb[r["ref_speaker"]]) if r["ref_speaker"] else float("nan")
            table.append({"cond": cond, "id": r["id"], "seed": r["seed"], "ref_lang": r["ref_lang"],
                          "ref_gender": r["ref_gender"], "ref_speaker": r["ref_speaker"], "spk_cos": f"{cos:.5f}",
                          "utmos": f"{mos[Path(r['wav16']).name]:.4f}"})
            if i % 500 == 0:
                print(cond, i, "/", len(rows), flush=True)
        with open(OUT / f"quality_{cond}.tsv", "w") as f:
            w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
