"""Speaker-information diagnostic, step 1: per-utterance teacher-forced loss (research plan §15.3).

Set A: JVS nonpara30 utterances whose sentence is not from BASIC5000 (different sentences per speaker).
Set B: the 5 parallel sentences held out of training (data/splits, diag_b), all JVS speakers + JSUT.
Text is the plain transcript (no tags: the stock model has no tag tokens).

Output: results/speaker_diag/loss_<tag>.jsonl (id, set, speaker, nll_sum, n, n_correct; sorted by id)
        and loss_<tag>.provenance.json. n = speech tokens + eos.
Usage: uv run python scripts/analysis/speaker_diag_loss.py --tag base
"""
import argparse
import csv
import json
import time
from datetime import datetime

import torch

from cv3lora.cosyvoice_loader import cosyvoice_commit, load_cosyvoice3
from cv3lora.lm_loss import PREFIX, length_batches, sequence_loss, text_tokens
from cv3lora.manifest import read_transcript
from cv3lora.paths import REPO, load_paths

OUT = REPO / "results/speaker_diag"


def diag_utterances(paths: dict) -> list[dict]:
    items = []
    for d in sorted(paths["jvs_root"].glob("jvs[0-9][0-9][0-9]")):
        # transcripts_utf8.txt lists more sentences than were recorded; go by the wav files
        text_of = read_transcript(d / "nonpara30/transcripts_utf8.txt")
        for fid in sorted(w.stem for w in (d / "nonpara30/wav24kHz16bit").glob("*.wav")):
            text = text_of[fid]
            if not fid.startswith("BASIC5000"):
                items.append({"id": f"{d.name}_nonpara30_{fid}", "set": "A", "speaker": d.name,
                              "text": text, "tok": f"jvs_nonpara30/{d.name}_nonpara30_{fid}.pt"})
    with open(REPO / "data/splits/utts.tsv") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["diag_b"] == "1"]
    jsut = read_transcript(paths["jsut_root"] / "voiceactress100/transcript_utf8.txt")
    for r in rows:
        if r["corpus"] == "jsut":
            text, spk = jsut[r["file_id"]], "jsut"
        else:
            spk = r["speaker"]
            text = read_transcript(paths["jvs_root"] / spk / "parallel100/transcripts_utf8.txt")[r["file_id"]]
        items.append({"id": r["id"], "set": "B", "speaker": spk, "text": text,
                      "tok": f"{r['corpus']}_{r['subset']}/{r['id']}.pt"})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-positions", type=int, default=12000)
    args = ap.parse_args()
    paths = load_paths()

    cv = load_cosyvoice3()
    lm, fe = cv.model.llm, cv.frontend
    del cv.model.flow, cv.model.hift
    torch.cuda.empty_cache()
    lm.eval()

    items = diag_utterances(paths)
    for it in items:
        it["tt"] = text_tokens(fe, it["text"])
        it["st"] = torch.load(paths["token_root"] / it["tok"])
    batches = length_batches(items, lambda it: it["tt"].numel() + it["st"].numel() + 3,
                             args.max_positions)
    t0 = time.time()
    rows = []
    with torch.inference_mode():
        for b in batches:
            ce, n, hit = sequence_loss(lm, [it["tt"] for it in b], [it["st"] for it in b])
            for it, c, k, h in zip(b, ce.tolist(), n.tolist(), hit.tolist()):
                rows.append({"id": it["id"], "set": it["set"], "speaker": it["speaker"],
                             "nll_sum": c, "n": k, "n_correct": h})
    rows.sort(key=lambda r: r["id"])
    (OUT / f"loss_{args.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    split = json.loads((REPO / "data/splits/split.json").read_text())
    (OUT / f"loss_{args.tag}.provenance.json").write_text(json.dumps({
        "tag": args.tag, "model_dir": str(paths["cosyvoice_model"]),
        "cosyvoice_commit": cosyvoice_commit(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "dtype": "fp32", "prefix": PREFIX,
        "token_root": str(paths["token_root"]), "split_seed": split["seed"],
        "n_utts": len(rows), "compute_s": round(time.time() - t0, 1),
        "date": datetime.now().isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} utterances, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
