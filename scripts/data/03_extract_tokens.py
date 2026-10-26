"""Extract CosyVoice3 speech tokens for JVS and JSUT (research plan §10).

One file per utterance: <token_root>/<subset>/<speaker>_<name>_<file_id>.pt (e.g.
jvs001_nonpara30_BASIC5000_0025.pt, jsut_basic5000_BASIC5000_0001.pt) holding an int32
tensor [T]. Speaker is "jsut" for JSUT. Existing files are skipped (resumable). After a subset is
done, index.tsv (id, wav, n_tokens, sha1 of the token bytes) and meta.json are rewritten.

Usage: uv run python scripts/data/03_extract_tokens.py [subset ...] [--limit N]
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import onnxruntime
import torch

from cv3lora.cosyvoice_loader import cosyvoice_commit, load_cosyvoice3
from cv3lora.paths import load_paths

SUBSETS = ["jvs_parallel100", "jvs_nonpara30", "jsut_basic5000", "jsut_voiceactress100"]


def list_wavs(subset: str, paths: dict) -> list[tuple[str, Path]]:
    corpus, name = subset.split("_", 1)
    if corpus == "jvs":
        wavs = sorted(paths["jvs_root"].glob(f"jvs[0-9][0-9][0-9]/{name}/wav24kHz16bit/*.wav"))
        return [(f"{w.parents[2].name}_{name}_{w.stem}", w) for w in wavs]
    wavs = sorted((paths["jsut_root"] / name / "wav").glob("*.wav"))
    return [(f"jsut_{name}_{w.stem}", w) for w in wavs]


def write_index(out: Path, items: list[tuple[str, Path]]) -> None:
    lines = ["id\twav\tn_tokens\tsha1"]
    for uid, wav in items:
        tok = torch.load(out / f"{uid}.pt")
        lines.append(f"{uid}\t{wav}\t{tok.numel()}\t{hashlib.sha1(tok.numpy().tobytes()).hexdigest()}")
    (out / "index.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="*", default=SUBSETS, choices=SUBSETS)
    ap.add_argument("--limit", type=int, help="first N utterances per subset (for testing)")
    args = ap.parse_args()

    paths = load_paths()
    fe = load_cosyvoice3().frontend
    meta = {
        "model_dir": str(paths["cosyvoice_model"]),
        "cosyvoice_commit": cosyvoice_commit(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "providers": fe.speech_tokenizer_session.get_providers(),
        "gpu": torch.cuda.get_device_name(0),
    }
    for subset in args.subsets:
        items = list_wavs(subset, paths)[: args.limit]
        out = paths["token_root"] / subset
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for n, (uid, wav) in enumerate(items):
            p = out / f"{uid}.pt"
            if not p.exists():
                tok, _ = fe._extract_speech_token(str(wav))
                torch.save(tok[0].int().cpu(), p)
            if n % 500 == 0:
                print(subset, n, len(items), f"{time.time() - t0:.0f}s", flush=True)
        write_index(out, items)
        (out / "meta.json").write_text(json.dumps(
            dict(meta, subset=subset, n_utts=len(items), updated=datetime.now().isoformat(timespec="seconds")),
            indent=2, ensure_ascii=False) + "\n")
        print(subset, "done", len(items), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
