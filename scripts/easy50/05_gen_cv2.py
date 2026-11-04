"""CosyVoice 2 side of the easy_50 2x2 (model {CV2, CV3} x {base, +LoRA}), all cross_lingual.

Conditions (inputs from results/easy50/inputs.tsv):
  CV2A  CosyVoice2-0.5B base, plain text (column A = easy_50 vanilla)
  UT    CosyVoice2-0.5B + UtterTune-CosyVoice2-ja-JSUTJVS LoRA, phon_tagged (column C), UtterTune's Test Set 1 input
cross_lingual as in the official CosyVoice2 code: the LLM sees the target text only (no prefix for CV2), the
reference enters the flow only (frontend_cross_lingual). Tokens once per sentence (LLM seed 0) through
CosyVoice2Model.llm_job (official silent-token rule); flow + HiFT with each JA dev reference through
CosyVoice2Model.token2wav (non-stream), seed 0 before each call. No text normalization (text tokens built directly);
wetext's Normalizer is stubbed because helen has no wetext model (never called).

Environment: /workH/kmt/pre_exps/1019_uttertune_eval_cmp/.venv_ut (torch 2.7.1+cu128, peft 0.17.0; built for helen).
Output: <work_root>/easy50/{tokens,wav}/<cond>/..., results/easy50/tokens_<cond>.tsv, wav_<cond>.tsv
        (same columns as 02_gen.py; _smoke suffix with --limit)
Usage (from the repo root):
  /workH/kmt/pre_exps/1019_uttertune_eval_cmp/.venv_ut/bin/python scripts/easy50/05_gen_cv2.py --cond CV2A UT [--limit 2]
"""
import argparse
import csv
import hashlib
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

REPO = Path(__file__).resolve().parents[2]
UT_ROOT = Path("/workH/kmt/gr/models/UtterTune")
MODEL = UT_ROOT / "pretrained_models/CosyVoice2-0.5B"
LORA = UT_ROOT / "lora_weights/UtterTune-CosyVoice2-ja-JSUTJVS"
sys.path[:0] = [str(UT_ROOT), str(UT_ROOT / "submodules/CosyVoice"),
                str(UT_ROOT / "submodules/CosyVoice/third_party/Matcha-TTS")]

from scripts.cv2.patch import apply_patch  # noqa: E402  (UtterTune repo)

apply_patch()
import wetext  # noqa: E402

wetext.Normalizer.__init__ = lambda self, *a, **k: None
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402
from scripts.cv2.infer import load_lora_adapter  # noqa: E402

OUT = REPO / "results/easy50"
TEXT = {"CV2A": "A", "UT": "C"}
SEED, FLOW_SEED, TOKEN_HZ = 0, 0, 25


def seed_all(n: int) -> None:
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def gen_tokens(cv, text: str) -> torch.Tensor:
    m, u = cv.model, str(uuid.uuid1())
    tt, _ = cv.frontend._extract_text_token(text)
    m.tts_speech_token_dict[u], m.llm_end_dict[u] = [], False
    seed_all(SEED)
    m.llm_job(tt, torch.zeros(1, 0, dtype=torch.int32), torch.zeros(1, 0, dtype=torch.int32), torch.zeros(0, 192), u)
    toks = torch.tensor(m.tts_speech_token_dict.pop(u), dtype=torch.int32)
    m.llm_end_dict.pop(u)
    return toks


def token2wav(cv, tokens: torch.Tensor, p: dict) -> torch.Tensor:
    m, u = cv.model, str(uuid.uuid1())
    m.hift_cache_dict[u] = None
    seed_all(FLOW_SEED)
    try:
        with torch.inference_mode():
            return m.token2wav(token=tokens[None], prompt_token=p["flow_prompt_speech_token"],
                               prompt_feat=p["prompt_speech_feat"], embedding=p["flow_embedding"],
                               token_offset=0, uuid=u, finalize=True).cpu()
    finally:
        m.hift_cache_dict.pop(u)


def write(name: str, table: list[dict]) -> None:
    with open(OUT / name, "w") as f:
        w = csv.DictWriter(f, list(table[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True, choices=list(TEXT))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    work = Path(yaml.safe_load(open(REPO / "configs/paths.local.yaml"))["work_root"])
    with open(OUT / "inputs.tsv") as f:
        items = list(csv.DictReader(f, delimiter="\t"))[: args.limit]
    with open(REPO / "data/refs/refs.tsv") as f:
        refs = [r for r in csv.DictReader(f, delimiter="\t")
                if r["role"] == "dev" and r["use"] == "main" and r["lang"] == "ja"]
    assert len(refs) == 4, len(refs)
    suffix = "_smoke" if args.limit else ""
    for cond in args.cond:  # base first; the LoRA wraps the LLM in place, so each condition loads its own model
        cv = CosyVoice2(model_dir=str(MODEL), fp16=False)
        model = "CosyVoice2-0.5B"
        if cond == "UT":
            cv.frontend.tokenizer = load_lora_adapter(cv, str(MODEL), LORA, cv.model.device)
            model += " + " + LORA.name
            for t in ("<PHON_START>", "<PHON_END>"):
                assert len(cv.frontend.tokenizer.encode(t)) == 1, t
        cv.model.llm.eval()
        prompts = {r["speaker"]: cv.frontend.frontend_cross_lingual("", load_wav(r["path"], 16000), cv.sample_rate, "")
                   for r in refs}
        tok_dir, wav_dir = (work / "easy50" / d / cond for d in ("tokens", "wav"))
        tok_dir.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)
        tok_table, wav_table = [], []
        for it in items:
            text = it[TEXT[cond]]
            tokens = gen_tokens(cv, text)
            n = tokens.numel()
            n_text = len(cv.frontend._extract_text_token(text)[0][0])
            tpath = tok_dir / f"{it['id']}_s{SEED}.pt"
            torch.save(tokens, tpath)
            tok_table.append({"cond": cond, "model": model, "id": it["id"], "seed": SEED, "n_text": n_text,
                              "n_tokens": n, "max_len": n_text * 20, "stop": "max_len" if n >= n_text * 20 else "eos",
                              "sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), "path": str(tpath)})
            for r in refs:
                wav = token2wav(cv, tokens, prompts[r["speaker"]])
                path = wav_dir / f"{it['id']}_s{SEED}__{r['lang']}_{r['speaker']}.wav"
                torchaudio.save(str(path), wav, cv.sample_rate, encoding="PCM_S", bits_per_sample=16)
                wav_table.append({"cond": cond, "id": it["id"], "seed": SEED, "ref_lang": r["lang"],
                                  "ref_gender": r["gender"], "ref_speaker": r["speaker"], "ref_path": r["path"],
                                  "n_tokens": n, "dur_s": f"{wav.shape[1] / cv.sample_rate:.3f}",
                                  "expected_s": f"{n / TOKEN_HZ:.3f}", "flow_seed": FLOW_SEED, "wav": str(path)})
            print(cond, it["id"], n, flush=True)
        write(f"tokens_{cond}{suffix}.tsv", tok_table)
        write(f"wav_{cond}{suffix}.tsv", wav_table)
        del cv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
