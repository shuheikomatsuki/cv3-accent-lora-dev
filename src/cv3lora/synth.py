"""cross_lingual synthesis split into its two halves (research plan §2.1, §13.3); zero_shot as a side check.

gen_tokens: LLM only (does not see the reference), once per system x text x seed; the tokens are saved.
token2wav: flow + HiFT for saved tokens and one reference. The CFM noise of the flow is a fixed buffer
(seeded at construction), HiFT draws noise, so the seed is set before each call.
Both follow the official CosyVoice3Model.llm_job / tts (non-stream) code paths.
"""
import uuid

import torch

from cv3lora.lm_loss import PREFIX, text_tokens

SILENT_RUN = 5  # llm_job drops silent/breath tokens after this many in a row


def gen_tokens(cv, text: str, seed: int, prompt: dict | None = None) -> dict:
    """Speech tokens. stop = "eos" or "max_len" (official max = 20 x text tokens, prompt text not counted).

    prompt None: cross_lingual, the LLM input is PREFIX + text (no reference).
    prompt from zero_shot_prompt: zero_shot, the LLM sees PREFIX + reference transcript and the reference
    speech tokens; text goes in without PREFIX (official CosyVoice3 zero_shot usage, example.py).
    """
    m, lm = cv.model, cv.model.llm
    if prompt is None:
        tt = text_tokens(cv.frontend, text)[None].to(m.device)
        pt = ps = torch.zeros(1, 0, dtype=torch.int32, device=m.device)
        emb = torch.zeros(0, 192, device=m.device)
    else:
        tt = cv.frontend._extract_text_token(text)[0].to(m.device)
        pt, ps = prompt["prompt_text"].to(m.device), prompt["llm_prompt_speech_token"].to(m.device)
        emb = prompt["llm_embedding"].to(m.device)
    n = torch.tensor([tt.shape[1]], dtype=torch.int32, device=m.device)
    max_len = int(tt.shape[1] * 20)  # CosyVoice3LM.inference max_token_text_ratio
    torch.manual_seed(seed)
    out, raw, run = [], 0, 0
    with torch.cuda.amp.autocast(m.fp16):
        for t in lm.inference(text=tt, text_len=n, prompt_text=pt,
                              prompt_text_len=torch.tensor([pt.shape[1]], dtype=torch.int32, device=m.device),
                              prompt_speech_token=ps,
                              prompt_speech_token_len=torch.tensor([ps.shape[1]], dtype=torch.int32, device=m.device),
                              embedding=emb):
            raw += 1
            if t in m.silent_tokens:
                run += 1
                if run > SILENT_RUN:
                    continue
            else:
                run = 0
            out.append(t)
    return {"tokens": torch.tensor(out, dtype=torch.int32), "n_raw": raw, "max_len": max_len,
            "stop": "max_len" if raw >= max_len else "eos", "n_text": tt.shape[1]}


def prompt_inputs(cv, wav_path) -> dict:
    """Flow-side reference features, as frontend_cross_lingual makes them."""
    mi = cv.frontend.frontend_cross_lingual("", str(wav_path), cv.sample_rate, "")
    return {k: mi[k] for k in ("flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding")}


def zero_shot_prompt(cv, wav_path, ref_text: str) -> dict:
    """LLM- and flow-side reference inputs, as frontend_zero_shot makes them (prompt text = PREFIX + transcript)."""
    return cv.frontend.frontend_zero_shot("", PREFIX + ref_text, str(wav_path), cv.sample_rate, "")


def token2wav(cv, tokens: torch.Tensor, prompt: dict, seed: int) -> torch.Tensor:
    """Waveform [1, samples] at cv.sample_rate for 1-D tokens with one reference."""
    m, u = cv.model, str(uuid.uuid1())
    m.hift_cache_dict[u] = None
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            return m.token2wav(token=tokens[None], prompt_token=prompt["flow_prompt_speech_token"],
                               prompt_feat=prompt["prompt_speech_feat"], embedding=prompt["flow_embedding"],
                               token_offset=0, uuid=u, finalize=True).cpu()
    finally:
        m.hift_cache_dict.pop(u)
