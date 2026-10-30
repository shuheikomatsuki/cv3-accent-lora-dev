"""cross_lingual synthesis split into its two halves (research plan §2.1, §13.3).

gen_tokens: LLM only (does not see the reference), once per system x text x seed; the tokens are saved.
token2wav: flow + HiFT for saved tokens and one reference. The CFM noise of the flow is a fixed buffer
(seeded at construction), HiFT draws noise, so the seed is set before each call.
Both follow the official CosyVoice3Model.llm_job / tts (non-stream) code paths.
"""
import uuid

import torch

from cv3lora.lm_loss import text_tokens

SILENT_RUN = 5  # llm_job drops silent/breath tokens after this many in a row


def gen_tokens(cv, text: str, seed: int) -> dict:
    """Speech tokens for PREFIX + text. stop = "eos" or "max_len" (official max = 20 x text tokens)."""
    m, lm = cv.model, cv.model.llm
    tt = text_tokens(cv.frontend, text)[None].to(m.device)
    empty = torch.zeros(1, 0, dtype=torch.int32, device=m.device)
    n = torch.tensor([tt.shape[1]], dtype=torch.int32, device=m.device)
    max_len = int(tt.shape[1] * 20)  # CosyVoice3LM.inference max_token_text_ratio
    torch.manual_seed(seed)
    out, raw, run = [], 0, 0
    with torch.cuda.amp.autocast(m.fp16):
        for t in lm.inference(text=tt, text_len=n, prompt_text=empty, prompt_text_len=torch.zeros_like(n),
                              prompt_speech_token=empty, prompt_speech_token_len=torch.zeros_like(n),
                              embedding=torch.zeros(0, 192, device=m.device)):
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
