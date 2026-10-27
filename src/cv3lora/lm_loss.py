"""Teacher-forced speech-token loss of the CosyVoice3 LLM (dev loss and speaker diagnostic).

Sequence = cross_lingual inference input = official CosyVoice3LM unistream training sequence:
  [sos][embed(PREFIX + text)][task_id][speech tokens]  -> targets: speech tokens + eos
No prompt audio and no instruct part.
"""
import torch
import torch.nn.functional as F

PREFIX = "You are a helpful assistant.<|endofprompt|>"


def text_tokens(frontend, text: str) -> torch.Tensor:
    tok, _ = frontend._extract_text_token(PREFIX + text)
    return tok[0].cpu()


def length_batches(items: list, length, max_positions: int) -> list[list]:
    """Sort by length and group so that (batch size x longest sequence) stays under max_positions."""
    items = sorted(items, key=length)
    batches, cur = [], []
    for it in items:
        if cur and length(it) * (len(cur) + 1) > max_positions:
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    return batches


def sequence_loss(lm, text_toks: list[torch.Tensor], speech_toks: list[torch.Tensor]):
    """Per-sequence (sum of cross-entropy in nats, number of targets, number of argmax hits).

    The sums are differentiable unless called under torch.inference_mode().
    """
    dev = lm.speech_embedding.weight.device
    emb_text = lm.llm.model.model.embed_tokens
    sos = lm.speech_embedding.weight[lm.sos][None]
    task = lm.speech_embedding.weight[lm.task_id][None]
    eos = torch.tensor([lm.eos_token], device=dev)
    xs, ys = [], []
    for tt, st in zip(text_toks, speech_toks):
        tt, st = tt.to(dev), st.long().to(dev)
        x = torch.cat([sos, emb_text(tt), task, lm.speech_embedding(st)], 0)
        n_ctx = x.shape[0] - st.numel() - 1
        y = torch.cat([torch.full((n_ctx,), -100, device=dev), st, eos])
        xs.append(x)
        ys.append(y)
    T = max(x.shape[0] for x in xs)
    X = xs[0].new_zeros(len(xs), T, xs[0].shape[1])
    Y = torch.full((len(xs), T), -100, device=dev, dtype=torch.long)
    M = torch.zeros(len(xs), T, device=dev, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, : len(x)] = x
        Y[i, : len(y)] = y
        M[i, : len(x)] = 1
    h = lm.llm.model.model(inputs_embeds=X, attention_mask=M, return_dict=True).last_hidden_state
    sel = Y != -100
    logits = lm.llm_decoder(h[sel]).float()
    ce = F.cross_entropy(logits, Y[sel], reduction="none")
    hit = (logits.argmax(-1) == Y[sel]).long()
    row = sel.nonzero()[:, 0]
    n = sel.sum(1)
    ce_sum = torch.zeros(len(xs), device=dev).index_add(0, row, ce)
    n_hit = torch.zeros(len(xs), device=dev, dtype=torch.long).index_add(0, row, hit)
    return ce_sum, n, n_hit
