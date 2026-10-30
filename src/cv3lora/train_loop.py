"""Pieces of the LoRA training loop shared by the pilot (§18.1 step 6) and later training scripts."""
import math
import random

import torch

from cv3lora.lm_loss import length_batches, sequence_loss


def tf_loss(lm, items, max_positions=12000) -> tuple[float, float]:
    """Token-weighted teacher-forced loss and argmax accuracy over items (eval mode, no grad)."""
    was = lm.training
    lm.eval()
    ce = n = hit = 0
    with torch.inference_mode():
        for b in length_batches(items, lambda it: it["tt"].numel() + it["st"].numel() + 3, max_positions):
            c, k, h = sequence_loss(lm, [it["tt"] for it in b], [it["st"] for it in b])
            ce, n, hit = ce + c.sum().item(), n + k.sum().item(), hit + h.sum().item()
    lm.train(was)
    return ce / n, hit / n


def batches(items, bs, seed):
    """Endless stream of batches; each epoch is a fresh shuffle (every utterance equally often, §11)."""
    rng = random.Random(seed)
    while True:
        order = items[:]
        rng.shuffle(order)
        for i in range(0, len(order) - bs + 1, bs):
            yield order[i: i + bs]


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
