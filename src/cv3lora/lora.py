"""LoRA on the CosyVoice3 LLM plus the two tag embedding rows (research plan §6, §11).

The tags <PHON_START> / <PHON_END> get ids 151924 / 151925, inside the existing 151,936 embedding rows
(no resize, §3). Only those two rows are trained: they live in a separate 2-row parameter that replaces
the looked-up rows, while the full embedding stays frozen. With Adam (per-element updates) this equals
masking the gradient of the other rows with weight decay 0 on the embedding, and the other rows can not
move. LoRA is injected in place into lm.llm.model, so lm.llm.model.model.embed_tokens (used by
lm_loss and by the official inference) keeps its path.

Requires peft on PYTHONPATH (installed with --target outside the uv project; see the training script).
"""
import torch
from torch import nn

TAGS = ("<PHON_START>", "<PHON_END>")
TAG_IDS = (151924, 151925)


def register_tags(tokenizer) -> None:
    """Add the tags to a CosyVoice3Tokenizer (frontend.tokenizer) in place; checks the ids."""
    tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": list(TAGS)},
                                           replace_additional_special_tokens=False)
    ids = tuple(tokenizer.tokenizer.convert_tokens_to_ids(list(TAGS)))
    assert ids == TAG_IDS, ids


class TagEmbedding(nn.Module):
    """Frozen embedding whose tag rows are taken from a trainable 2-row parameter."""

    def __init__(self, base: nn.Embedding):
        super().__init__()
        self.base = base
        self.tag_rows = nn.Parameter(base.weight[list(TAG_IDS)].detach().clone())

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.base(ids)
        k = ids - TAG_IDS[0]
        is_tag = (k >= 0) & (k < len(TAG_IDS))
        rows = self.tag_rows.to(out.dtype)[k.clamp(0, len(TAG_IDS) - 1)]
        return torch.where(is_tag[..., None], rows, out)


def apply_lora(lm, r: int, alpha: int, dropout: float, targets: list[str]) -> None:
    """Freeze lm, inject LoRA into its Qwen2 model and install the tag rows. Only those train."""
    from peft import LoraConfig, inject_adapter_in_model

    for p in lm.parameters():
        p.requires_grad_(False)
    q = lm.llm.model
    inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets), q)
    q.model.embed_tokens = TagEmbedding(q.model.embed_tokens)
    for n, p in lm.named_parameters():
        # inject_adapter_in_model leaves modules outside the Qwen2 model (speech_embedding, llm_decoder)
        # trainable, hence the explicit freeze above and this explicit selection.
        p.requires_grad_("lora_" in n or n.endswith("tag_rows"))


def trainable_state(lm) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in lm.named_parameters() if p.requires_grad}


def save_adapter(path, lm, config: dict) -> None:
    torch.save({"config": config, "state": trainable_state(lm)}, path)


def load_adapter(lm, path) -> dict:
    """apply_lora with the saved settings, then load the trained values. Returns the saved config."""
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]["lora"]
    apply_lora(lm, c["r"], c["alpha"], c["dropout"], c["targets"])
    params = dict(lm.named_parameters())
    assert set(ck["state"]) == {n for n, p in params.items() if p.requires_grad}
    with torch.no_grad():
        for n, v in ck["state"].items():
            params[n].copy_(v)
    return ck["config"]
