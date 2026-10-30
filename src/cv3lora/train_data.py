"""Training / dev-loss utterances: tagged manifest text + canonical speech tokens (research plan §7–§10).

Excluded recordings (exclude.tsv) are dropped; JVS tokens are trimmed with the token spans (§9).
"""
import csv

import torch

from cv3lora.manifest import load_excluded, load_manifest
from cv3lora.paths import REPO
from cv3lora.trim import load_token_spans

UTTS = REPO / "data/splits/utts.tsv"


def load_set(paths: dict, name: str) -> list[dict]:
    """Utterances of split set `name` ("train", "dev", ...) without excluded recordings, sorted by id.

    Each item: id, corpus, text (tagged manifest text), st (speech tokens, trimmed for JVS).
    """
    with open(UTTS) as f:
        ids = {r["id"] for r in csv.DictReader(f, delimiter="\t") if r["set"] == name}
    excluded, spans = load_excluded(), load_token_spans()
    items = []
    for r in load_manifest():
        if r["id"] not in ids or r["id"] in excluded:
            continue
        st = torch.load(paths["token_root"] / f"{r['corpus']}_{r['subset']}" / f"{r['id']}.pt")
        a, b = spans.get(r["id"], (0, st.numel()))
        items.append({"id": r["id"], "corpus": r["corpus"], "text": r["text"], "st": st[a:b]})
    return sorted(items, key=lambda it: it["id"])
