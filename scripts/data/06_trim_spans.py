"""Token slices that trim leading/trailing silence of JVS recordings (research plan §9).

Reads <token_root>/jvs_*/index.tsv and the wavs it lists. Output data/trim/jvs_token_spans.tsv:
  id, n_tokens, start, end (token slice [start, end) to use), lead_s, trail_s (silence before
  trimming, seconds). JSUT is not trimmed and has no rows; loaders use the full token sequence.
Usage: uv run python scripts/data/06_trim_spans.py
"""
import csv

from cv3lora.paths import REPO, load_paths
from cv3lora.trim import speech_bounds, token_span


def main() -> None:
    paths = load_paths()
    out = REPO / "data/trim/jvs_token_spans.tsv"
    out.parent.mkdir(exist_ok=True)
    lines = ["id\tn_tokens\tstart\tend\tlead_s\ttrail_s"]
    for index in sorted(paths["token_root"].glob("jvs_*/index.tsv")):
        with open(index) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                start, end, dur = speech_bounds(r["wav"])
                a, b = token_span(start, end, int(r["n_tokens"]))
                lines.append(f"{r['id']}\t{r['n_tokens']}\t{a}\t{b}\t{start:.2f}\t{dur - end:.2f}")
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} utterances -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
