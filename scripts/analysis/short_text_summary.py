"""Summarize short_text_probe.py: collapse by sentence length, carrier length and the eos patch.

Output: results/short_text/summary.tsv (printed too)
Usage: uv run python scripts/analysis/short_text_summary.py
"""
import csv
import statistics as st

from cv3lora.paths import REPO

OUT = REPO / "results/short_text"
BINS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 999)]


def rate(xs, f):
    return f"{sum(map(f, xs))}/{len(xs)}"


def main() -> None:
    rows = list(csv.DictReader(open(OUT / "probe.tsv"), delimiter="\t"))
    rec = {r["key"]: float(r["cer"]) for r in csv.DictReader(open(OUT / "recordings.tsv"), delimiter="\t")}
    out = []
    for patch in ("0", "1"):
        for lo, hi in BINS:
            g = [r for r in rows if r["set"] == "dev" and r["patch"] == patch and lo <= int(r["n_chars"]) < hi]
            if not g:
                continue
            out.append({"group": f"dev {lo}-{hi - 1 if hi < 999 else ''}字", "patch": patch, "n": len(g),
                        "collapse": rate(g, lambda r: r["collapse"] == "1"),
                        "short(<0.6)": rate(g, lambda r: float(r["ratio_to_rec"]) < 0.6),
                        "cer>0.5": rate(g, lambda r: float(r["cer"]) > 0.5),
                        "median_cer": f"{st.median(float(r['cer']) for r in g):.3f}",
                        "median_cer_rec": f"{st.median(rec[r['key']] for r in g):.3f}",
                        "median_ratio_or_s": f"{st.median(float(r['ratio_to_rec']) for r in g):.2f}",
                        "word_found": ""})
        for c in ("c0", "c1", "c2", "c3"):
            g = [r for r in rows if r["set"] == c and r["patch"] == patch]
            word = lambda r: ("はし" if "はし" in r["kana"] else "はな") in r["hyp"]  # noqa: E731
            out.append({"group": f"carrier {c}（{len(g[0]['text']) }字）", "patch": patch, "n": len(g),
                        "collapse": rate(g, lambda r: r["collapse"] == "1"), "short(<0.6)": "",
                        "cer>0.5": rate(g, lambda r: float(r["cer"]) > 0.5),
                        "median_cer": f"{st.median(float(r['cer']) for r in g):.3f}", "median_cer_rec": "",
                        "median_ratio_or_s": f"{st.median(int(r['n_tokens']) / 25 for r in g):.2f}s",
                        "word_found": rate(g, word)})
    with open(OUT / "summary.tsv", "w") as f:
        w = csv.DictWriter(f, list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)
    for r in out:
        print("\t".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
