"""Reference speakers and candidate reference utterances, by rule only (research plan §8.2, §13.2).

Speakers:
  JA  JVS dev 4 / eval 20 from data/splits/split.json (§8.2).
  EN  eval: LibriTTS-R test-clean, simple random sample of 10 per gender (seed "<SEED>-en-speakers")
      among speakers with at least N_CAND eligible utterances. Gender from LibriTTS speakers.tsv.
      dev: the 4 dev-clean speakers of the EN corpus pilot (1272 M, 3000 M, 1673 F, 1993 F), drawn there
      by the same per-gender random rule with seed 20261027 (/workH/kmt/gr/review/en_corpus/select_refs.py).
  aux JECS neutral, the single bilingual speaker; a candidate is a JA/EN pair with the same index.
Eligible utterances, fixed before measuring or listening:
  JVS nonpara30 non-BASIC5000 sentences (§13.2); LibriTTS-R without double quotes in the text (quoted
  dialogue in audiobooks may be voiced as a character); JECS pairs with no emphasis mark (*) in JA or EN.
  4–8 s (JECS: both JA and EN); no clipped sample (|x| >= 0.999); active-speech level within 10 dB of the
  median of the measured utterances of the same corpus (the "extreme level" check of §13.2).
  Reference F0 range is recorded, not used for selection.
Candidates: per speaker, the eligible utterances in a random order (seed "<SEED>-refs-<speaker>"), first N_CAND.
Listening (by a person, after this script; not by synthesis results): go down the ranks and adopt the first
  candidate with no noticeable noise, reverberation, distortion, cut onset/offset or other voices, read in a
  normal style (no character voice, whisper or shouting). The next passing one is kept for auxiliary
  comparison. Verdicts go to data/refs/listening.tsv.
Measures (crude, for display): level_db = RMS of 20 ms frames within 40 dB of the loudest frame (dBFS);
  lead_s / trail_s = time before the first / after the last such frame; f0 (WORLD dio+stonemask) median in Hz
  and 5–95 percentile range in semitones. JECS rows also get the JA text's overlap with the training
  sentences and the eval TSVs (exact match after removing punctuation/spaces, and the best difflib ratio).

Output: data/refs/candidates.tsv
  lang, role (dev / eval / aux), gender, speaker, rank, id, path, text, dur, lead_s, trail_s, level_db, peak,
  f0_med, f0_range_st, pair (JECS: the id of the other language), overlap (JECS JA only)
Usage: uv run python scripts/data/07_ref_candidates.py
"""
import csv
import difflib
import json
import random
import re
import statistics
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

from cv3lora.manifest import EVAL, load_manifest, original_texts, read_transcript
from cv3lora.paths import REPO, load_paths

SEED = 20261027
N_EN_PER_GENDER, N_CAND = 10, 5
EN_DEV = {"1272": "M", "3000": "M", "1673": "F", "1993": "F"}
DUR, LEVEL_TOL = (4.0, 8.0), 10.0
OUT = REPO / "data/refs/candidates.tsv"


def dur_ok(p: Path) -> bool:
    return DUR[0] <= sf.info(p).duration <= DUR[1]


def measure(p: str) -> dict:
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x[:, 0]
    hop = int(0.02 * sr)
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    act = np.where(e > e.max() - 40)[0]
    f0, t = pyworld.dio(x, sr, frame_period=10)
    v = pyworld.stonemask(x, f0, t, sr)
    v = v[v > 0]
    st = 12 * np.log2(v / np.median(v)) if len(v) > 10 else None
    return {"dur": round(len(x) / sr, 3), "lead_s": round(act[0] * 0.02, 2), "trail_s": round((len(e) - 1 - act[-1]) * 0.02, 2),
            "level_db": round(float(10 * np.log10(np.mean(fr[act] ** 2) + 1e-12)), 2),
            "peak": round(float(np.abs(x).max()), 4), "clip": int(np.sum(np.abs(x) >= 0.999)),
            "f0_med": round(float(np.median(v)), 1) if st is not None else "",
            "f0_range_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2) if st is not None else ""}


def jvs_pool(paths, split) -> list[dict]:
    out = []
    for role in ("dev", "eval"):
        for s in split["speakers"][role]:
            spk, g = s[:6], s[7]
            d = paths["jvs_root"] / spk / "nonpara30"
            text = read_transcript(d / "transcripts_utf8.txt")
            for p in sorted((d / "wav24kHz16bit").glob("*.wav")):
                if not p.stem.startswith("BASIC5000") and dur_ok(p):
                    out.append({"lang": "ja", "role": role, "gender": g, "speaker": spk, "id": p.stem, "path": str(p),
                                "text": text.get(p.stem, ""), "corpus": "jvs"})
    return out


def libri_utts(root: Path, spk: str) -> list[dict]:
    out = []
    for p in sorted((root / spk).glob("*/*.wav")):
        t = p.with_suffix(".normalized.txt").read_text().strip()
        if not re.search(r'[“”"]', t) and dur_ok(p):
            out.append({"id": p.stem, "path": str(p), "text": t})
    return out


def en_pool(paths) -> list[dict]:
    with open(paths["libritts_root"] / "speakers.tsv") as f:
        gender = {r["READER"]: r["GENDER"] for r in csv.DictReader(f, delimiter="\t")}
    test = paths["libritts_r_root"] / "test-clean"
    utts = {s.name: libri_utts(test, s.name) for s in sorted(test.iterdir()) if s.is_dir()}
    rng = random.Random(f"{SEED}-en-speakers")
    picked = {}
    for g in ("M", "F"):
        cand = sorted(s for s, u in utts.items() if gender.get(s) == g and len(u) >= N_CAND)
        print(f"EN test-clean {g}: {len(cand)} speakers with >= {N_CAND} eligible utterances")
        picked.update({s: ("eval", g) for s in rng.sample(cand, N_EN_PER_GENDER)})
    dev = paths["libritts_r_root"] / "dev-clean"
    utts.update({s: libri_utts(dev, s) for s in EN_DEV})
    picked.update({s: ("dev", g) for s, g in EN_DEV.items()})
    return [{"lang": "en", "role": role, "gender": g, "speaker": s, "corpus": "libritts_r", **u}
            for s, (role, g) in sorted(picked.items()) for u in utts[s]]


def jecs_pool(paths) -> list[dict]:
    d = paths["jecs_root"] / "neutral"
    ja, en = read_transcript(d / "transcripts_ja.txt"), read_transcript(d / "transcripts_en.txt")
    out = []
    for k in sorted(ja):
        i = k.removesuffix("_JA")
        tj, te = ja[k].strip(), en.get(f"{i}_EN", "*").strip()
        pj, pe = d / f"wav24kHz16bit/{i}_JA.wav", d / f"wav24kHz16bit/{i}_EN.wav"
        if "*" in tj or "*" in te or not (pj.exists() and pe.exists() and dur_ok(pj) and dur_ok(pe)):
            continue
        out.append({"lang": "ja-en", "role": "aux", "gender": "", "speaker": "jecs", "id": i, "path": str(pj),
                    "text": tj, "corpus": "jecs", "en_path": str(pe), "en_text": te})
    return out


def norm(t: str) -> str:
    return re.sub(r"[\s、。，．,.!?！？「」『』（）()・…ー〜~-]", "", t)


def overlap(text: str, refs: list[str]) -> str:
    t = norm(text)
    if t in refs:
        return "exact"
    best = max(refs, key=lambda r: difflib.SequenceMatcher(None, t, r).ratio())
    return f"{difflib.SequenceMatcher(None, t, best).ratio():.2f} {best}"


def known_texts(paths) -> list[str]:
    rows = load_manifest()
    texts = set(original_texts(paths, rows).values())
    for name in ("easy_50", "difficult_50"):
        with open(EVAL / f"{name}.tsv") as f:
            texts |= {r["vanilla"] for r in csv.DictReader(f, delimiter="\t")}
    with open(paths["uttertune_root"] / "data/eval/katakana_50.tsv") as f:
        r = list(csv.DictReader(f, delimiter="\t"))
        texts |= {x["text"] for x in r}
    return sorted({norm(t) for t in texts})


def main():
    paths = load_paths()
    split = json.loads((REPO / "data/splits/split.json").read_text())
    pool = jvs_pool(paths, split) + en_pool(paths) + jecs_pool(paths)
    files = [u["path"] for u in pool] + [u["en_path"] for u in pool if "en_path" in u]
    with Pool(16) as p:
        m = dict(zip(files, p.map(measure, files, chunksize=8)))
    med = {c: statistics.median(m[u["path"]]["level_db"] for u in pool if u["corpus"] == c) for c in ("jvs", "libritts_r", "jecs")}
    med["jecs_en"] = statistics.median(m[u["en_path"]]["level_db"] for u in pool if "en_path" in u)
    print("level median (dBFS):", {k: round(v, 2) for k, v in med.items()})

    def ok(path, c):
        r = m[path]
        return r["clip"] == 0 and abs(r["level_db"] - med[c]) <= LEVEL_TOL

    by = {}
    for u in pool:
        if ok(u["path"], u["corpus"]) and ("en_path" not in u or ok(u["en_path"], "jecs_en")):
            by.setdefault((u["lang"], u["role"], u["speaker"]), []).append(u)
        else:
            print("excluded:", u["speaker"], u["id"], m[u["path"]]["clip"], m[u["path"]]["level_db"])
    known = known_texts(paths)
    rows = []
    cols = ["dur", "lead_s", "trail_s", "level_db", "peak", "f0_med", "f0_range_st"]
    for (lang, role, spk), us in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        us = sorted(us, key=lambda u: u["id"])
        random.Random(f"{SEED}-refs-{spk}").shuffle(us)
        if len(us) < N_CAND:
            print(f"{spk}: only {len(us)} candidates")
        for k, u in enumerate(us[:N_CAND], 1):
            base = {"lang": "ja" if lang == "ja-en" else lang, "role": role, "gender": u["gender"], "speaker": spk, "rank": k}
            rows.append({**base, "id": u["id"] if lang != "ja-en" else f"{u['id']}_JA", "path": u["path"], "text": u["text"],
                         **{c: m[u["path"]][c] for c in cols},
                         "pair": f"{u['id']}_EN" if lang == "ja-en" else "",
                         "overlap": overlap(u["text"], known) if lang == "ja-en" else ""})
            if lang == "ja-en":
                rows.append({**base, "lang": "en", "id": f"{u['id']}_EN", "path": u["en_path"], "text": u["en_text"],
                             **{c: m[u["en_path"]][c] for c in cols}, "pair": f"{u['id']}_JA", "overlap": ""})
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(OUT, len(rows), "rows,", len(by), "speakers")


if __name__ == "__main__":
    main()
