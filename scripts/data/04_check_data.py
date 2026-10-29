"""Automatic data checks (research plan §9, steps 1-3). Listening (steps 4-6) is done by people.

Checks, one issues.tsv row per finding (id, check, detail):
  missing_wav / missing_token / unlisted_wav   manifest <-> audio <-> speech tokens
  audio_unreadable / token_length              wav header; tokens vs duration at 25 Hz
  duration_outlier                             parallel100: robust z of duration among the speakers
                                               of the same sentence; basic5000: seconds per mora
  text_vs_transcript                           text outside the tag differs from the corpus transcript;
                                               JVS rows must match the JSUT row up to accent marks
  surface_vs_kana                              kana characters of the replaced surface (e.g. が in 貝がら)
                                               are not found, in order, in the tag kana
  jsut_label_text_differs                      basic5000: jsut-label says the recording differs from
                                               the transcript (text_level1 != text_level2)
  tag_kana_not_in_reading                      basic5000: tag kana not found in jsut-label's
                                               spoken reading (kana_level3); イ/ー etc. count as equal
  tag_format                                   tag count, characters, accent marks
  nucleus_on_special_mora                      accent mark right after ー/ッ/ン (unusual, not an error)
  duplicate_audio                              identical speech tokens under different ids
Also checks the evaluation TSVs' tags (tag_format, kana_baseline).

Outputs: results/data_check/issues.tsv, summary.md
Usage: uv run python scripts/data/04_check_data.py
"""
import csv
import re
import statistics
from collections import Counter, defaultdict

import soundfile as sf
import yaml

from cv3lora.manifest import TAG, load_manifest, morae, original_texts, surface_of_tag, target_kana
from cv3lora.paths import REPO, load_paths

OUT = REPO / "results/data_check"
TOKEN_HZ = 25
Z_MAX = 5.0
KANA = re.compile(r"[ァ-ヴー'/]+")
SPECIAL = set("ーッン")  # morae that rarely carry the accent nucleus


def to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def kana_pattern(kana: str) -> re.Pattern:
    """Regex for kana where a vowel kana and ー match each other (ケイベツ ~ ケーベツ, イイ ~ イー)."""
    return re.compile("".join("[ーアイウエオ]" if c in "ーアイウエオ" else re.escape(c) for c in kana))


READ_AS = {"ヶ": "[カケガ]", "ヅ": "[ヅズ]", "ヂ": "[ヂジ]", "ハ": "[ハワ]", "ヘ": "[ヘエ]", "ヲ": "[ヲオ]",
           "ヴ": "[ヴバビブベボ]", **dict.fromkeys("ァィゥェォ", "")}


def surface_matches(surface: str, kana: str) -> bool:
    """Kana written in the surface must occur in order in the tag kana (kanji may read as anything)."""
    parts = [READ_AS.get(c, kana_pattern(c).pattern) for c in to_kata(surface)
             if "ァ" <= c <= "ヺ" or c == "ー"]
    return re.search(".*".join(p for p in parts if p), kana) is not None


def check_tag(add, uid: str, tagged: str, plain: str) -> None:
    for pr in tag_problems(tagged):
        add(uid, *tag_issue(pr))
    if (surface := surface_of_tag(tagged, plain)) is None:
        add(uid, "text_vs_transcript", f"{tagged} | {plain}")
    elif not surface_matches(surface, kana := target_kana(TAG.search(tagged).group(1))):
        add(uid, "surface_vs_kana", f"{surface} -> {kana} | {tagged}")


def tag_problems(text: str) -> list[str]:
    spans = TAG.findall(text)
    probs = []
    if text.count("<PHON_START>") != 1 or text.count("<PHON_END>") != 1 or len(spans) != 1:
        return [f"{len(spans)} tags"]
    span = spans[0]
    if not KANA.fullmatch(span):
        probs.append(f"non-kana in tag: {span}")
    for ph in span.split("/"):
        if not ph.replace("'", ""):
            probs.append(f"empty phrase: {span}")
            continue
        if ph.count("'") > 1:
            probs.append(f"{ph.count(chr(39))} nuclei in one phrase: {span}")
        if ph.startswith("'"):
            probs.append(f"nucleus before the first mora: {span}")
        i = ph.find("'")
        if i > 0 and morae(ph[:i])[-1][0] in SPECIAL:
            probs.append(f"special: {span}")
    return probs


def tag_issue(problem: str) -> tuple[str, str]:
    if problem.startswith("special: "):
        return "nucleus_on_special_mora", problem.removeprefix("special: ")
    return "tag_format", problem


def robust_z(x: float, xs: list[float]) -> float:
    med = statistics.median(xs)
    mad = statistics.median(abs(v - med) for v in xs) * 1.4826
    return (x - med) / mad if mad else 0.0


def main() -> None:
    paths = load_paths()
    rows = load_manifest()
    orig = original_texts(paths, rows)
    issues = []

    def add(uid, check, detail=""):
        issues.append([uid, check, detail])

    # Correspondence and audio
    wav_of, dur = {}, {}
    for r in rows:
        if r["corpus"] == "jsut":
            wav_of[r["id"]] = paths["jsut_root"] / r["subset"] / "wav" / f"{r['file_id']}.wav"
        else:
            wav_of[r["id"]] = (paths["jvs_root"] / r["speaker"] / r["subset"] / "wav24kHz16bit"
                               / f"{r['file_id']}.wav")
    listed = set(wav_of.values())
    on_disk = [*paths["jsut_root"].glob("basic5000/wav/*.wav"),
               *paths["jsut_root"].glob("voiceactress100/wav/*.wav"),
               *paths["jvs_root"].glob("jvs[0-9][0-9][0-9]/parallel100/wav24kHz16bit/*.wav")]
    for p in sorted(on_disk):
        if p not in listed:
            add(p.stem, "unlisted_wav", str(p))
    n_tok, sha = {}, defaultdict(list)
    for sub in ("jsut_basic5000", "jsut_voiceactress100", "jvs_parallel100"):
        with open(paths["token_root"] / sub / "index.tsv") as f:
            for t in csv.DictReader(f, delimiter="\t"):
                n_tok[t["id"]] = int(t["n_tokens"])
                sha[t["sha1"]].append(t["id"])
    for r in rows:
        uid, p = r["id"], wav_of[r["id"]]
        if not p.exists():
            add(uid, "missing_wav", str(p))
            continue
        try:
            dur[uid] = sf.info(p).duration
        except RuntimeError as e:
            add(uid, "audio_unreadable", str(e))
            continue
        if uid not in n_tok:
            add(uid, "missing_token")
        elif abs(n_tok[uid] - dur[uid] * TOKEN_HZ) > 2:
            add(uid, "token_length", f"{n_tok[uid]} tokens for {dur[uid]:.2f} s")
    for ids in sha.values():
        if len(ids) > 1:
            for uid in ids:
                add(uid, "duplicate_audio", " ".join(i for i in ids if i != uid))

    # Durations
    by_sent = defaultdict(list)
    for r in rows:
        if r["subset"] == "parallel100" and r["id"] in dur:
            by_sent[r["file_id"]].append(r["id"])
    for ids in by_sent.values():
        xs = [dur[i] for i in ids]
        for i in ids:
            if abs(z := robust_z(dur[i], xs)) > Z_MAX:
                add(i, "duration_outlier", f"{dur[i]:.2f} s, z={z:.1f} among {len(xs)} speakers")
    label = yaml.safe_load((paths["jsut_label_root"] / "text_kana/basic5000.yaml").read_text())
    basic = [r for r in rows if r["subset"] == "basic5000" and r["id"] in dur]
    spm = {r["id"]: dur[r["id"]] / len(morae(to_kata(re.sub(r"[、，]", "", label[r["file_id"]]["kana_level3"]))))
           for r in basic}
    xs = list(spm.values())
    for uid, v in spm.items():
        if abs(z := robust_z(v, xs)) > Z_MAX:
            add(uid, "duration_outlier", f"{v:.3f} s/mora, z={z:.1f}")

    # Text
    jsut_text = {r["file_id"]: r["text"] for r in rows if r["subset"] == "voiceactress100"}
    for r in rows:
        uid, text = r["id"], r["text"]
        if "<PHON_START>" not in text:
            if text != orig[uid]:
                add(uid, "text_vs_transcript", f"{text} | {orig[uid]}")
            continue
        if r["corpus"] == "jvs":  # same text as the JSUT row, which is checked once
            if text.replace("'", "") != jsut_text[r["file_id"]].replace("'", ""):
                add(uid, "text_vs_transcript", f"differs from the JSUT row: {text}")
            elif text != jsut_text[r["file_id"]]:  # accent marks follow each recording (fixes.tsv)
                check_tag(add, uid, text, orig[uid])
            continue
        check_tag(add, uid, text, orig[uid])
    for r in rows:
        if r["subset"] != "basic5000":
            continue
        e = label[r["file_id"]]
        if e["text_level1"] != e["text_level2"]:
            add(r["id"], "jsut_label_text_differs", f"{e['text_level1']} -> {e['text_level2']}")
        if m := TAG.search(r["text"]):
            spoken = to_kata(re.sub(r"[、，]", "", e["kana_level3"]))
            if not kana_pattern(target_kana(m.group(1))).search(spoken):
                add(r["id"], "tag_kana_not_in_reading", f"{m.group(1)} | {e['kana_level3']}")

    # Evaluation TSVs
    ev = paths["uttertune_root"] / "data/eval"
    for name in ("easy_50", "difficult_50"):
        with open(ev / f"{name}.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                uid = f"{name}_{r['id']}"
                check_tag(add, uid, r["phon_tagged"], r["vanilla"])
                if "kana_baseline" in r:
                    kana = target_kana(TAG.search(r["phon_tagged"]).group(1))
                    if r["phon_tagged"].replace(TAG.search(r["phon_tagged"]).group(0), kana) != r["kana_baseline"]:
                        add(uid, "kana_baseline", f"{r['kana_baseline']} | {r['phon_tagged']}")

    OUT.mkdir(parents=True, exist_ok=True)
    issues.sort(key=lambda x: (x[1], x[0]))
    with open(OUT / "issues.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "check", "detail"])
        w.writerows(issues)
    counts = Counter(i[1] for i in issues)
    lines = ["# Data check (§9, steps 1-3)", "",
             (f"Manifest rows {len(rows)}; "
             f"wav durations read {len(dur)}; total {sum(dur.values()) / 3600:.2f} h."), "",
             "| check | findings |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "Details: issues.tsv. Findings are candidates for review, not exclusions."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
