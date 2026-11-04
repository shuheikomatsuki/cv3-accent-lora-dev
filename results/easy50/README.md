# easy_50 with CV3+LoRA, cross_lingual (check against UtterTune Table 1, Test Set 1)

- Decided by the user on 2026-11-04, before the final-evaluation design (§13–§14) is fixed. Purpose: see the CER
  of this study's model on UtterTune's Test Set 1 input. The user stated that the result will not be used to
  decide the final-evaluation conditions.
- Input: `data/eval/easy_50.tsv` column `phon_tagged` as is (= condition C, UtterTune's input form).
  Model: step 10000 LoRA. Mode: cross_lingual. LLM seeds 0, 1; flow seed 0. References: JA dev 4 only
  (no eval references). ASR: kana-whisper only.
- Reference readings: pyopenjtalk 0.4.1 (added to envs/eval) + target kana from the tag; checked by Claude,
  5 overrides in `data/eval/easy50_reading_fixes.tsv`. Alternative valid readings (e.g. 家中, 町中) are not
  accepted, so CER is slightly pessimistic.
- Not comparable one-to-one with UtterTune: base model (CV3 vs CV2), mode (cross_lingual vs zero_shot),
  references (JVS dev 4 vs ReazonSpeech 42), ASR (kana-whisper vs constrained Whisper large-v3), their CER>0.5
  exclusion and trailing-repeat trimming. Both aggregations are reported.
- Scripts: `scripts/easy50/01_inputs.py` … `04_summary.py`.

## 2x2: model {CV2, CV3} x {base, +LoRA}, all cross_lingual (2026-11-04)

To find why condition C scores several times UtterTune's CER on the same input. Same sentences, mode, references
(JA dev 4), reading references and ASR (kana-whisper); LLM seed 0. CV2 side: `scripts/easy50/05_gen_cv2.py` in
`/workH/kmt/pre_exps/1019_uttertune_eval_cmp/.venv_ut`; CV3 base: `02_gen.py --cond A`. The user included the
CV3 base cell in the same decision (not used to choose final-evaluation conditions).

| | base, plain text | + LoRA, tagged |
|---|---|---|
| CV2 | CV2A 0.081 (target 0.71) | UT 0.035 (target 0.95) |
| CV3 | A 0.562 (target 0.22) | C 0.385 (target 0.95) |

CER capped per file, seed 0, 200 files each (`summary.json`). UtterTune's Table 1: CosyVoice 2 0.0557,
UtterTune 0.0639. The gap is the base model: CV2 reads the kanji, CV3 does not; both LoRAs fix the target word.
CV2A ran to max_len on 20 of 50 sentences (trailing extra audio); ignoring trailing extra characters its CER is 0.076.
