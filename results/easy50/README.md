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
