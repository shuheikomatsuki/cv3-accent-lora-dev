# cv3-accent-lora-dev

LoRA for local reading/accent control in Fun-CosyVoice3 (UtterTune-style `<PHON_START>…<PHON_END>` tags).

## Read first
- Research plan: `/workH/kmt/gr/research_plan_cv3_accent_lora.md` (outside this repo). Follow its §0 rules:
  one fact in one place, never treat 【検討中】 items as decided, do not re-propose §3 items.

## Environment
- Host: helen only (RTX PRO 4000 Blackwell, sm_120). `uv sync`, then `uv run python …`.
- Versions are pinned in `pyproject.toml` (Python 3.10, torch 2.7.1+cu128, onnxruntime-gpu 1.22.0); see the comments there before changing any pin.
- `third_party/CosyVoice` is a submodule pinned to official 074ca6d. Never edit it; patch at runtime in `src/cv3lora/` instead.
- Load the model with `cv3lora.cosyvoice_loader.load_cosyvoice3()` (sets sys.path, disables wetext).

## Layout
- `src/cv3lora/`: importable code. `scripts/<stage>/NN_*.py`: entry points, run in number order.
- `configs/paths.local.yaml` (gitignored): machine paths; `load_paths()` reads it.
- `data/`: small text only. `data/uttertune/jsutjvs_manifest.tsv` is the training manifest
  (UtterTune original committed first; each correction is its own commit, reason in `fixes.tsv`).
  Read it with `cv3lora.manifest.load_manifest()`. Recordings listed in `data/uttertune/exclude.tsv`
  (`load_excluded()`) stay in the split but must be dropped by training and dev-loss loaders, by the
  training side of seen/unseen, and by any evaluation that uses the recordings themselves (e.g. §15.4
  condition O); synthesis from dev/aux_eval text keeps them. Transcript typos the speaker read correctly
  are overridden by `transcript_fixes.tsv` (`original_texts()`). `data/splits/`: split and seen/unseen flags. `results/`: small tables and figures.
- Never commit audio, speech tokens, model weights or checkpoints. They live under
  `token_root` and `work_root` (see paths.local.yaml).
- Speech tokens are canonical once extracted (§10); do not re-extract or overwrite them.
