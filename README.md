# cv3-accent-lora-dev

LoRA for local reading/accent control in Fun-CosyVoice3.

## Setup

```bash
git clone --recursive https://github.com/shuheikomatsuki/cv3-accent-lora-dev.git
cd cv3-accent-lora-dev
uv sync
cp configs/paths.example.yaml configs/paths.local.yaml  # then edit
```

If you cloned without `--recursive`: `git submodule update --init --recursive`.

## Speech tokens

```bash
uv run python scripts/data/03_extract_tokens.py            # all subsets
uv run python scripts/data/03_extract_tokens.py jsut_basic5000 --limit 3
```

## Data and licenses

Audio, speech tokens and model weights are not distributed. JVS and JSUT must be obtained from their
official sites. Files under `data/uttertune/` derive from UtterTune data (CC BY-SA 4.0).
