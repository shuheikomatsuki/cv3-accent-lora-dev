"""Machine-specific paths, read from configs/paths.local.yaml (copy paths.example.yaml)."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
COSYVOICE_SRC = REPO / "third_party/CosyVoice"


def load_paths(path: Path | None = None) -> dict[str, Path]:
    path = path or REPO / "configs/paths.local.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; copy configs/paths.example.yaml and edit it")
    return {k: Path(v) for k, v in yaml.safe_load(path.read_text()).items()}
