"""Load Fun-CosyVoice3 from the pinned submodule without modifying it."""
import subprocess
import sys

from cv3lora.paths import COSYVOICE_SRC, load_paths


def _add_cosyvoice_to_path() -> None:
    for p in (COSYVOICE_SRC, COSYVOICE_SRC / "third_party/Matcha-TTS"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _disable_wetext() -> None:
    # helen has no wetext model and ModelScope returns 403. Making the constructor fail sends
    # CosyVoice to its "no frontend" branch without a download attempt. We never use its text
    # normalization: text tokens are built directly from the tagged text.
    import wetext

    def _fail(self, *a, **k):
        raise RuntimeError("wetext disabled")

    wetext.Normalizer.__init__ = _fail


def load_cosyvoice3(model_dir=None, fp16: bool = False):
    _add_cosyvoice_to_path()
    _disable_wetext()
    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = model_dir or load_paths()["cosyvoice_model"]
    return AutoModel(model_dir=str(model_dir), fp16=fp16)


def cosyvoice_commit() -> str:
    return subprocess.run(["git", "-C", str(COSYVOICE_SRC), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
