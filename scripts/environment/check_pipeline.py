"""Require the installed encoder pipeline, immutable configs and working CLI help."""

from importlib import import_module
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODULES = ("data.dataset", "data.prepare", "data.resources", "models.encoder",
           "losses.smart", "losses.mlm", "training.train", "training.predict",
           "evaluation.evaluate", "evaluation.aggregate")
CLIS = ("data.prepare", "data.resources", "training.train", "training.predict", "evaluation.evaluate")


def main():
    import neu_esc
    from neu_esc.training.config import load_config
    from transformers.utils.import_utils import check_torch_load_is_safe

    for name in MODULES:
        import_module("neu_esc." + name)
    # ViSoBERT's pinned repository supplies pytorch_model.bin, so verify the
    # installed Transformers/PyTorch combination permits that loader path.
    # No model download or pretrained-weight execution is performed here.
    check_torch_load_is_safe()
    configs = sorted((ROOT / "configs/experiments").glob("*/*.yaml"))
    if len(configs) != 22:
        raise RuntimeError(f"Release must contain 22 encoder configs; found {len(configs)}")
    for path in configs:
        load_config(path)
    for name in CLIS:
        subprocess.run([sys.executable, "-m", "neu_esc." + name, "--help"],
                       cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
    print(f"PASS: installed pipeline from {neu_esc.__file__}; {len(configs)} configs; {len(CLIS)} CLI helps; torch bin-loader compatibility guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
