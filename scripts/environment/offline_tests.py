"""Run CPU fixtures offline; exclude externally marked modules and test cases."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BLOCKED_MARKERS = {"gpu", "network", "paid_api", "dataset"}


def has_external_marker(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    # Only a module-level pytestmark suppresses import of the whole file.
    # Function/class markers are filtered by pytest so mixed files retain CPU cases.
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
            continue
        if statement.value is None:
            continue
        for node in ast.walk(statement.value):
            if isinstance(node, ast.Attribute) and node.attr in BLOCKED_MARKERS:
                if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
                    return True
    return False


def main() -> int:
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
               CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    import neu_esc

    print(f"neu_esc import: {neu_esc.__version__}", flush=True)
    ignores = [
        f"--ignore={path.relative_to(ROOT).as_posix()}"
        for base in (ROOT / "tests", ROOT / "scripts/environment/tests")
        if base.exists()
        for path in sorted(base.rglob("test_*.py"))
        if has_external_marker(path)
    ]
    cmd = [sys.executable, "-m", "pytest", *ignores, "-m",
           "not gpu and not network and not paid_api and not dataset"]
    result = subprocess.run([*cmd, "--collect-only", "-q"], cwd=ROOT, env=env, check=False)
    if result.returncode:
        return result.returncode
    return subprocess.run([*cmd, "-q"], cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
