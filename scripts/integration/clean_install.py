"""Build an unstaged source copy and verify a wheel in a fresh CPU venv.

This checks packaging from current bytes, NOT a committed Colab release.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / filename, source / filename)
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    evidence = {"status": "RUNNING", "source": "current uncommitted bytes", "benchmark_eligible": False, "commands": []}
    clean_env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        clean_env.pop(name, None)

    def run(name, argv, cwd=base, timeout=900):
        result = subprocess.run(list(map(str, argv)), cwd=cwd, env=clean_env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout)
        evidence["commands"].append(dict(name=name, argv=list(map(str, argv)), returncode=result.returncode,
                                         stdout=result.stdout, stderr=result.stderr))
        (base / "receipt.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name}: {result.stderr}")
    run("build_current_source_wheel", [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", base / "dist", source])
    venv = base / ".venv"
    run("create_isolated_venv", [sys.executable, "-m", "venv", venv])
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run("hash_locked_cpu_install", [shutil.which("uv"), "pip", "install", "--python", python,
                                     "--require-hashes", "-r", ROOT / "requirements-cpu.lock"])
    wheel = next((base / "dist").glob("*.whl"))
    run("install_wheel", [python, "-m", "pip", "install", "--no-deps", wheel])
    run("pip_check", [python, "-m", "pip", "check"])
    run("module_imports", [python, "-c", "import neu_esc; import neu_esc.data.dataset; import neu_esc.training.train; import neu_esc.training.predict; import neu_esc.evaluation.evaluate; print(neu_esc.__file__)"])
    for module in ("data.prepare", "training.train", "training.predict", "evaluation.evaluate"):
        run("cli_help_" + module, [python, "-m", "neu_esc." + module, "--help"])
    run("wheel_p0_e2e", [python, ROOT / "scripts/integration/audit_pipeline.py", "--output", base / "p0", "--variants", "mtl_smart"], cwd=ROOT)
    evidence["status"] = "PASS"
    evidence["python"] = str(python)
    (base / "receipt.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "receipt": str(base / "receipt.json")}))


if __name__ == "__main__":
    main()
