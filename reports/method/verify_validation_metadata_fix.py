"""Bounded fresh CLI smokes and the unchanged module 06 reproduction for I06-02."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

from cli_smoke import run_smoke
from neu_esc.training.config import file_digest, implementation_digest


def main():
    output = Path("reports/method/i06-02-verification.json")
    if output.exists():
        raise FileExistsError("Verification receipt already exists; preserve evidence and choose a new receipt path")
    baseline = json.loads(Path("reports/method/i06-02-baseline.json").read_text(encoding="utf-8"))
    base = Path("runs/smoke/training/i06-02-" + uuid.uuid4().hex[:12])
    base.mkdir(parents=True, exist_ok=False)
    evidence = {"issue": "I06-02", "fixture_only": True, "benchmark": False,
                "implementation_hash": implementation_digest(), "base": str(base), "variants": {}}
    for variant in ("st_sentiment", "st_topic", "mtl_smart"):
        smoke = run_smoke(base / variant, variant=variant)
        run_dir = base / variant / "run"
        command = [sys.executable, "-X", "utf8", "scripts/integration/reproduce_validation_metadata.py",
                   "--run-dir", str(run_dir)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr + result.stdout
        evidence["variants"][variant] = {"smoke": smoke, "reproduction": {
            "argv": command[3:], "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}}
        print(variant, "fresh CLI/validation aggregate/reproduction PASS", flush=True)
    old_run = Path(baseline["old_run"])
    after = {p.relative_to(old_run).as_posix(): file_digest(p) for p in old_run.rglob("*") if p.is_file()}
    assert after == baseline["old_run_sha256"], "Old integration evidence changed"
    changes = [p for p, digest in baseline["protected_files"].items() if file_digest(p) != digest]
    assert not changes, f"Protected model/loss/config/data/evaluator files changed: {changes}"
    evidence.update(old_integration_artifacts_unchanged=True, protected_files_unchanged=True, status="PASS")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "base": str(base), "receipt": str(output)}))


if __name__ == "__main__":
    main()
