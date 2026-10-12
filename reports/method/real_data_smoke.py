"""Tiny initialized CPU smoke on eight real records per split. Never benchmark."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

from neu_esc.training.config import load_config


def main():
    base = Path("runs/smoke/training/real-tiny-" + uuid.uuid4().hex[:12])
    base.mkdir(parents=True, exist_ok=False)
    config = load_config("tests/training/fixtures/tiny.yaml")
    assert config["training"]["mode"] == "smoke"
    config["training"]["smoke_sample_limit"] = 8
    path = base / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = base / "run"
    commands = [
        ["-m", "neu_esc.training.train", "--config", str(path), "--seed", "42", "--run-dir", str(run_dir)],
        ["-m", "neu_esc.training.predict", "--run-dir", str(run_dir), "--split", "test"],
        ["-m", "neu_esc.evaluation.evaluate", "--predictions", str(run_dir / "predictions/test.jsonl"),
         "--dataset-manifest", config["dataset_manifest"], "--output", str(run_dir / "evaluation/test")],
    ]
    # The evaluator CLI correctly rejects subset predictions against a full split.
    # Only train and predict run here; subset metrics use evaluate_records in predictor.
    commands = commands[:2]
    evidence = {"fixture_only": False, "tiny_smoke_only": True, "benchmark": False,
                "model_initialization": "tiny_random", "sample_limit_per_split": 8,
                "base": str(base), "commands": []}
    for command in commands:
        result = subprocess.run([sys.executable, "-X", "utf8", *command], capture_output=True,
                                text=True, encoding="utf-8")
        evidence["commands"].append({"argv": command, "exit_code": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr})
        (base / "commands.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise RuntimeError(result.stderr)
    evidence["status"] = "PASS"
    Path("reports/method/real_data_smoke.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "base": evidence["base"], "benchmark": False}))


if __name__ == "__main__":
    main()
