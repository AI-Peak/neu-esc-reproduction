"""End-to-end fixture-only smoke of real data/evaluator APIs, CLI resume and fresh prediction."""
import json
from pathlib import Path
import subprocess
import sys
import uuid
import argparse

import yaml

from neu_esc.data.common import ID_ALGORITHM, LABEL_MAPS
from neu_esc.training.config import load_config, canonical_digest, file_digest, validate_config, TASKS
from neu_esc.evaluation.aggregate import aggregate_cohort
from neu_esc.evaluation.evaluate import evaluate_records
from neu_esc.data.dataset import load_split


def run_smoke(base=None, variant="mtl_mlm_smart"):
    base = Path(base or ("runs/smoke/training/cli-" + uuid.uuid4().hex[:12]))
    base.mkdir(parents=True, exist_ok=False)
    processed = base / "fixture_data"
    processed.mkdir()
    raw_hash = "a" * 64
    manifest = {"schema_version": 1, "source_urls": ["synthetic://module02-cli-smoke"],
                "source_revision": "synthetic-fixture-only", "raw_sha256": {}, "processed_sha256": {},
                "row_counts": {}, "id_algorithm": ID_ALGORITHM, "label_maps": LABEL_MAPS,
                "preprocessing": {"steps": [], "segmentation": "fixture"},
                "hf_equivalence_status": "not_applicable_fixture", "assumptions": ["NO BENCHMARK"],
                "processed_files": {}}
    for split, count in (("train", 9), ("validation", 7), ("test", 8)):
        rows = [{"sample_id": f"{split}:{i:06d}:{raw_hash[:12]}", "split": split, "source_row": i,
                 "text": f"word{i % 20} word2 word3 word4 word5", "text_segmented": None,
                 "sentiment": i % 4, "topic": i % 10} for i in range(count)]
        path = processed / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest["row_counts"][split] = count
        manifest["raw_sha256"][split] = raw_hash
        manifest["processed_sha256"][split] = file_digest(path)
        manifest["processed_files"][split] = f"fixture_data/{split}.jsonl"
    manifest["manifest_digest"] = canonical_digest(manifest)
    manifest_path = base / "dataset.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = load_config("tests/training/fixtures/tiny.yaml")
    config["variant"] = variant
    config["tasks"] = TASKS[variant].copy()
    config["training"]["task_weights"] = {task: 1.0 for task in config["tasks"]}
    config["training"]["st_native_head_dropout"] = 0.1 if variant.startswith("st_") else None
    config["smart"]["enabled"] = "smart" in variant
    config["mlm"]["enabled"] = "mlm" in variant
    config["dataset_manifest"] = manifest_path.as_posix()
    config["dataset_manifest_digest"] = manifest["manifest_digest"]
    config["training"]["smoke_sample_limit"] = 20
    validate_config(config)
    config_path = base / "tiny.resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = base / "run"
    commands = [
        ["-m", "neu_esc.training.train", "--config", str(config_path), "--seed", "42", "--run-dir", str(run_dir), "--stop-after-steps", "2"],
        ["-m", "neu_esc.training.train", "--config", str(config_path), "--seed", "42", "--run-dir", str(run_dir), "--resume"],
        ["-m", "neu_esc.training.predict", "--run-dir", str(run_dir), "--split", "test"],
        ["-m", "neu_esc.evaluation.evaluate", "--predictions", str(run_dir / "predictions/test.jsonl"),
         "--dataset-manifest", str(manifest_path), "--output", str(run_dir / "evaluation/test")],
        ["-m", "neu_esc.evaluation.evaluate", "--predictions", str(run_dir / "predictions/validation.jsonl"),
         "--dataset-manifest", str(manifest_path), "--output", str(run_dir / "evaluation/validation-regenerated")],
    ]
    evidence = {"fixture_only": True, "benchmark": False, "base": str(base),
                "variant": variant, "commands": []}
    for command in commands:
        result = subprocess.run([sys.executable, "-X", "utf8", *command], capture_output=True, text=True, encoding="utf-8")
        evidence["commands"].append({"argv": command, "exit_code": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr})
        (base / "commands.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"CLI smoke failed: {command}\n{result.stderr}")
    predictions = [json.loads(line) for line in (run_dir / "predictions/test.jsonl").read_text().splitlines()]
    assert len(predictions) == 8 * len(config["tasks"])
    assert {(row["sample_id"], row["task"]) for row in predictions} == {
        (f"test:{i:06d}:{raw_hash[:12]}", task) for i in range(8) for task in config["tasks"]}
    metrics = json.loads((run_dir / "evaluation/validation/metrics.json").read_text(encoding="utf-8"))
    regenerated = json.loads((run_dir / "evaluation/validation-regenerated/metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    resolved = load_config(run_dir / "config.resolved.yaml")
    fields = ("method_mode", "backbone", "variant")
    for field in fields:
        assert metrics[field] == regenerated[field] == provenance[field] == resolved[field]
    validation_predictions = [json.loads(line) for line in
                              (run_dir / "predictions/validation.jsonl").read_text(encoding="utf-8").splitlines()]
    recomputed = evaluate_records(validation_predictions, load_split(manifest_path, "validation", "raw"),
                                 config["tasks"], allow_invalid=False)
    assert metrics["tasks"] == regenerated["tasks"] == recomputed["tasks"]
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["score"] == sum(metrics["tasks"][task]["macro_f1"] for task in config["tasks"]) / len(config["tasks"])
    aggregate = aggregate_cohort([metrics], expected_seeds=[42])
    assert aggregate["status"] == "complete" and aggregate["seeds"] == [42]
    assert set(aggregate["tasks"]) == set(config["tasks"])
    evidence.update(status="PASS", predictions_rows=len(predictions),
                    validation_metadata={field: metrics[field] for field in fields},
                    validation_aggregation="PASS", validation_metrics_recomputed="PASS",
                    evaluator_cli_metadata_matches="PASS", checkpoint_selection_score_matches="PASS",
                    validation_metrics_sha256=file_digest(run_dir / "evaluation/validation/metrics.json"))
    (base / "commands.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(TASKS), default="mtl_mlm_smart")
    parser.add_argument("--output", type=Path, help="New smoke directory (must not exist)")
    args = parser.parse_args()
    result = run_smoke(args.output, variant=args.variant)
    print(json.dumps({"status": result["status"], "base": result["base"], "predictions_rows": result["predictions_rows"]}))
