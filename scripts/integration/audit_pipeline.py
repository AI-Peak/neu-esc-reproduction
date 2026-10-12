"""Bounded CPU integration audit. All predictions here are smoke/diagnostic only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def fixture(base):
    from neu_esc.data.common import ID_ALGORITHM, LABEL_MAPS
    from neu_esc.training.config import canonical_digest
    folder = base / "fixture_data"
    folder.mkdir()
    manifest = dict(schema_version=1, source_urls=["synthetic://module06"],
                    source_revision="fixture-only", raw_sha256={}, processed_sha256={},
                    row_counts={}, processed_files={}, id_algorithm=ID_ALGORITHM,
                    label_maps=LABEL_MAPS, preprocessing={"steps": [], "segmentation": "fixture"},
                    hf_equivalence_status="not_applicable_fixture", assumptions=["NO BENCHMARK"])
    for split, count in (("train", 13), ("validation", 11), ("test", 10)):
        samples = [dict(sample_id=f"{split}:{i:06d}:{'a'*12}", split=split, source_row=i,
                        text=f"word{i % 20} word2 word3", text_segmented=None,
                        sentiment=i % 4, topic=i % 10) for i in range(count)]
        path = folder / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in samples), encoding="utf-8", newline="\n")
        manifest["raw_sha256"][split] = "a" * 64
        manifest["processed_sha256"][split] = sha(path)
        manifest["row_counts"][split] = count
        manifest["processed_files"][split] = f"fixture_data/{split}.jsonl"
    manifest["manifest_digest"] = canonical_digest(manifest)
    save(base / "dataset.json", manifest)
    return manifest


def audit(base, variants):
    from neu_esc.data.dataset import load_split
    from neu_esc.evaluation.evaluate import evaluate_records, EvaluationError
    from neu_esc.evaluation.aggregate import aggregate_cohort
    from neu_esc.training.config import load_config, canonical_digest, TASKS
    from neu_esc.training.checkpoint import load_checkpoint
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=False)
    evidence = dict(schema_version="1.0", fixture_only=True, benchmark_eligible=False,
                    commands=[], variants={}, failures=[])
    receipt = base / "receipt.json"

    def command(name, args, expected=0):
        result = subprocess.run([sys.executable, "-X", "utf8", *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180)
        evidence["commands"].append(dict(name=name, argv=args, returncode=result.returncode,
                                         expected_returncode=expected, stdout=result.stdout, stderr=result.stderr))
        save(receipt, evidence)
        if expected == "nonzero":
            assert result.returncode != 0, name
        else:
            assert result.returncode == expected, f"{name}: {result.stderr}"
        return result

    manifest = fixture(base)
    manifest_path = base / "dataset.json"
    expected = load_split(manifest_path, "test", "raw")
    for variant in variants:
        config = load_config(ROOT / "tests/training/fixtures/tiny.yaml")
        config.update(variant=variant, tasks=TASKS[variant], experiment_id=f"integration-{variant}",
                      dataset_manifest=manifest_path.as_posix(), dataset_manifest_digest=manifest["manifest_digest"])
        config["training"].update(smoke_sample_limit=30,
                                  task_weights={t: 1.0 for t in config["tasks"]},
                                  st_native_head_dropout=0.1 if variant.startswith("st_") else None)
        config["smart"]["enabled"] = "smart" in variant
        config["mlm"]["enabled"] = "mlm" in variant
        config_path = base / f"{variant}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
        run = base / variant
        train_args = ["-m", "neu_esc.training.train", "--config", str(config_path), "--seed", "42", "--run-dir", str(run)]
        if variant == "mtl_smart":
            # Train must succeed even when the test file cannot be read.
            test_path = base / "fixture_data/test.jsonl"
            test_path.rename(base / "withheld-test.jsonl")
            try:
                command("p0_interrupt_without_test_file", [*train_args, "--stop-after-steps", "2"])
                assert json.loads((run / "provenance.json").read_text())["run_status"] == "INTERRUPTED"
                command("predict_before_completion_rejected", ["-m", "neu_esc.training.predict", "--run-dir", str(run), "--split", "test"], "nonzero")
                for key, value in (("seed", 43), ("model_revision", "b" * 40)):
                    bad = copy.deepcopy(config)
                    bad[key] = value
                    bad_path = base / f"stale-{key}.yaml"
                    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8", newline="\n")
                    command(f"stale_{key}_resume_rejected", ["-m", "neu_esc.training.train", "--config", str(bad_path), "--run-dir", str(run), "--resume"], "nonzero")
                command("p0_resume_without_test_file", [*train_args, "--resume"])
            finally:
                (base / "withheld-test.jsonl").rename(test_path)
        else:
            command(f"train_{variant}", train_args)
        command(f"predict_{variant}_fresh_process", ["-m", "neu_esc.training.predict", "--run-dir", str(run), "--split", "test"])
        prediction_path = run / "predictions/test.jsonl"
        eval_args = ["-m", "neu_esc.evaluation.evaluate", "--predictions", str(prediction_path),
                     "--dataset-manifest", str(manifest_path), "--output", str(run / "evaluation/test")]
        command(f"evaluate_{variant}_standard_cli", eval_args)
        predictions = rows(prediction_path)
        metrics = json.loads((run / "evaluation/test/metrics.json").read_text(encoding="utf-8"))
        recomputed = evaluate_records(predictions, expected, config["tasks"])
        assert metrics["tasks"] == recomputed["tasks"]
        permuted = copy.deepcopy(predictions)
        random.Random(60).shuffle(permuted)
        assert evaluate_records(permuted, list(reversed(expected)), config["tasks"]) == recomputed
        assert len(predictions) == len(expected) * len(config["tasks"])
        for task, values in metrics["tasks"].items():
            k = 4 if task == "sentiment" else 10
            cm = values["confusion_matrix"]
            assert len(cm) == k and all(len(row) == k for row in cm)
            assert sum(map(sum, cm)) == len(expected) == values["n_samples"]
            assert abs(sum(cm[i][i] for i in range(k))/len(expected)-values["accuracy"]) <= 1e-12
            for suffix in ("", "_normalized"):
                from PIL import Image
                with Image.open(run / f"evaluation/test/confusion_{task}{suffix}.png") as picture:
                    picture.verify()
        assert metrics["predictions_sha256"] == sha(prediction_path)
        assert metrics["dataset_manifest_digest"] == manifest["manifest_digest"]
        one = aggregate_cohort([metrics], expected_seeds=[42])
        assert all(v["std"] is None for t in one["tasks"].values() for v in t["metrics"].values())
        selected = json.loads((run / "selection.json").read_text())
        best, best_sha = load_checkpoint(run / "checkpoints/best")
        latest, latest_sha = load_checkpoint(run / "checkpoints/latest")
        assert best_sha == selected["checkpoint_sha256"]
        assert best["identity"] == latest["identity"]
        assert best["position"]["global_step"] <= latest["position"]["global_step"]
        assert best["resolved_config"] == config
        provenance = json.loads((run / "provenance.json").read_text())
        assert provenance["run_status"] == "COMPLETED" and provenance["benchmark_eligible"] is False
        assert provenance["config_hash"] == canonical_digest(config)
        for relative in ("environment.json", "logs/steps.jsonl", "predictions/validation.jsonl", "evaluation/validation/metrics.json"):
            assert (run / relative).is_file()
        validation_metrics = json.loads((run / "evaluation/validation/metrics.json").read_text())
        try:
            aggregate_cohort([validation_metrics], expected_seeds=[42])
            validation_aggregate = "PASS"
        except EvaluationError as exc:
            validation_aggregate = str(exc)
        evidence["variants"][variant] = dict(status="PASS", model="initialized tiny BERT, NOT pretrained ViSoBERT",
                                             tasks=config["tasks"], predictions_rows=len(predictions),
                                             confusion_n_per_task=len(expected), best_sha256=best_sha,
                                             latest_sha256=latest_sha, validation_aggregation=validation_aggregate,
                                             run=str(run.relative_to(ROOT)).replace("\\", "/"))
        if variant == "mtl_smart":
            bad_cases = {}
            bad_cases["wrong_truth"] = [{**p, "y_true": (p["y_true"] + 1) % (4 if p["task"] == "sentiment" else 10)} if i == 0 else p for i, p in enumerate(predictions)]
            bad_cases["duplicate_id"] = predictions + [predictions[0]]
            bad_cases["missing_task"] = [p for p in predictions if p["task"] == "sentiment"]
            bad_cases["wrong_id"] = [{**p, "sample_id": "test:unknown"} if i == 0 else p for i, p in enumerate(predictions)]
            for name, corrupted in bad_cases.items():
                negative_path = base / f"{name}.jsonl"
                negative_path.write_text("".join(json.dumps(p) + "\n" for p in corrupted), encoding="utf-8", newline="\n")
                output = base / f"rejected-{name}"
                command(f"negative_{name}", ["-m", "neu_esc.evaluation.evaluate", "--predictions", str(negative_path),
                                             "--dataset-manifest", str(manifest_path), "--tasks", "sentiment", "topic",
                                             "--output", str(output)], 2)
                assert not output.exists()
            # Manifest change and checkpoint incompleteness are genuine reject tests.
            original_manifest = manifest_path.read_bytes()
            altered = copy.deepcopy(manifest)
            altered["source_revision"] = "stale-source"
            altered["manifest_digest"] = canonical_digest({k: v for k, v in altered.items() if k != "manifest_digest"})
            save(manifest_path, altered)
            try:
                command("stale_manifest_resume_rejected", [*train_args, "--resume"], "nonzero")
            finally:
                manifest_path.write_bytes(original_manifest)
            marker = run / "checkpoints/best/COMPLETE.json"
            original_marker = marker.read_bytes()
            marker.unlink()
            try:
                command("incomplete_best_prediction_rejected", ["-m", "neu_esc.training.predict", "--run-dir", str(run), "--split", "test"], "nonzero")
            finally:
                marker.write_bytes(original_marker)
        save(receipt, evidence)
    evidence["status"] = "PASS"
    save(receipt, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", nargs="+", default=["mtl_smart", "st_sentiment", "st_topic", "mtl", "mtl_mlm", "mtl_mlm_smart"])
    args = parser.parse_args()
    os.chdir(ROOT)
    base = args.output or ROOT / f"runs/smoke/integration/audit-{uuid.uuid4().hex[:12]}"
    result = audit(base, args.variants)
    print(json.dumps({"status": result["status"], "receipt": str(base / "receipt.json"), "variants": list(result["variants"])}))


if __name__ == "__main__":
    main()
