"""Evaluate long-form predictions by (sample_id, task), never by row order.

The Python API is dependency-free. The CLI loads the authoritative split through
neu_esc.data.dataset.load_split and uses Pillow only when rendering PNGs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
LABELS = {
    "sentiment": ("Neutral", "Positive", "Negative", "Toxic"),
    "topic": ("Spam", "News", "Academic", "Other", "Service",
              "Jobs & Recruitment", "Personal Affairs", "Social Affairs",
              "Help & Share", "Club & Events"),
}
SPLITS = {"train", "validation", "test"}
METRICS = ("accuracy", "macro_f1", "weighted_f1")


class EvaluationError(ValueError):
    """Prediction integrity failure; no benchmark metrics may be published."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _label(value: Any, task: str, context: str) -> int:
    _require(isinstance(value, Integral) and not isinstance(value, bool),
             f"invalid class in {context}: expected integer, got {value!r}")
    value = int(value)
    _require(0 <= value < len(LABELS[task]),
             f"invalid class in {context}: {value} for {task}")
    return value


def _identity(record: Mapping, context: str) -> tuple[str, str]:
    sample_id, split = record.get("sample_id"), record.get("split")
    _require(isinstance(sample_id, str) and bool(sample_id.strip()),
             f"missing/invalid sample_id in {context}")
    _require(isinstance(split, str) and split in SPLITS,
             f"wrong split in {context}: {split!r}")
    return sample_id, split


def _task_list(tasks: Sequence[str]) -> list[str]:
    _require(not isinstance(tasks, (str, bytes)), "tasks must be a sequence")
    tasks = list(tasks)
    _require(bool(tasks) and all(isinstance(t, str) and t in LABELS for t in tasks),
             f"tasks must contain sentiment and/or topic: {tasks!r}")
    _require(len(tasks) == len(set(tasks)), "duplicate task in tasks")
    return tasks


def metrics_from_counts(counts: Sequence[Sequence[int]], task: str) -> dict:
    """Calculate fixed-label metrics; rows=true, columns=predicted/INVALID.

    This helper also permits auditing the transcribed paper matrices. Runtime
    predictions must always enter through evaluate_records for coverage checks.
    """
    _require(task in LABELS, f"unknown task: {task!r}")
    k = len(LABELS[task])
    _require(len(counts) == k, f"CM for {task} requires {k} rows")
    width = len(counts[0])
    _require(width in (k, k + 1), f"CM for {task} requires {k} or {k+1} columns")
    cm = []
    for row in counts:
        _require(len(row) == width, "ragged CM")
        _require(all(isinstance(v, Integral) and not isinstance(v, bool) and v >= 0
                     for v in row), "CM counts must be nonnegative integers")
        cm.append([int(v) for v in row])
    support = [sum(row) for row in cm]
    n = sum(support)
    _require(n > 0, "cannot evaluate an empty split/CM")
    trace = sum(cm[i][i] for i in range(k))
    per_class = []
    for i, name in enumerate(LABELS[task]):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k)) - tp
        fn = support[i] - tp  # includes INVALID predictions
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append(dict(class_id=i, label=name, precision=precision,
                              recall=recall, f1=f1, support=support[i],
                              true_positive=tp, false_positive=fp, false_negative=fn))
    accuracy = trace / n
    invalid = sum(row[-1] for row in cm) if width == k + 1 else 0
    result = dict(
        n_samples=n, accuracy=accuracy,
        macro_f1=sum(c["f1"] for c in per_class) / k,
        weighted_f1=sum(c["f1"] * c["support"] for c in per_class) / n,
        per_class=per_class, label_order=list(range(k)), label_names=list(LABELS[task]),
        invalid_count=invalid, valid_count=n-invalid, record_coverage=1.0,
        valid_prediction_coverage=(n-invalid)/n,
        confusion_matrix=cm,
        confusion_matrix_normalized=[
            [v / s if s else 0.0 for v in row] for row, s in zip(cm, support)],
        confusion_columns=list(range(k)) + (["INVALID"] if width == k+1 else []),
        confusion_orientation="rows=true, columns=predicted",
    )
    assert sum(map(sum, cm)) == n
    assert math.isclose(trace/n, result["accuracy"], rel_tol=0, abs_tol=1e-12)
    return result


def evaluate_records(records: Iterable[Mapping], expected_records: Iterable[Mapping],
                     tasks: Sequence[str], allow_invalid: bool = False) -> dict:
    """Validate complete ID/task coverage and return metrics/CMs, without I/O.

    expected_records is the wide-form split returned by the data API. Invalid
    LLM outputs require status='invalid', y_pred=None, and a nonempty error.
    An absent record always fails, including when allow_invalid is True.
    """
    tasks = _task_list(tasks)
    _require(isinstance(allow_invalid, bool), "allow_invalid must be bool")
    expected = {}
    split = None
    for idx, record in enumerate(expected_records):
        _require(isinstance(record, Mapping), f"expected record {idx} must be a mapping")
        sample_id, row_split = _identity(record, f"expected record {idx}")
        _require(sample_id not in expected, f"duplicate expected sample_id: {sample_id}")
        if split is None:
            split = row_split
        _require(row_split == split, f"wrong split in expected records: {row_split} != {split}")
        expected[sample_id] = {t: _label(record.get(t), t, f"expected {sample_id}/{t}")
                               for t in tasks}
    _require(bool(expected), "expected_records must be a nonempty authoritative split")
    seen = set()
    matrices = {t: [[0] * (len(LABELS[t]) + int(allow_invalid))
                    for _ in LABELS[t]] for t in tasks}
    for idx, record in enumerate(records):
        _require(isinstance(record, Mapping), f"prediction record {idx} must be a mapping")
        sample_id, row_split = _identity(record, f"prediction {idx}")
        task = record.get("task")
        _require(isinstance(task, str) and task in tasks,
                 f"extra/wrong task in prediction {idx}: {task!r}; expected {tasks}")
        key = (sample_id, task)
        _require(key not in seen, f"duplicate prediction ID/task: {key}")
        seen.add(key)
        _require(row_split == split, f"wrong split for {key}: {row_split} != {split}")
        _require(sample_id in expected, f"extra prediction sample_id: {sample_id}")
        y_true = _label(record.get("y_true"), task, f"y_true {key}")
        _require(y_true == expected[sample_id][task],
                 f"wrong labels for {key}: prediction y_true={y_true}, authoritative={expected[sample_id][task]}")
        _require("y_pred" in record and "error" in record,
                 f"prediction {key} requires y_pred and error fields")
        status = record.get("status")
        if status == "invalid":
            _require(allow_invalid, f"invalid prediction not allowed for encoder: {key}")
            _require(record["y_pred"] is None, f"invalid prediction must have y_pred=null: {key}")
            _require(isinstance(record["error"], str) and bool(record["error"].strip()),
                     f"invalid prediction requires nonempty error: {key}")
            column = len(LABELS[task])
        else:
            _require(status == "ok", f"unknown status for {key}: {status!r}; use invalid after retries")
            _require(record["error"] is None, f"ok prediction must have error=null: {key}")
            column = _label(record["y_pred"], task, f"y_pred {key}")
        matrices[task][expected[sample_id][task]][column] += 1
    missing = {(sid, task) for sid in expected for task in tasks} - seen
    _require(not missing,
             f"missing prediction ID/task coverage: {len(missing)}; examples={sorted(missing)[:5]}")
    return dict(schema_version=SCHEMA_VERSION, split=split,
                tasks={t: metrics_from_counts(matrices[t], t) for t in tasks})


def _read_jsonl(path: Path, content: bytes | None = None) -> list[dict]:
    rows = []
    content = path.read_bytes() if content is None else content
    for line_no, line in enumerate(content.decode("utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"blank JSONL line {line_no} in {path}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSONL line {line_no} in {path}: {exc}") from exc
        _require(isinstance(value, dict), f"JSONL line {line_no} must be an object")
        rows.append(value)
    return rows


def _manifest_digest(manifest: Mapping) -> str:
    _require(isinstance(manifest, Mapping), "dataset manifest must be a JSON object")
    value = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise EvaluationError("dataset manifest contains nonfinite values") from exc
    digest = hashlib.sha256(canonical).hexdigest()
    _require(manifest.get("manifest_digest") == digest, "dataset manifest_digest mismatch/missing")
    return digest


def write_evaluation(result: Mapping, output: str | Path, render_png: bool = True) -> None:
    """Write metrics and counts from one validated result; refuse nonempty output."""
    output = Path(output)
    _require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
             f"refusing to overwrite nonempty evaluation output: {output}")
    # Fail before writing if PNG dependencies are unavailable.
    if render_png:
        from .plotting import render_confusion_png
        try:
            import PIL  # noqa: F401
        except ImportError as exc:
            raise EvaluationError("PNG output requires Pillow; install Pillow or use --no-png") from exc
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "n_samples", "accuracy_pct", "macro_f1_pct",
                         "weighted_f1_pct", "invalid_count", "record_coverage_pct",
                         "valid_prediction_coverage_pct"])
        for task, values in result["tasks"].items():
            writer.writerow([task, values["n_samples"], *[values[m]*100 for m in METRICS],
                             values["invalid_count"], values["record_coverage"]*100,
                             values["valid_prediction_coverage"]*100])
    for task, values in result["tasks"].items():
        names = values["label_names"]
        columns = names + (["INVALID"] if "INVALID" in values["confusion_columns"] else [])
        for suffix, key in (("", "confusion_matrix"), ("_normalized", "confusion_matrix_normalized")):
            with (output / f"confusion_{task}{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["true_label/predicted_label", *columns])
                writer.writerows([name, *row] for name, row in zip(names, values[key]))
            if render_png:
                render_confusion_png(values[key], names, columns,
                                     output / f"confusion_{task}{suffix}.png",
                                     title=f"{task} | {'row-normalized' if suffix else 'counts'} | N={values['n_samples']}",
                                     normalized=bool(suffix))
        with (output / f"per_class_{task}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_id", "label", "precision_pct", "recall_pct", "f1_pct", "support"])
            for row in values["per_class"]:
                writer.writerow([row["class_id"], row["label"], row["precision"]*100,
                                 row["recall"]*100, row["f1"]*100, row["support"]])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(LABELS))
    parser.add_argument("--split", choices=sorted(SPLITS))
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--cohort-id")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        prediction_bytes = args.predictions.read_bytes()
        records = _read_jsonl(args.predictions, prediction_bytes)
        _require(bool(records), "empty predictions; coverage cannot be inferred")
        splits = {r.get("split") for r in records if isinstance(r.get("split"), str)}
        _require(len(splits) == 1, f"predictions require one split, got {splits}")
        split = args.split or next(iter(splits))
        _require(split in SPLITS, f"wrong split: {split!r}")
        run_dir = args.predictions.parent.parent
        provenance_path = run_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
        _require(isinstance(provenance, dict), "provenance.json must be a JSON object")
        # Prefer declared run tasks, so a missing whole MTL task cannot masquerade as ST.
        tasks = args.tasks or provenance.get("tasks")
        config = {}
        resolved = run_dir / "config.resolved.yaml"
        if resolved.exists():
            try:
                import yaml
            except ImportError as exc:
                raise EvaluationError("reading config.resolved.yaml requires PyYAML") from exc
            config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            _require(isinstance(config, dict), "config.resolved.yaml must be a mapping")
            tasks = tasks or config.get("tasks")
        for declared in (provenance.get("tasks"), config.get("tasks")):
            if declared is not None:
                _require(set(_task_list(tasks)) == set(_task_list(declared)),
                         "requested tasks conflict with declared run tasks")
        _require(tasks is not None,
                 "declare tasks via --tasks, provenance.json or config.resolved.yaml; observed predictions cannot prove task coverage")
        tasks = _task_list(tasks)
        task_source = "declared"
        manifest_bytes = args.dataset_manifest.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        digest = _manifest_digest(manifest)
        try:
            from neu_esc.data.dataset import load_split
        except ImportError as exc:
            raise EvaluationError("missing data API neu_esc.data.dataset.load_split; integrate module 01") from exc
        try:
            expected = list(load_split(args.dataset_manifest, split, "raw"))
        except ValueError as exc:
            raise EvaluationError(f"authoritative data loader rejected split: {exc}") from exc
        _require(args.dataset_manifest.read_bytes() == manifest_bytes,
                 "dataset manifest changed during evaluation")
        _require(manifest.get("row_counts", {}).get(split) == len(expected),
                 "authoritative split length differs from manifest row_counts")
        result = evaluate_records(records, expected, tasks, args.allow_invalid)
        result.update(
            run_id=args.run_id or provenance.get("run_id"),
            cohort_id=args.cohort_id or provenance.get("cohort_id"),
            seed=args.seed if args.seed is not None else provenance.get("seed", config.get("seed")),
            predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
            dataset_manifest_digest=digest,
            task_source=task_source,
            method_mode=provenance.get("method_mode") or config.get("method_mode"),
            backbone=provenance.get("backbone") or config.get("backbone"),
            variant=provenance.get("variant") or config.get("variant"),
        )
        for field in ("run_id", "cohort_id", "seed"):
            override = getattr(args, field)
            if override is not None and provenance.get(field) is not None:
                _require(override == provenance[field], f"{field} override conflicts with provenance")
        if args.seed is not None and config.get("seed") is not None:
            _require(args.seed == config["seed"], "seed override conflicts with config")
        known_digest = provenance.get("dataset_manifest_digest")
        _require(known_digest is None or known_digest == digest, "run/dataset manifest digest mismatch")
        for field in ("backbone", "variant", "method_mode", "seed"):
            if provenance.get(field) is not None and config.get(field) is not None:
                _require(provenance[field] == config[field], f"config/provenance conflict: {field}")
        _require(args.predictions.read_bytes() == prediction_bytes,
                 "predictions changed during evaluation; preserve a stable predictions file")
        write_evaluation(result, args.output, not args.no_png)
    except (EvaluationError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"evaluation failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "split": split,
                      "tasks": list(result["tasks"]), "task_source": task_source}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
