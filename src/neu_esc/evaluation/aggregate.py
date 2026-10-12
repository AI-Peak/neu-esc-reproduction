"""Summarize predeclared seeds without mixing cohorts or selecting on test."""

from __future__ import annotations

import math
import csv
from pathlib import Path
from collections.abc import Mapping, Sequence
from numbers import Integral
from statistics import mean, stdev

from .evaluate import EvaluationError, LABELS, METRICS, SCHEMA_VERSION, _require, metrics_from_counts


def _paper_value(paper_metrics, task, metric):
    if paper_metrics is None:
        return None
    _require(task in paper_metrics and metric in paper_metrics[task],
             f"missing paper metric: {task}/{metric}")
    value = paper_metrics[task][metric]
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and 0 <= value <= 100,
             f"paper metric must be a percentage: {task}/{metric}")
    return float(value)


def aggregate_cohort(evaluations: Sequence[Mapping], *, paper_metrics=None,
                     expected_seeds=None, cm_seed: int = 42) -> dict:
    """Return per-seed metrics, mean/sample std, paper deltas and sum CMs.

    paper_metrics: {task: {accuracy/macro_f1/weighted_f1: percentage}}.
    expected_seeds, when supplied, is an exact predeclared set, not a seed filter.
    cm_seed is predeclared 42 by default; absence is reported, never substituted.
    A sum CM describes N across seeds, not the mean of per-seed metrics.
    """
    evaluations = list(evaluations)
    _require(bool(evaluations), "cannot aggregate zero runs")
    first = evaluations[0]
    fields = ("cohort_id", "method_mode", "dataset_manifest_digest", "split")
    for field in fields:
        _require(isinstance(first.get(field), str) and bool(first[field]),
                 f"aggregation requires nonempty {field}")
    _require(first["method_mode"] in ("paper_method", "reference_code"), "unknown method_mode")
    tasks = list(first.get("tasks", {}))
    _require(bool(tasks) and all(t in LABELS for t in tasks), "unknown/empty task set")
    seeds, run_ids = set(), set()
    for result in evaluations:
        for field in fields:
            _require(result.get(field) == first[field], f"cannot mix {field} across runs")
        _require(set(result.get("tasks", {})) == set(tasks), "cannot mix task sets across runs")
        seed, run_id = result.get("seed"), result.get("run_id")
        _require(isinstance(seed, Integral) and not isinstance(seed, bool), "aggregation requires integer seed")
        _require(seed not in seeds, f"duplicate seed: {seed}")
        _require(isinstance(run_id, str) and bool(run_id), "aggregation requires run_id")
        _require(run_id not in run_ids, f"duplicate run_id: {run_id}")
        seeds.add(seed)
        run_ids.add(run_id)
        for task in tasks:
            value = result["tasks"][task]
            recomputed = metrics_from_counts(value["confusion_matrix"], task)
            for metric in METRICS:
                score = value[metric]
                _require(isinstance(score, (int, float)) and not isinstance(score, bool)
                         and math.isfinite(score) and 0 <= score <= 1,
                         f"invalid metric: {task}/{metric}")
                _require(math.isclose(score, recomputed[metric], rel_tol=0, abs_tol=1e-12),
                         f"metric/CM mismatch: {task}/{metric}")
            for key in ("n_samples", "invalid_count", "label_order", "label_names", "confusion_columns"):
                _require(value[key] == recomputed[key], f"CM metadata mismatch: {task}/{key}")
            _require(value["n_samples"] == first["tasks"][task]["n_samples"],
                     f"cannot mix sample counts for {task}")
            _require(value["confusion_columns"] == first["tasks"][task]["confusion_columns"],
                     f"cannot mix CM columns for {task}")
            _require([sum(r) for r in value["confusion_matrix"]] ==
                     [sum(r) for r in first["tasks"][task]["confusion_matrix"]],
                     f"cannot mix authoritative supports for {task}")
    if expected_seeds is not None:
        declared = list(expected_seeds)
        _require(all(isinstance(s, Integral) and not isinstance(s, bool) for s in declared)
                 and len(set(declared)) == len(declared) and bool(declared), "invalid expected_seeds")
        _require(seeds == set(declared),
                 f"seed coverage mismatch: missing={sorted(set(declared)-seeds)}, extra={sorted(seeds-set(declared))}")
    ordered = sorted(evaluations, key=lambda r: r["seed"])
    summaries = {}
    for task in tasks:
        seed_values = [{"seed": r["seed"], "run_id": r["run_id"],
                        **{m: r["tasks"][task][m] for m in METRICS},
                        "n_samples": r["tasks"][task]["n_samples"],
                        "invalid_count": r["tasks"][task]["invalid_count"]} for r in ordered]
        metrics = {}
        for metric in METRICS:
            values = [r[metric] for r in seed_values]
            avg = mean(values)
            std = stdev(values) if len(values) > 1 else None
            paper = _paper_value(paper_metrics, task, metric)
            metrics[metric] = dict(n=len(values), mean=avg, std=std, ddof=1,
                                   mean_pct=avg*100, std_pct=std*100 if std is not None else None,
                                   paper_pct=paper, delta_pp=avg*100-paper if paper is not None else None)
        base = first["tasks"][task]["confusion_matrix"]
        summed = [[sum(r["tasks"][task]["confusion_matrix"][i][j] for r in ordered)
                   for j in range(len(base[0]))] for i in range(len(base))]
        aggregate_n = sum(map(sum, summed))
        assert aggregate_n == sum(v["n_samples"] for v in seed_values)
        selected = next((r for r in ordered if r["seed"] == cm_seed), None)
        summaries[task] = dict(
            per_seed=seed_values, metrics=metrics,
            per_seed_confusion_matrices=[dict(seed=r["seed"], run_id=r["run_id"],
                                             n_samples=r["tasks"][task]["n_samples"],
                                             counts=r["tasks"][task]["confusion_matrix"]) for r in ordered],
            figure3_comparison=dict(seed=cm_seed, status="available" if selected else "not_available",
                                    n_samples=selected["tasks"][task]["n_samples"] if selected else None,
                                    counts=selected["tasks"][task]["confusion_matrix"] if selected else None),
            aggregate_confusion=dict(n_samples=aggregate_n, counts=summed,
                                     label_order=first["tasks"][task]["label_order"],
                                     columns=first["tasks"][task]["confusion_columns"],
                                     description="Sum of per-seed counts; not mean metrics; compare paper using seed 42"),
        )
    return dict(schema_version=SCHEMA_VERSION, **{f: first[f] for f in fields},
                n_seeds=len(ordered), seeds=[r["seed"] for r in ordered], std_ddof=1,
                status="complete" if expected_seeds is not None else "observed_seeds_only",
                tasks=summaries)


def combine_st_runs(sentiment_runs, topic_runs, *, backbone: str, pairing_id: str,
                    paper_metrics=None, expected_seeds=None) -> dict:
    """Pair two ST cohorts by seed for one paper row; retain both cohort IDs.

    pairing_id is an explicit campaign declaration that these separately trained
    configurations belong to one ST row. It must not be selected using test scores.
    Task-specific configs have different hashes/cohort IDs by design.
    """
    _require(isinstance(pairing_id, str) and bool(pairing_id), "ST pairing_id is required")
    inputs = {"sentiment": list(sentiment_runs), "topic": list(topic_runs)}
    aggregates = {}
    for task, results in inputs.items():
        for result in results:
            _require(set(result.get("tasks", {})) == {task}, f"ST {task} requires a single-task run")
            _require(result.get("variant") == f"st_{task}", f"wrong ST variant for {task}")
            _require(result.get("backbone") == backbone, f"wrong ST backbone for {task}")
        aggregates[task] = aggregate_cohort(results, paper_metrics=paper_metrics, expected_seeds=expected_seeds)
    left, right = aggregates["sentiment"], aggregates["topic"]
    for field in ("seeds", "method_mode", "dataset_manifest_digest", "split"):
        _require(left[field] == right[field], f"ST pair mismatch: {field}")
    _require({r["run_id"] for r in inputs["sentiment"]}.isdisjoint(
        {r["run_id"] for r in inputs["topic"]}), "ST tasks require two distinct runs per seed")
    _require(left["tasks"]["sentiment"]["per_seed"][0]["n_samples"] ==
             right["tasks"]["topic"]["per_seed"][0]["n_samples"], "ST pair sample count mismatch")
    return dict(schema_version=SCHEMA_VERSION, backbone=backbone, variant="single_task_pair",
                pairing_id=pairing_id, cohort_ids={t: a["cohort_id"] for t,a in aggregates.items()},
                seeds=left["seeds"], n_seeds=left["n_seeds"], method_mode=left["method_mode"],
                dataset_manifest_digest=left["dataset_manifest_digest"], split=left["split"],
                status=left["status"], tasks={t:a["tasks"][t] for t,a in aggregates.items()})


def paper_metrics_from_row(row: Mapping) -> dict:
    """Convert one table6.csv row to aggregate_cohort's percentage inputs."""
    return {task: {metric: float(row[f"{task}_{metric}_pct"]) for metric in METRICS}
            for task in LABELS}


def summary_rows(summary: Mapping, *, row_id: str, backbone: str, variant: str) -> list[dict]:
    """Long-form report rows: percent mean/std/paper; delta in percentage points.

    A paired ST row retains its task-specific cohort ID and explicit pairing ID.
    Missing tasks/paper comparisons are not fabricated as zero scores.
    """
    rows = []
    for task, values in summary["tasks"].items():
        cohort = summary.get("cohort_id") or summary.get("cohort_ids", {}).get(task)
        for metric, stats in values["metrics"].items():
            rows.append(dict(row_id=row_id, backbone=backbone, variant=variant, task=task,
                             metric=metric, n_seeds=stats["n"], mean_pct=stats["mean_pct"],
                             std_pct=stats["std_pct"], paper_pct=stats["paper_pct"],
                             delta_pp=stats["delta_pp"], method_mode=summary["method_mode"],
                             cohort_id=cohort, pairing_id=summary.get("pairing_id"),
                             split=summary["split"], status=summary["status"]))
    return rows


def write_summary_csv(summary: Mapping, output, *, row_id: str, backbone: str, variant: str) -> None:
    rows = summary_rows(summary, row_id=row_id, backbone=backbone, variant=variant)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards against concurrent report writers.
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
