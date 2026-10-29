"""Summarise the neu_esc_caitien.py factorial cohort against the reproduced baselines.

    python src_caitien/aggregate_caitien.py runs/caitien runs/s42 runs/s43 ... --out reports/caitien_results

Reads every */metrics.json under the given directories: factorial runs (config "improvement") and
the baseline runs (st_sentiment, st_topic, mtl, mtl_smart from neu_esc_kaggle.py). Writes
per_run.csv and summary.md with:
  - mean +- std per setup, with deltas and Welch p against single-task and against sep_sum_ce;
  - main effects of each factor, paired by seed and by the other two factors;
  - joint-argmax inference for factorized heads, surgery diagnostics, the accuracy targets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ("sentiment", "topic")
METRICS = ("accuracy", "macro_f1")
FACTORIAL = [f"{h}_{o}_{l}" for h in ("sep", "fac") for o in ("sum", "pcgrad", "patgrad") for l in ("ce", "cbf")]
BASELINE = ("single", "mtl", "mtl_smart")
TARGET_ACC = {"sentiment": 83.7, "topic": 79.8}  # project goal: above the paper's best 2-task accuracy
FACTORS = {"head": (0, ("sep", "fac")), "opt": (1, ("sum", "pcgrad", "patgrad")), "loss": (2, ("ce", "cbf"))}


def load(directories):
    runs, epochs, seen = [], {}, {}
    for directory in directories:
        for path in sorted(Path(directory).rglob("metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED" or not metrics.get("benchmark_eligible"):
                continue
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            name = metrics["run_name"]
            if name in seen:
                raise SystemExit(f"duplicate run {name}: {seen[name]} and {path.parent}")
            seen[name] = path.parent
            setup = config["setup"]
            group = "single" if setup.startswith("st_") else setup
            epochs[name] = [json.loads(line) for line in
                            (path.parent / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
            for task, values in metrics["test"].items():
                row = {"run": name, "backbone": config["backbone"], "setup": group, "task": task,
                       "seed": config["seed"], "best_epoch": metrics["best_epoch"],
                       "hours": metrics["timing"]["wall_seconds"] / 3600,
                       "runner_sha256": config.get("code", {}).get("runner_sha256"),
                       **{m: 100 * values[m] for m in METRICS}}
                joint = (metrics.get("joint_decode") or {}).get("test", {}).get(task)
                if joint:
                    row.update({f"joint_{m}": 100 * joint[m] for m in METRICS})
                    val_marg = metrics["val"][task]["macro_f1"]
                    val_joint = metrics["joint_decode"]["val"][task]["macro_f1"]
                    row["val_prefers_joint"] = val_joint > val_marg
                runs.append(row)
    return runs, epochs


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def fmt(values):
    if not len(values):
        return "—"
    mean, std = mean_std(values)
    return f"{mean:.2f}" if math.isnan(std) else f"{mean:.2f} ± {std:.2f}"


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    from scipy.stats import ttest_ind
    return float(ttest_ind(b, a, equal_var=False).pvalue)


def paired_p(diffs):
    if len(diffs) < 2 or np.allclose(diffs, diffs[0]):
        return float("nan")
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(diffs, 0.0).pvalue)


def pfmt(p):
    return "—" if math.isnan(p) else f"{p:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    runs, epochs = load(args.run_dirs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    columns = sorted({k for r in runs for k in r})
    with open(out / "per_run.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(runs, key=lambda r: (r["backbone"], r["setup"], r["task"], r["seed"])))

    cell = defaultdict(dict)  # (backbone, setup, task) -> seed -> row
    for r in runs:
        cell[(r["backbone"], r["setup"], r["task"])][r["seed"]] = r
    backbones = sorted({r["backbone"] for r in runs})
    versions = sorted({r["runner_sha256"][:12] for r in runs if r["setup"] in FACTORIAL and r["runner_sha256"]})

    lines = ["# Improvement cohort (neu_esc_caitien.py) vs reproduced baselines", "",
             "Test-set metrics in %, mean ± sample std over seeds. Δ columns are macro-F1 / accuracy "
             "differences of means; p is Welch's t-test over seeds. sep_sum_ce is the baseline 2-task "
             "setup re-run by the improvement runner. Accuracy targets: sentiment > 83.7, topic > 79.8.",
             "", f"Factorial runner version(s): {', '.join(versions) or '—'}", ""]

    for metric_name, metric in (("macro-F1", "macro_f1"), ("accuracy", "accuracy")):
        lines += [f"## Test {metric_name}", "",
                  "| Backbone | Task | Setup | n | Acc | mF1 | Δ vs single | p | Δ vs sep_sum_ce | p | Acc target |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"] if metric == "macro_f1" else [
                  f"## Test {metric_name}: deltas", "",
                  "| Backbone | Task | Setup | Δ acc vs single | p | Δ acc vs sep_sum_ce | p |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for bb in backbones:
            for task in TASKS:
                single = [r[metric] for r in cell[(bb, "single", task)].values()]
                control = [r[metric] for r in cell[(bb, "sep_sum_ce", task)].values()]
                for setup in BASELINE + tuple(FACTORIAL):
                    rows = list(cell[(bb, setup, task)].values())
                    if not rows:
                        continue
                    values = [r[metric] for r in rows]
                    d_single = np.mean(values) - np.mean(single) if single else float("nan")
                    d_ctrl = np.mean(values) - np.mean(control) if control and setup in FACTORIAL else float("nan")
                    p_single = welch(single, values) if setup != "single" else float("nan")
                    p_ctrl = welch(control, values) if setup in FACTORIAL and setup != "sep_sum_ce" else float("nan")
                    show = lambda d: "—" if math.isnan(d) else f"{d:+.2f}"
                    if metric == "macro_f1":
                        acc = [r["accuracy"] for r in rows]
                        hit = "yes" if np.mean(acc) > TARGET_ACC[task] else "no"
                        lines.append(f"| {bb} | {task} | {setup} | {len(rows)} | {fmt(acc)} | {fmt(values)} | "
                                     f"{show(d_single)} | {pfmt(p_single)} | {show(d_ctrl)} | {pfmt(p_ctrl)} | {hit} |")
                    else:
                        lines.append(f"| {bb} | {task} | {setup} | {show(d_single)} | {pfmt(p_single)} | "
                                     f"{show(d_ctrl)} | {pfmt(p_ctrl)} |")
        lines.append("")

    lines += ["## Main effects (paired by seed and the other two factors)", "",
              "Each difference compares two setups that differ only in this factor, for the same seed; "
              "p is a one-sample t-test of those differences against 0.", "",
              "| Backbone | Task | Contrast | pairs | Δ mF1 | p | Δ acc | p |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for bb in backbones:
        for task in TASKS:
            for factor, (position, levels) in FACTORS.items():
                for level in levels[1:]:
                    diffs = {m: [] for m in METRICS}
                    for setup in FACTORIAL:
                        parts = setup.split("_")
                        if parts[position] != levels[0]:
                            continue
                        other = "_".join(parts[:position] + [level] + parts[position + 1:])
                        base, alt = cell[(bb, setup, task)], cell[(bb, other, task)]
                        for seed in sorted(set(base) & set(alt)):
                            for m in METRICS:
                                diffs[m].append(alt[seed][m] - base[seed][m])
                    if not diffs["macro_f1"]:
                        continue
                    lines.append(f"| {bb} | {task} | {factor}: {levels[0]} → {level} | {len(diffs['macro_f1'])} | "
                                 f"{fmt(diffs['macro_f1'])} | {pfmt(paired_p(diffs['macro_f1']))} | "
                                 f"{fmt(diffs['accuracy'])} | {pfmt(paired_p(diffs['accuracy']))} |")
    lines.append("")

    lines += ["## Factorized head: marginal vs joint-argmax inference (test)", "",
              "| Backbone | Task | Setup | mF1 marginal | mF1 joint | Acc marginal | Acc joint | val prefers joint |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for bb in backbones:
        for task in TASKS:
            for setup in FACTORIAL:
                rows = [r for r in cell[(bb, setup, task)].values() if "joint_macro_f1" in r]
                if rows:
                    lines.append(f"| {bb} | {task} | {setup} | {fmt([r['macro_f1'] for r in rows])} | "
                                 f"{fmt([r['joint_macro_f1'] for r in rows])} | {fmt([r['accuracy'] for r in rows])} | "
                                 f"{fmt([r['joint_accuracy'] for r in rows])} | "
                                 f"{sum(r['val_prefers_joint'] for r in rows)}/{len(rows)} |")
    lines.append("")

    lines += ["## Gradient conflict diagnostics (surgery runs, mean over epochs and seeds)", "",
              "global = whole encoder; per layer = share of (step, layer group) pairs with cos(g_s, g_t) < 0.", "",
              "| Backbone | Setup | global cos | global conflict rate | per-layer conflict rate | "
              "projections / step | PaTGrad protects sentiment | layers chosen (seed 42) |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for bb in backbones:
        for setup in FACTORIAL:
            if "_sum_" in setup:
                continue
            names = [name for name in epochs if name.startswith(f"{bb}-{setup}-s")]
            records = [e["surgery"] for name in names for e in epochs[name] if e.get("surgery")]
            if not records:
                continue
            avg = lambda key: np.nanmean([r[key] for r in records if r.get(key) is not None] or [np.nan])
            chosen = next((e["surgery"].get("selected_groups") for e in epochs.get(f"{bb}-{setup}-s42", [])
                           if e.get("surgery", {}).get("selected_groups")), None)
            protect = avg("protect_sentiment_rate")
            lines.append(f"| {bb} | {setup} | {avg('global_cos_mean'):.3f} | {avg('global_conflict_rate'):.2f} | "
                         f"{avg('group_conflict_rate'):.2f} | {avg('projections_per_step'):.2f} | "
                         f"{'—' if math.isnan(protect) else f'{protect:.2f}'} | {', '.join(chosen) if chosen else '—'} |")
    lines.append("")

    hours = defaultdict(list)
    for r in runs:
        if r["task"] == "sentiment" and r["setup"] in FACTORIAL:
            hours[r["setup"]].append(r["hours"])
    lines += ["## GPU hours per run (T4)", "", "| Setup | mean h |", "|---|---:|"]
    lines += [f"| {s} | {np.mean(hours[s]):.2f} |" for s in FACTORIAL if hours[s]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(runs)} task rows from {len(epochs)} runs -> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
