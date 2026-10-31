"""Summarise CR-SD (selfdistill) and CR-SD + D (crsd_ens) runs against the reproduced baselines.

    python src_caitien/aggregate_caitien.py runs/caitien runs/s42 ... runs/gap/a --out reports/crsd_results

Reads every */metrics.json under the given directories (fold runs, debug runs and fp32 runs are
skipped). Writes per_run.csv and summary.md with mean +- std per backbone/setup, the EMA student
ablation, and each improvement's delta against mtl and mtl_smart with Welch's p over seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ("sentiment", "topic")
BASELINES = ("single", "mtl", "mtl_smart", "mtl_mlm", "mtl_mlm_smart")
IMPROVEMENTS = ("selfdistill", "crsd_ens")
# Paper Table 6 accuracy (sentiment, topic), for reference only.
PAPER = {("phobert", "single"): (81.75, 78.65), ("phobert", "mtl"): (82.88, 79.25),
         ("phobert", "mtl_smart"): (83.26, 79.31), ("phobert", "mtl_mlm"): (82.67, 78.48),
         ("phobert", "mtl_mlm_smart"): (83.66, 79.49), ("visobert", "single"): (82.78, 77.94),
         ("visobert", "mtl"): (82.63, 78.56), ("visobert", "mtl_smart"): (83.58, 79.80),
         ("visobert", "mtl_mlm"): (82.23, 78.38), ("visobert", "mtl_mlm_smart"): (83.12, 79.25)}


def load(directories):
    rows, seen = [], set()
    for directory in directories:
        for path in sorted(Path(directory).rglob("metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED" or not metrics.get("benchmark_eligible"):
                continue
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            setup = "single" if config["setup"].startswith("st_") else config["setup"]
            if config.get("fold") or setup not in BASELINES + IMPROVEMENTS:
                continue
            key = (config["backbone"], config["setup"], config["seed"])
            if key in seen:  # the same run copied into two folders
                continue
            seen.add(key)
            student = metrics.get("student") or {}
            for task, values in metrics["test"].items():
                rows.append({"backbone": config["backbone"], "setup": setup, "task": task,
                             "seed": config["seed"], "acc": 100 * values["accuracy"],
                             "mf1": 100 * values["macro_f1"],
                             "student_acc": 100 * student["test"][task]["accuracy"] if student else None,
                             "run": path.parent.name})
    return rows


def fmt(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return "—"
    return f"{values.mean():.2f}" if len(values) == 1 else f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return "—"
    from scipy.stats import ttest_ind
    return f"{ttest_ind(b, a, equal_var=False).pvalue:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = load(args.dirs)
    cell = defaultdict(list)
    for r in rows:
        cell[(r["backbone"], r["setup"], r["task"])].append(r)

    lines = ["# CR-SD results against the reproduced baselines", "",
             "Test accuracy in %, mean ± std over seeds, validation selection. Paper = Table 6 (one run).", "",
             "| Backbone | Setup | n | Sentiment | Topic | Paper |", "|---|---|---:|---:|---:|---:|"]
    for backbone in ("phobert", "visobert"):
        for setup in BASELINES + IMPROVEMENTS:
            got = {t: [r["acc"] for r in cell[(backbone, setup, t)]] for t in TASKS}
            if not any(got.values()):
                continue
            paper = PAPER.get((backbone, setup))
            lines.append(f"| {backbone} | {setup} | {max(len(v) for v in got.values())} | {fmt(got['sentiment'])} | "
                         f"{fmt(got['topic'])} | {f'{paper[0]:.2f} / {paper[1]:.2f}' if paper else '—'} |")
    lines += ["", "## Improvement vs baseline (Δ of means, Welch p)", "",
              "| Backbone | Setup | Task | n | EMA (reported) | Student | Δ vs mtl | p | Δ vs mtl_smart | p |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for backbone in ("phobert", "visobert"):
        for setup in IMPROVEMENTS:
            for task in TASKS:
                runs = cell[(backbone, setup, task)]
                if not runs:
                    continue
                acc = [r["acc"] for r in runs]
                student = [r["student_acc"] for r in runs if r["student_acc"] is not None]
                cols = []
                for base in ("mtl", "mtl_smart"):
                    ref = [r["acc"] for r in cell[(backbone, base, task)]]
                    cols += [f"{np.mean(acc) - np.mean(ref):+.2f}" if ref else "—", welch(ref, acc)]
                lines.append(f"| {backbone} | {setup} | {task} | {len(acc)} | {fmt(acc)} | {fmt(student)} | "
                             + " | ".join(cols) + " |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "per_run.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
