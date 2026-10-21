"""Summarise runs as mean +- std over seeds and compare with NEU-ESC Table 6.

    python kaggle/aggregate.py runs/ other_account_runs/ --out results/

Reads every */metrics.json under the given directories (several Kaggle sessions or accounts
can be combined). Writes per_run.csv, summary.csv, gains.csv and summary.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# reports/reference/table6.csv rows 04/05 (single task), 07/08 (2-task), 10/11 (2-task & SMART).
PAPER = {
    ("visobert", "single"): {"sentiment": (82.78, 75.79, 82.17), "topic": (77.94, 60.65, 78.06)},
    ("phobert", "single"): {"sentiment": (81.75, 77.70, 82.02), "topic": (78.65, 62.73, 78.36)},
    ("visobert", "mtl"): {"sentiment": (82.63, 76.17, 82.58), "topic": (78.56, 61.66, 78.68)},
    ("phobert", "mtl"): {"sentiment": (82.88, 75.99, 82.85), "topic": (79.25, 62.11, 79.59)},
    ("visobert", "mtl_smart"): {"sentiment": (83.58, 77.17, 84.19), "topic": (79.80, 63.13, 80.55)},
    ("phobert", "mtl_smart"): {"sentiment": (83.26, 77.52, 83.66), "topic": (79.31, 62.79, 80.04)},
}
METRICS = ("accuracy", "macro_f1", "weighted_f1", "weighted_f1_author_mtl")
# Single-task rows in Table 6 combine two runs: sentiment from st_sentiment, topic from st_topic.
SETUP_GROUP = {"st_sentiment": "single", "st_topic": "single", "mtl": "mtl", "mtl_smart": "mtl_smart"}
GAINS = (("single", "mtl"), ("mtl", "mtl_smart"), ("single", "mtl_smart"))


def load_runs(directories, include_debug):
    runs, seen = [], {}
    for directory in directories:
        for path in sorted(Path(directory).rglob("metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED":
                continue
            if not metrics.get("benchmark_eligible") and not include_debug:
                continue
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            name = metrics["run_name"]
            if name in seen:
                raise SystemExit(f"duplicate run {name}: {seen[name]} and {path.parent}")
            seen[name] = path.parent
            for task, values in metrics["test"].items():
                runs.append({"run": name, "backbone": config["backbone"], "setup": config["setup"],
                             "group": SETUP_GROUP[config["setup"]], "task": task, "seed": config["seed"],
                             "best_epoch": metrics["best_epoch"], "epochs_run": metrics["epochs_run"],
                             "hours": metrics["timing"]["wall_seconds"] / 3600,
                             "runner_sha256": config.get("code", {}).get("runner_sha256"),
                             **{m: 100 * values[m] for m in METRICS}})
    return runs


def mean_std(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return float(values.mean()), std


def welch_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        return float("nan")
    return float(ttest_ind(b, a, equal_var=False).pvalue)


def fmt(mean, std):
    return f"{mean:.2f}" if math.isnan(std) else f"{mean:.2f} ± {std:.2f}"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-debug", action="store_true", help="also count smoke/debug runs")
    args = parser.parse_args(argv)

    runs = load_runs(args.run_dirs, args.include_debug)
    if not runs:
        raise SystemExit("no COMPLETED benchmark runs found (smoke/debug runs need --include-debug)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_run.csv", runs)

    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["backbone"], row["group"], row["task"])].append(row)

    summary, lines = [], [
        "| Backbone | Setup | Task | n | Acc | mF1 | wF1 | wF1 (author MTL eval) | Paper Acc / mF1 / wF1 "
        "| Δ Acc | Δ mF1 | Paper mF1 in seed range |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for (backbone, group, task), rows in sorted(grouped.items()):
        stats = {m: mean_std([r[m] for r in rows]) for m in METRICS}
        paper = PAPER.get((backbone, group), {}).get(task)
        f1s = [r["macro_f1"] for r in rows]
        record = {"backbone": backbone, "setup": group, "task": task, "n": len(rows),
                  "seeds": " ".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
                  "mean_hours": float(np.mean([r["hours"] for r in rows]))}
        for metric, (mean, std) in stats.items():
            record[f"{metric}_mean"], record[f"{metric}_std"] = mean, std
        if paper:
            record.update(paper_accuracy=paper[0], paper_macro_f1=paper[1], paper_weighted_f1=paper[2],
                          delta_accuracy=stats["accuracy"][0] - paper[0],
                          delta_macro_f1=stats["macro_f1"][0] - paper[1],
                          paper_macro_f1_in_range=min(f1s) <= paper[1] <= max(f1s))
        summary.append(record)
        author_wf1 = fmt(*stats["weighted_f1_author_mtl"]) if group != "single" else "—"
        paper_cells = (f"{paper[0]:.2f} / {paper[1]:.2f} / {paper[2]:.2f} | {record['delta_accuracy']:+.2f} | "
                       f"{record['delta_macro_f1']:+.2f} | {'yes' if record['paper_macro_f1_in_range'] else 'no'}"
                       if paper else "— | — | — | —")
        lines.append(f"| {backbone} | {group} | {task} | {len(rows)} | {fmt(*stats['accuracy'])} | "
                     f"{fmt(*stats['macro_f1'])} | {fmt(*stats['weighted_f1'])} | {author_wf1} | {paper_cells} |")
    write_csv(out / "summary.csv", summary)

    gains, gain_lines = [], ["| Backbone | Task | Comparison | Δ mF1 (ours) | Welch p | Δ mF1 (paper) |",
                             "|---|---|---|---:|---:|---:|"]
    for backbone in sorted({r["backbone"] for r in runs}):
        for task in ("sentiment", "topic"):
            for base, other in GAINS:
                a = [r["macro_f1"] for r in grouped.get((backbone, base, task), [])]
                b = [r["macro_f1"] for r in grouped.get((backbone, other, task), [])]
                if not a or not b:
                    continue
                ours = float(np.mean(b) - np.mean(a))
                paper = PAPER[(backbone, other)][task][1] - PAPER[(backbone, base)][task][1]
                p = welch_p(a, b)
                gains.append({"backbone": backbone, "task": task, "base": base, "other": other,
                              "delta_macro_f1": ours, "welch_p": p, "paper_delta_macro_f1": paper})
                gain_lines.append(f"| {backbone} | {task} | {base} → {other} | {ours:+.2f} | "
                                  f"{'—' if math.isnan(p) else f'{p:.3f}'} | {paper:+.2f} |")
    write_csv(out / "gains.csv", gains)

    # Every run should come from the same runner file; a mixed set is a different cohort.
    versions = defaultdict(set)
    for row in runs:
        versions[row["runner_sha256"] or "unknown"].add(row["run"])
    code_lines = [f"- `{sha[:16]}`: {len(names)} runs" for sha, names in sorted(versions.items())]
    if len(versions) > 1:
        code_lines.insert(0, "**WARNING: runs come from different runner versions; check per_run.csv "
                             "before treating them as one cohort.**")
        print(f"WARNING: {len(versions)} different runner versions", file=sys.stderr)

    text = ["# NEU-ESC reproduction summary", "",
            "Test-set metrics in %, mean ± sample std over seeds. Δ = ours − paper. "
            "The paper reports one run per row.", "", *lines, "",
            "## Gains between setups (macro-F1)", "", *gain_lines, "",
            "## Runner version (SHA-256 of neu_esc_kaggle.py)", "", *code_lines, ""]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
