"""Return 2 when an existing persisted validation artifact cannot be aggregated."""
import argparse
import json
from pathlib import Path

from neu_esc.evaluation.aggregate import aggregate_cohort
from neu_esc.evaluation.evaluate import EvaluationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / "evaluation/validation/metrics.json").read_text(encoding="utf-8"))
    missing = [field for field in ("method_mode", "backbone", "variant") if not metrics.get(field)]
    try:
        aggregate_cohort([metrics], expected_seeds=[metrics["seed"]])
    except EvaluationError as error:
        print(json.dumps({"status": "FAIL", "owner_module": "02", "missing_fields": missing, "error": str(error)}))
        return 2
    if missing:
        print(json.dumps({"status": "FAIL", "missing_fields": missing}))
        return 2
    print(json.dumps({"status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
