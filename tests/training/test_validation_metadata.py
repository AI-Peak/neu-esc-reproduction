import json

import pytest

from neu_esc.evaluation.evaluate import evaluate_records
from neu_esc.training.runtime import publish_validation


def inputs(config, validation_records):
    predictions = [{"sample_id": row["sample_id"], "split": "validation", "task": task,
                    "y_true": row[task], "y_pred": row[task], "status": "ok", "error": None}
                   for row in validation_records for task in config["tasks"]]
    metrics = evaluate_records(predictions, validation_records, config["tasks"])
    provenance = {"run_id": "metadata-fixture", "cohort_id": "metadata-fixture-cohort",
                  "seed": 42, "dataset_manifest_digest": "fixture",
                  **{field: config[field] for field in ("method_mode", "backbone", "variant")}}
    return metrics, provenance


def test_validation_metadata_falls_back_to_resolved_config(config, validation_records, tmp_path):
    metrics, provenance = inputs(config, validation_records)
    for field in ("method_mode", "backbone", "variant"):
        provenance.pop(field)
    publish_validation(metrics, tmp_path, provenance, "fixture-predictions-hash", config=config)
    stored = json.loads((tmp_path / "evaluation/validation/metrics.json").read_text(encoding="utf-8"))
    for field in ("method_mode", "backbone", "variant"):
        assert stored[field] == config[field]
    assert stored["tasks"] == metrics["tasks"]


@pytest.mark.parametrize("field", ["method_mode", "backbone", "variant"])
def test_validation_metadata_conflicts_fail_before_publication(config, validation_records, tmp_path, field):
    metrics, provenance = inputs(config, validation_records)
    provenance[field] = "conflicting-value"
    with pytest.raises(ValueError, match=f"config/provenance conflict: {field}"):
        publish_validation(metrics, tmp_path, provenance, "fixture-predictions-hash", config=config)
    assert not (tmp_path / "evaluation").exists()
