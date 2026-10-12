"""Cross-module checks on actual inputs; no GPU, training or API requests."""
import copy
import json
from pathlib import Path

import pytest
import torch

from neu_esc.data.dataset import load_split
from neu_esc.evaluation.evaluate import evaluate_records, EvaluationError
from neu_esc.training.config import load_config, cohort_digest, file_digest
from neu_esc.training.engine import seed_everything
from neu_esc.training.runtime import tiny_model_and_tokenizer
from neu_esc.losses.smart import smart_loss

ROOT = Path(__file__).resolve().parents[2]


def test_22_configs_locked_to_actual_manifest():
    manifest = json.loads((ROOT / "data/manifests/dataset.json").read_text(encoding="utf-8"))
    paths = sorted((ROOT / "configs/experiments").glob("*/*.yaml"))
    assert len(paths) == 22
    inventory = json.loads((ROOT / "reports/method/delivery_inventory.json").read_text(encoding="utf-8"))
    for path in paths:
        config = load_config(path)
        assert config["dataset_manifest_digest"] == manifest["manifest_digest"]
        assert config["training"]["mode"] == "full"
        assert not Path(config["dataset_manifest"]).is_absolute()
        key = path.relative_to(ROOT).as_posix()
        assert file_digest(path) == inventory["files"][key]
        assert cohort_digest(config, manifest["manifest_digest"], "implementation") != cohort_digest(config, "changed", "implementation")


def test_p0_continuous_gradient_is_real(tmp_path):
    seed_everything(42)
    config = load_config(ROOT / "tests/training/fixtures/tiny.yaml")
    model, tokenizer = tiny_model_and_tokenizer(config, tmp_path / "assets")
    encoded = tokenizer(["word1 word2 word3", "word4 word5"], padding="max_length", max_length=12, return_tensors="pt")
    loss, diagnostics = smart_loss(model, encoded["input_ids"], encoded["attention_mask"], tokenizer.all_special_ids,
                                   config["smart"], return_diagnostics=True)
    loss.backward()
    assert model.training
    for task, result in diagnostics.items():
        assert all(value > 0 for value in result["noise_gradient_norms"])
        assert torch.isfinite(result["noise"]).all()
        assert (result["noise"] * ~result["eligible_mask"]).eq(0).all()
        assert result["noise"].abs().max() <= config["smart"]["epsilon"]
        assert model.heads[task].weight.grad.abs().sum() > 0
    assert model.encoder.get_input_embeddings().weight.grad.abs().sum() > 0


@pytest.mark.dataset
def test_real_test_ids_permutation_and_source_hashes():
    manifest_path = ROOT / "data/manifests/dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (manifest_path.parent / manifest["processed_files"]["test"]).exists():
        pytest.skip("Real dataset cache unavailable; never replace it with a fixture")
    records = load_split(manifest_path, "test", "raw")
    assert len(records) == 6613
    assert file_digest(manifest_path.parent / manifest["raw_files"]["test"]) == manifest["raw_sha256"]["test"]
    # Oracle labels only check identity/denominator; NOT model predictions or a benchmark.
    diagnostic = [dict(sample_id=r["sample_id"], split="test", task=t,
                       y_true=r[t], y_pred=r[t], status="ok", error=None)
                  for r in records for t in ("sentiment", "topic")]
    original = evaluate_records(diagnostic, records, ["sentiment", "topic"])
    assert original == evaluate_records(list(reversed(diagnostic)), records[1:]+records[:1], ["sentiment", "topic"])
    for t, k in (("sentiment", 4), ("topic", 10)):
        cm = original["tasks"][t]["confusion_matrix"]
        assert len(cm) == k and sum(map(sum, cm)) == 6613
        assert sum(cm[i][i] for i in range(k))/6613 == original["tasks"][t]["accuracy"]
    wrong = copy.deepcopy(diagnostic)
    wrong[0]["y_true"] = (wrong[0]["y_true"]+1) % 4
    with pytest.raises(EvaluationError, match="wrong labels"):
        evaluate_records(wrong, records, ["sentiment", "topic"])
