import copy
import json

import pytest
import torch

from neu_esc.models import EncoderModel
from neu_esc.evaluation.evaluate import evaluate_records
from neu_esc.training.checkpoint import load_checkpoint, save_checkpoint, restore_tokenizer
from neu_esc.training.config import canonical_digest, validate_config, cohort_digest
from neu_esc.training.engine import TrainingEngine, prediction_records, seed_everything
from neu_esc.training.runtime import tiny_model_and_tokenizer, select_device
from conftest import variant_config


IDENTITY = {"config_hash": "fixture", "cohort_id": "tiny-smoke", "seed": 42}


def engine(config, directory, train_records, validation_records):
    seed_everything(42)
    model, tokenizer = tiny_model_and_tokenizer(config, directory / "tiny_assets")
    return TrainingEngine(model, tokenizer, config, train_records, validation_records,
                          run_dir=directory, identity=IDENTITY, evaluator=evaluate_records)


@pytest.mark.parametrize("variant", ["st_sentiment", "st_topic", "mtl", "mtl_smart", "mtl_mlm", "mtl_mlm_smart"])
def test_exact_optimizer_boundary_resume(config, tmp_path, train_records, validation_records, variant):
    config = variant_config(config, variant)
    continuous = engine(config, tmp_path / "continuous", train_records, validation_records)
    assert continuous.run() == "COMPLETED"
    final, _ = load_checkpoint(tmp_path / "continuous/checkpoints/latest", IDENTITY)
    interrupted = engine(config, tmp_path / "resumed", train_records, validation_records)
    assert interrupted.run(stop_after_steps=2) == "INTERRUPTED"
    saved, _ = load_checkpoint(tmp_path / "resumed/checkpoints/latest", IDENTITY)
    assert saved["position"]["batch_position"] == 4
    resumed = engine(config, tmp_path / "resumed", train_records, validation_records)
    resumed.resume()
    assert resumed.run() == "COMPLETED"
    actual, _ = load_checkpoint(tmp_path / "resumed/checkpoints/latest", IDENTITY)
    assert final["position"] == actual["position"]
    assert final["scheduler"] == actual["scheduler"]
    assert final["scaler"] == actual["scaler"]
    for key, tensor in final["model"].items():
        assert torch.equal(tensor, actual["model"][key]), key
    for key, state in final["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, actual["optimizer"]["state"][key][name])
    assert torch.equal(final["rng"]["torch"], actual["rng"]["torch"])
    assert final["selection"]["score"] == actual["selection"]["score"]
    # Compare selected predictions as well as final optimizer state.
    chosen, _ = load_checkpoint(tmp_path / "resumed/checkpoints/best", IDENTITY)
    model = EncoderModel.from_specification(chosen["model_spec"])
    model.load_state_dict(chosen["model"], strict=True)
    result = prediction_records(model, resumed.tokenizer, validation_records, config["max_length"], 3, "cpu")
    evaluate_records(result, validation_records, config["tasks"])
    assert [(r["sample_id"], r["task"]) for r in result] == [
        (r["sample_id"], t) for r in validation_records for t in config["tasks"]]


def test_checkpoint_roundtrip_and_rejection(config, tmp_path, train_records, validation_records):
    instance = engine(config, tmp_path / "run", train_records, validation_records)
    instance.save_latest()
    path = tmp_path / "run/checkpoints/latest"
    payload, digest = load_checkpoint(path, IDENTITY)
    assert len(digest) == 64
    reconstructed = EncoderModel.from_specification(payload["model_spec"])
    reconstructed.load_state_dict(payload["model"], strict=True)
    restore_tokenizer(payload["tokenizer_files"], tmp_path / "restored_tokenizer")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, {**IDENTITY, "seed": 43})
    incomplete = dict(payload)
    incomplete.pop("optimizer")
    with pytest.raises(ValueError, match="Incomplete"):
        save_checkpoint(tmp_path / "bad", incomplete)
    midbatch = copy.deepcopy(payload)
    midbatch["position"]["at_optimizer_boundary"] = False
    with pytest.raises(ValueError, match="optimizer boundaries"):
        save_checkpoint(tmp_path / "bad", midbatch)
    marker = json.loads((path / "COMPLETE.json").read_text())
    (path / marker["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="missing COMPLETE"):
        load_checkpoint(tmp_path / "missing")


def test_data_splits_and_configuration_gates(config, tmp_path, train_records, validation_records):
    changed = copy.deepcopy(config)
    changed["training"]["mode"] = "full"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA GPU"):
            select_device(changed)
    assert select_device(config) == "cpu"
    broken = copy.deepcopy(config)
    broken["model_revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        validate_config(broken)
    broken = copy.deepcopy(config)
    broken["tasks"] = ["topic"]
    with pytest.raises(ValueError, match="Variant/task"):
        validate_config(broken)
    broken = copy.deepcopy(config)
    broken["backbone"] = "phobert"
    with pytest.raises(ValueError, match="segmented"):
        validate_config(broken)
    different_seed = copy.deepcopy(config)
    different_seed["seed"] = 43
    assert canonical_digest(config) != canonical_digest(different_seed)
    assert cohort_digest(config, "data", "code") == cohort_digest(different_seed, "data", "code")
    assert cohort_digest(config, "data", "code") != cohort_digest(config, "changed", "code")
    with pytest.raises(ValueError, match="validation only"):
        engine(config, tmp_path / "bad", train_records, [{**r, "split": "test"} for r in validation_records])


def test_earliest_epoch_tie_and_empty_mlm_branch(config, tmp_path, train_records, validation_records, monkeypatch):
    instance = engine(config, tmp_path / "run", train_records, validation_records)
    monkeypatch.setattr(instance, "validate", lambda: ([], {"tasks": {}}, 0.0))
    instance.run()
    assert instance.selection["best_epoch"] == 1
    assert instance.selection["score"] == 0.0  # Even a zero initial metric creates a best checkpoint.
    # Reuse the real supervised path with only special tokens: MLM forward must be skipped.
    instance.encoded = {"input_ids": torch.tensor([[2, 3, 0]] * len(train_records)),
                        "attention_mask": torch.tensor([[1, 1, 0]] * len(train_records))}
    monkeypatch.setattr(instance.model, "mlm_logits", lambda *args: pytest.fail("No-target MLM forward should be skipped"))
    instance.model.zero_grad(set_to_none=True)
    loss = instance.objective([0, 1])
    assert torch.isfinite(loss)
    loss.backward()
    assert instance.model.mlm_head[-1].weight.grad is None
    assert instance.model.heads["sentiment"].weight.grad is not None
