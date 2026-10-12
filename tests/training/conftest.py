from pathlib import Path
import copy

import pytest

from neu_esc.training.config import load_config
from neu_esc.training.engine import seed_everything
from neu_esc.training.runtime import tiny_model_and_tokenizer


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "tests/training/fixtures/tiny.yaml")


def records(split, count=9):
    return [{"sample_id": f"{split}:{i:06d}:fixture", "split": split, "source_row": i,
             "text": f"word{i % 20} word2 word3 word4 word5", "sentiment": i % 4, "topic": i % 10}
            for i in range(count)]


@pytest.fixture
def train_records():
    return records("train")


@pytest.fixture
def validation_records():
    return records("validation", 7)


@pytest.fixture
def tiny(config, tmp_path):
    seed_everything(42)
    return tiny_model_and_tokenizer(config, tmp_path / "tiny_assets")


def variant_config(config, variant):
    value = copy.deepcopy(config)
    value["variant"] = variant
    value["tasks"] = [variant[3:]] if variant.startswith("st_") else ["sentiment", "topic"]
    value["training"]["task_weights"] = {t: 1.0 for t in value["tasks"]}
    value["training"]["st_native_head_dropout"] = 0.1 if variant.startswith("st_") else None
    value["mlm"]["enabled"] = "mlm" in variant
    value["smart"]["enabled"] = "smart" in variant
    return value
