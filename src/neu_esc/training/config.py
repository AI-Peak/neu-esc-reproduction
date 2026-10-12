from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import yaml

TASKS = {"st_sentiment": ["sentiment"], "st_topic": ["topic"],
         **{v: ["sentiment", "topic"] for v in
            ("mtl", "mtl_smart", "mtl_mlm", "mtl_mlm_smart")}}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_digest():
    """Include executable data/evaluator/package code and 02 code, exclude caches."""
    root = Path(__file__).resolve().parents[1]
    files = {p.relative_to(root).as_posix(): file_digest(p)
             for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
    return canonical_digest(files)


def validate_config(config):
    required = {"schema_version", "experiment_id", "backbone", "model_id", "model_revision",
                "tokenizer_id", "tokenizer_revision", "variant", "tasks", "method_mode",
                "dataset_manifest", "dataset_manifest_digest", "text_variant", "max_length",
                "seed", "training", "smart", "mlm", "selection", "assumptions"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if config["schema_version"] != "1.0":
        raise ValueError("Unsupported config schema")
    if config["variant"] not in TASKS or config["tasks"] != TASKS[config["variant"]]:
        raise ValueError("Variant/task mismatch")
    if config["method_mode"] not in ("paper_method", "reference_code"):
        raise ValueError("Invalid method_mode")
    if config["method_mode"] == "reference_code" and config["variant"] != "mtl_smart":
        raise ValueError("reference_code supports only the isolated two-task SMART diagnostic")
    if config["smart"]["enabled"] != ("smart" in config["variant"]):
        raise ValueError("SMART enabled/variant mismatch")
    if config["mlm"]["enabled"] != ("mlm" in config["variant"]):
        raise ValueError("MLM enabled/variant mismatch")
    for key in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", config[key]):
            raise ValueError(f"{key} must be an immutable 40-character commit")
    if config["text_variant"] not in ("raw", "segmented"):
        raise ValueError("Unknown text_variant")
    if config["backbone"] == "phobert" and config["text_variant"] != "segmented":
        raise ValueError("PhoBERT requires segmented text")
    training = config["training"]
    if training["mode"] not in ("full", "smoke"):
        raise ValueError("training.mode must be full or smoke")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "eval_batch_size"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"Invalid training.{key}")
    if training["effective_batch_size"] != training["batch_size"] * training["gradient_accumulation_steps"]:
        raise ValueError("Effective batch size mismatch")
    if training["num_workers"] != 0:
        raise ValueError("Exact resume supports num_workers=0 only")
    if training["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("Unsupported precision")
    if training["optimizer"] != "adamw" or training["scheduler"] != "linear_epoch":
        raise ValueError("Only explicitly implemented AdamW/linear_epoch are supported")
    if training["warmup_steps"] != 0 or training["max_steps"] is not None:
        raise ValueError("Baseline supports epoch budget and zero warmup")
    if training["classification_reduction"] != "mean":
        raise ValueError("Classification requires mean reduction")
    if set(training["task_weights"]) != set(config["tasks"]):
        raise ValueError("task_weights mismatch")
    if training["head_dropout"] != 0.0 or training["pooling"] != "first_token":
        raise ValueError("Baseline MTL uses first-token pooling and inactive head dropout")
    if training["encoder_hidden_dropout"] != 0.1 or training["encoder_attention_dropout"] != 0.1:
        raise ValueError("Pinned pretrained encoders use dropout=0.1; overrides require implementation audit")
    if training["st_native_head_dropout"] != (0.1 if config["variant"].startswith("st_") else None):
        raise ValueError("ST native head dropout must match the audited pretrained config")
    if training["padding"] != "max_length" or training["tokenizer_use_fast"] is not False:
        raise ValueError("Baseline requires fixed padding and use_fast=False")
    smart = config["smart"]
    if (smart["reduction"] != "batchmean" or smart["norm"] != "linf_per_token"
            or smart["dropout_policy"] != "eval_for_regularizer" or smart["steps"] < 1):
        raise ValueError("Unsupported SMART configuration")
    if smart["exclude_special_pad"] is not True or smart["independent_task_noise"] is not True:
        raise ValueError("SMART masking/independent-task policy is fixed")
    if min(smart["epsilon"], smart["gradient_eps"], smart["step_size"], smart["noise_std"]) <= 0:
        raise ValueError("SMART perturbation parameters must be positive")
    mlm = config["mlm"]
    if (mlm["mask_rate"] != 0.3 or mlm["selected_distribution"] != [0.8, 0.1, 0.1]
            or mlm["denominator"] != "selected_tokens" or mlm["reduction"] != "sum_then_normalize"
            or mlm["input_policy"] != "separate_masked_forward"):
        raise ValueError("Unsupported MLM configuration")
    if (mlm["exclude_special_pad"] is not True or mlm["head_initialization"] != "random_untied"
            or mlm["empty_target_policy"] != "zero_skip_forward"):
        raise ValueError("Unsupported MLM masking/head/empty-target policy")
    if config["selection"] != {"metric": "macro_f1", "aggregation": "mean_tasks",
                               "split": "validation", "tie_break": "earliest_epoch"}:
        raise ValueError("Unsupported selection rule")
    return config


def load_config(path, seed=None):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if seed is not None:
        config["seed"] = seed
    return validate_config(config)


def cohort_digest(config, manifest_digest, implementation):
    stable = copy.deepcopy(config)
    for key in ("seed", "experiment_id", "dataset_manifest"):
        stable.pop(key, None)
    return canonical_digest({"config": stable, "manifest": manifest_digest,
                             "implementation": implementation})
