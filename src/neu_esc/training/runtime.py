from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from .checkpoint import atomic_json
from .config import canonical_digest, file_digest


def select_device(config):
    if config["training"]["mode"] == "full":
        if not torch.cuda.is_available():
            raise RuntimeError("Full training requires CUDA GPU on Colab; CPU is reserved for tiny smoke")
        if config["training"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 is unsupported on this GPU; choose a separately resolved fp32/fp16 config")
        return "cuda"
    if config["training"]["precision"] != "fp32":
        raise ValueError("CPU tiny smoke uses fp32")
    return "cpu"


def environment():
    packages = {}
    for name in ("torch", "transformers", "numpy", "PyYAML", "tokenizers", "sentencepiece",
                 "huggingface-hub", "safetensors", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "num_threads": torch.get_num_threads(),
            "nondeterministic_ops": "deterministic_algorithms=True raises on unsupported nondeterministic ops"}


def git_state(root):
    def command(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain", "--untracked-files=all")
    return {"code_commit": commit, "clean_status": "clean" if commit and status == "" else "dirty_or_unversioned"}


def read_manifest(config):
    path = Path(config["dataset_manifest"])
    if not path.is_file():
        raise RuntimeError("Missing module 01 dataset manifest; training integration is blocked")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = canonical_digest({k: v for k, v in manifest.items() if k != "manifest_digest"})
    if manifest.get("manifest_digest") != digest:
        raise ValueError("Dataset manifest digest mismatch")
    if config["training"]["mode"] == "full":
        if config["dataset_manifest_digest"] is None:
            raise ValueError("Full config is not locked to an audited dataset manifest")
        if config["dataset_manifest_digest"] != digest:
            raise ValueError("Full config/dataset manifest mismatch")
        if manifest["row_counts"] != {"train": 23048, "validation": 3305, "test": 6613}:
            raise ValueError("Full split sizes do not match the locked contract")
    elif config["dataset_manifest_digest"] not in (None, digest):
        raise ValueError("Smoke config/manifest mismatch")
    return manifest, digest


def data_records(config, split):
    try:
        from neu_esc.data.dataset import load_split
    except ImportError as exc:
        raise RuntimeError("Module 01 API neu_esc.data.dataset.load_split is required") from exc
    records = list(load_split(config["dataset_manifest"], split, config["text_variant"]))
    if config["training"]["mode"] == "smoke":
        records = records[:config["training"]["smoke_sample_limit"]]
    return records


def evaluator_api():
    try:
        from neu_esc.evaluation.evaluate import evaluate_records
    except ImportError as exc:
        raise RuntimeError("Module 03 evaluate_records API is required for validation selection") from exc
    return evaluate_records


def tiny_model_and_tokenizer(config, directory):
    """CPU smoke ONLY: randomly initialized transformer, never a paper benchmark."""
    from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
    from neu_esc.models import EncoderModel
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocab_path = directory / "vocab.txt"
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"word{i}" for i in range(27)]
    vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)
    hf_config = BertConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
                           hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                           pad_token_id=0, num_labels=4 if config["tasks"] == ["sentiment"] else 10)
    single = config["variant"].startswith("st_")
    encoder = BertForSequenceClassification(hf_config) if single else BertModel(hf_config)
    return EncoderModel(encoder, config["tasks"], single_task=single, mlm=config["mlm"]["enabled"]), tokenizer


def write_jsonl(path, records, *, identical_ok=False):
    path = Path(path)
    content = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if identical_ok and path.read_text(encoding="utf-8") == content:
            return file_digest(path)
        raise FileExistsError(f"Refusing to overwrite predictions: {path}")
    # Exclusive reservation; interrupted files are deliberately rejected for prediction.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return file_digest(path)


def publish_validation(result, run_dir, provenance, prediction_digest, *, config):
    from neu_esc.evaluation.evaluate import write_evaluation
    # Match evaluator CLI: provenance first, resolved config fallback, reject conflicts.
    metadata = {}
    for field in ("method_mode", "backbone", "variant"):
        declared = provenance.get(field)
        resolved = config.get(field)
        if declared is not None and resolved is not None and declared != resolved:
            raise ValueError(f"config/provenance conflict: {field}")
        metadata[field] = declared or resolved
    result = dict(result)
    result.update(run_id=provenance["run_id"], cohort_id=provenance["cohort_id"],
                  seed=provenance["seed"], predictions_sha256=prediction_digest,
                  dataset_manifest_digest=provenance["dataset_manifest_digest"], **metadata)
    output = Path(run_dir) / "evaluation" / "validation"
    if (output / "metrics.json").exists():
        previous = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if previous != result:
            raise ValueError("Existing validation artifact does not match selected checkpoint")
        return
    write_evaluation(result, output, render_png=True)
