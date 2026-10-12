"""Load selected weights and tokenizer in a new process; preserve all sample IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
from transformers import AutoTokenizer

from neu_esc.models import EncoderModel
from .checkpoint import load_checkpoint, restore_tokenizer
from .config import load_config, canonical_digest
from .engine import prediction_records
from .runtime import read_manifest, data_records, evaluator_api, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)
    config = load_config(args.run_dir / "config.resolved.yaml")
    provenance = json.loads((args.run_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance["run_status"] != "COMPLETED":
        raise ValueError("Prediction requires a completed run and validated best checkpoint")
    selection = json.loads((args.run_dir / "selection.json").read_text(encoding="utf-8"))
    payload, digest = load_checkpoint(args.run_dir / "checkpoints" / "best")
    if digest != selection["checkpoint_sha256"]:
        raise ValueError("Best checkpoint hash differs from selection.json")
    if canonical_digest(config) != payload["identity"]["config_hash"]:
        raise ValueError("Selected weights/resolved config mismatch")
    if payload["resolved_config"] != config:
        raise ValueError("Selected checkpoint does not contain the run's complete config")
    _, manifest_digest = read_manifest(config)
    if manifest_digest != payload["identity"]["dataset_manifest_digest"]:
        raise ValueError("Prediction dataset differs from training")
    records = data_records(config, args.split)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if config["training"]["mode"] == "full" and device != "cuda":
        raise RuntimeError("Full prediction requires GPU; only tiny smoke runs use CPU")
    model = EncoderModel.from_specification(payload["model_spec"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    with tempfile.TemporaryDirectory(prefix="neu-esc-tokenizer-") as directory:
        token_dir = Path(directory) / "tokenizer"
        restore_tokenizer(payload["tokenizer_files"], token_dir)
        tokenizer = AutoTokenizer.from_pretrained(token_dir, use_fast=False, local_files_only=True)
        predictions = prediction_records(model, tokenizer, records, config["max_length"],
                                         config["training"]["eval_batch_size"], device)
    # The authoritative evaluator checks complete IDs and labels before any output publication.
    evaluator_api()(predictions, records, config["tasks"], allow_invalid=False)
    output = args.run_dir / "predictions" / f"{args.split}.jsonl"
    digest = write_jsonl(output, predictions)
    print(json.dumps({"predictions": str(output), "sha256": digest, "rows": len(predictions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
