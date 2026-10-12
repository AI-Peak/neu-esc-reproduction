"""python -m neu_esc.training.train --config ... --seed 42 --run-dir ... [--resume]."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import yaml

from neu_esc.models import EncoderModel
from .checkpoint import atomic_json, load_checkpoint
from .config import load_config, canonical_digest, cohort_digest, implementation_digest
from .engine import TrainingEngine, seed_everything
from .runtime import (select_device, environment, git_state, read_manifest, data_records,
                      evaluator_api, tiny_model_and_tokenizer, write_jsonl, publish_validation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int, help="Absolute optimizer boundary; saves INTERRUPTED run")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = select_device(config)  # Fail BEFORE data/model downloads on CPU full mode.
    code = git_state(Path(__file__).resolve().parents[3])
    if config["training"]["mode"] == "full" and code["clean_status"] != "clean":
        raise RuntimeError("Full runs require a clean, committed snapshot from module 04")
    _, manifest_digest = read_manifest(config)
    seed_everything(config["seed"], config["training"]["deterministic"], config["training"]["num_threads"])
    current_environment = environment()
    implementation = implementation_digest()
    digest = canonical_digest(config)
    cohort = cohort_digest(config, manifest_digest, implementation)
    identity = {"config_hash": digest, "cohort_id": cohort, "dataset_manifest_digest": manifest_digest,
                "implementation_hash": implementation, "environment_hash": canonical_digest(current_environment),
                "seed": config["seed"]}
    evaluator = evaluator_api()
    # Only these two split calls occur in train; test is reserved for predict.
    train_records = data_records(config, "train")
    validation_records = data_records(config, "validation")
    if args.resume:
        if not args.run_dir.is_dir():
            raise ValueError("Resume run directory does not exist")
        previous_config = load_config(args.run_dir / "config.resolved.yaml")
        if previous_config != config:
            raise ValueError("Resume resolved config mismatch")
        provenance = json.loads((args.run_dir / "provenance.json").read_text(encoding="utf-8"))
        if config["training"]["mode"] == "full" and provenance["code_commit"] != code["code_commit"]:
            raise ValueError("Full resume must use the same committed snapshot")
        if provenance["run_status"] == "COMPLETED":
            raise ValueError("A completed run cannot be resumed")
        payload, _ = load_checkpoint(args.run_dir / "checkpoints" / "latest", identity)
        model = EncoderModel.from_specification(payload["model_spec"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.run_dir / "tokenizer", local_files_only=True, use_fast=False)
    else:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        for relative in ("logs", "checkpoints", "predictions", "evaluation"):
            (args.run_dir / relative).mkdir()
        (args.run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        provenance = {"schema_version": "1.0", "run_id": args.run_dir.name,
                      "cohort_id": cohort, "seed": config["seed"], **code,
                      "tasks": config["tasks"], "method_mode": config["method_mode"],
                      "backbone": config["backbone"], "variant": config["variant"],
                      "config_hash": digest, "dataset_manifest_digest": manifest_digest,
                      "implementation_hash": implementation,
                      "model_id": config["model_id"], "model_revision": config["model_revision"],
                      "tokenizer_id": config["tokenizer_id"], "tokenizer_revision": config["tokenizer_revision"],
                      "environment": current_environment,
                      "start_time": datetime.now(timezone.utc).isoformat(), "end_time": None,
                      "run_status": "INITIALIZING", "training_mode": config["training"]["mode"],
                      "model_initialization": "pretrained" if device == "cuda" else "tiny_random_smoke",
                      "benchmark_eligible": device == "cuda" and config["method_mode"] == "paper_method"}
        atomic_json(args.run_dir / "provenance.json", provenance)
        atomic_json(args.run_dir / "environment.json", current_environment)
        if device == "cpu":
            model, tokenizer = tiny_model_and_tokenizer(config, args.run_dir / "smoke_assets")
        else:
            model, info = EncoderModel.from_pretrained(config)
            atomic_json(args.run_dir / "logs" / "pretrained_loading.json", info)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_id"],
                                                     revision=config["tokenizer_revision"], use_fast=False,
                                                     trust_remote_code=False)
    engine = TrainingEngine(model, tokenizer, config, train_records, validation_records,
                            run_dir=args.run_dir, identity=identity, evaluator=evaluator, device=device)
    if args.resume:
        engine.resume()
    else:
        engine.save_latest()
    provenance.update(run_status="RUNNING", last_resume_time=datetime.now(timezone.utc).isoformat() if args.resume else None)
    atomic_json(args.run_dir / "provenance.json", provenance)
    try:
        status = engine.run(args.stop_after_steps)
        if status == "COMPLETED":
            selected, _ = load_checkpoint(args.run_dir / "checkpoints" / "best", identity)
            engine.model.load_state_dict(selected["model"], strict=True)
            predictions, metrics, _ = engine.validate()
            prediction_digest = write_jsonl(args.run_dir / "predictions" / "validation.jsonl", predictions,
                                            identical_ok=True)
            publish_validation(metrics, args.run_dir, provenance, prediction_digest, config=config)
        provenance.update(run_status=status, end_time=datetime.now(timezone.utc).isoformat(),
                          global_step=engine.position["global_step"], epochs_completed=engine.position["epoch"])
    except BaseException as exc:
        # Never save a partially accumulated batch. The last COMPLETE checkpoint stays usable.
        provenance.update(run_status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                          end_time=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__)
        atomic_json(args.run_dir / "provenance.json", provenance)
        raise
    atomic_json(args.run_dir / "provenance.json", provenance)
    print(json.dumps({"run_id": args.run_dir.name, "run_status": provenance["run_status"],
                      "benchmark_eligible": provenance["benchmark_eligible"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
