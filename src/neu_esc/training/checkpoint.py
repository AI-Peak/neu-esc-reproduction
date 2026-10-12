"""Versioned single-file checkpoints and atomically published completion pointers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from .config import file_digest

REQUIRED = {"schema_version", "identity", "model_spec", "model", "optimizer", "scheduler",
            "scaler", "rng", "position", "selection", "tokenizer_files", "runtime", "resolved_config"}


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA topology mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def tokenizer_snapshot(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(Path(path).rglob("*"))
            if p.is_file()}


def restore_tokenizer(files, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for relative, data in files.items():
        target = path / relative
        if target.resolve().is_relative_to(path.resolve()) is False:
            raise ValueError("Invalid tokenizer checkpoint path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def save_checkpoint(path, payload):
    validate_payload(payload)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filename = "state-" + uuid.uuid4().hex + ".pt"
    final = path / filename
    temp = final.with_suffix(".tmp")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, final)
    digest = file_digest(final)
    atomic_json(path / "COMPLETE.json", {"schema_version": "1.0", "file": filename,
                                         "sha256": digest, "identity": payload["identity"]})
    # Only a completed generation is published; a crash before marker retains old state.
    # Remove superseded state files inside this exact owned checkpoint directory.
    for old in path.glob("state-*.pt"):
        if old != final:
            old.unlink()
    return digest


def validate_payload(payload):
    if REQUIRED - payload.keys() or payload.get("schema_version") != "1.0":
        raise ValueError("Incomplete or incompatible checkpoint")
    position = payload["position"]
    if {"epoch", "global_step", "order", "batch_position", "at_optimizer_boundary"} - position.keys():
        raise ValueError("Checkpoint lacks sampler/order/batch position")
    if not position["at_optimizer_boundary"]:
        raise ValueError("Resume supports completed optimizer boundaries only")
    if not payload["tokenizer_files"]:
        raise ValueError("Checkpoint lacks tokenizer files")
    if any(not isinstance(payload[k], dict) for k in
           ("identity", "model_spec", "model", "optimizer", "scheduler", "scaler", "rng", "runtime", "resolved_config")):
        raise ValueError("Incomplete checkpoint state dictionaries")
    if {"python", "numpy", "torch", "cuda"} - payload["rng"].keys():
        raise ValueError("Incomplete RNG state")
    if {"state", "param_groups"} - payload["optimizer"].keys():
        raise ValueError("Incomplete optimizer state")
    if not isinstance(position["order"], list) or any(type(v) is not int for v in position["order"]):
        raise ValueError("Invalid sampler order state")
    if any(type(position[k]) is not int or position[k] < 0 for k in ("epoch", "global_step", "batch_position")):
        raise ValueError("Invalid checkpoint progress state")


def load_checkpoint(path, expected_identity=None):
    path = Path(path)
    marker_path = path / "COMPLETE.json"
    if not marker_path.is_file():
        raise ValueError("Incomplete checkpoint: missing COMPLETE.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    filename = marker["file"]
    if Path(filename).name != filename or not filename.startswith("state-"):
        raise ValueError("Invalid checkpoint filename")
    state_path = path / filename
    if not state_path.is_file() or file_digest(state_path) != marker["sha256"]:
        raise ValueError("Incomplete/corrupt checkpoint checksum")
    # Load only checkpoints produced by this trusted run; Python/NumPy RNG uses pickle.
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_payload(payload)
    if payload["identity"] != marker["identity"]:
        raise ValueError("Checkpoint identity/marker mismatch")
    if expected_identity is not None and payload["identity"] != expected_identity:
        raise ValueError("Resume config/cohort/data/implementation hash mismatch")
    return payload, marker["sha256"]
