"""Resumable optimizer-boundary engine. No test data enters this module."""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import random
import json
import time

import numpy as np
import torch

from neu_esc.losses import classification_loss, mask_tokens, mlm_loss, smart_loss
from neu_esc.losses.smart import reference_smart_loss
from .checkpoint import (atomic_json, capture_rng, restore_rng, save_checkpoint,
                         load_checkpoint, tokenizer_snapshot)


def seed_everything(seed, deterministic=True, num_threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tokenize_records(records, tokenizer, max_length):
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Training split must be nonempty with unique IDs")
    encoded = tokenizer([r["text"] for r in records], padding="max_length", truncation=True,
                        max_length=max_length, return_attention_mask=True, return_tensors="pt")
    return {key: encoded[key].cpu() for key in ("input_ids", "attention_mask")}


def prediction_records(model, tokenizer, records, max_length, batch_size, device):
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Prediction split must be nonempty and IDs unique")
    old_mode = model.training
    model.eval()
    output = []
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                rows = records[start:start + batch_size]
                encoded = tokenize_records(rows, tokenizer, max_length)
                logits = model(**{k: v.to(device) for k, v in encoded.items()})
                predicted = {t: v.argmax(-1).cpu().tolist() for t, v in logits.items()}
                for index, row in enumerate(rows):
                    for task in model.tasks:
                        output.append({"sample_id": row["sample_id"], "split": row["split"],
                                       "task": task, "y_true": int(row[task]),
                                       "y_pred": int(predicted[task][index]),
                                       "status": "ok", "error": None})
    finally:
        model.train(old_mode)
    return output


class TrainingEngine:
    def __init__(self, model, tokenizer, config, train_records, validation_records, *,
                 run_dir, identity, evaluator, device="cpu"):
        if {r["split"] for r in train_records} != {"train"}:
            raise ValueError("Trainer accepts the train split only")
        if {r["split"] for r in validation_records} != {"validation"}:
            raise ValueError("Checkpoint selection accepts validation only")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.training = config["training"]
        self.train_records = train_records
        self.validation_records = validation_records
        self.encoded = tokenize_records(train_records, tokenizer, config["max_length"])
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.identity = identity
        self.evaluator = evaluator
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.training["learning_rate"],
                                          betas=tuple(self.training["betas"]),
                                          eps=self.training["adam_epsilon"],
                                          weight_decay=self.training["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=self.training["scheduler_end_factor"],
            total_iters=self.training["epochs"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.training["precision"] == "fp16")
        self.position = {"epoch": 0, "global_step": 0, "order": [], "batch_position": 0,
                         "at_optimizer_boundary": True}
        self.selection = {"rule": config["selection"], "score": None, "best_epoch": None,
                          "best_step": None, "checkpoint_sha256": None}
        token_dir = self.run_dir / "tokenizer"
        if not token_dir.exists():
            token_dir.mkdir(parents=True)
            tokenizer.save_pretrained(token_dir)
        self.tokenizer_files = tokenizer_snapshot(token_dir)
        self.runtime = {"torch": torch.__version__, "numpy": np.__version__,
                        "device_type": self.device.type, "precision": self.training["precision"],
                        "num_threads": torch.get_num_threads(), "cuda_version": torch.version.cuda,
                        "cuda_device_count": torch.cuda.device_count() if self.device.type == "cuda" else 0,
                        "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}

    def autocast(self):
        precision = self.training["precision"]
        if precision == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)

    def payload(self):
        return {"schema_version": "1.0", "identity": self.identity,
                "model_spec": self.model.specification(), "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(), "rng": capture_rng(),
                "position": self.position, "selection": self.selection,
                "tokenizer_files": self.tokenizer_files, "runtime": self.runtime,
                "resolved_config": self.config}

    def save_latest(self):
        return save_checkpoint(self.run_dir / "checkpoints" / "latest", self.payload())

    def resume(self):
        payload, _ = load_checkpoint(self.run_dir / "checkpoints" / "latest", self.identity)
        if payload["runtime"] != self.runtime:
            raise ValueError("Exact resume requires identical device/software/deterministic runtime")
        if payload["resolved_config"] != self.config:
            raise ValueError("Resume checkpoint/resolved config mismatch")
        if payload["model_spec"] != self.model.specification():
            raise ValueError("Resume model specification mismatch")
        if payload["tokenizer_files"] != self.tokenizer_files:
            raise ValueError("Resume tokenizer bytes mismatch")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.position = payload["position"]
        self.selection = payload["selection"]
        order = self.position["order"]
        if order and (len(order) != len(self.train_records) or sorted(order) != list(range(len(order)))):
            raise ValueError("Checkpoint sampler order is not a split permutation")
        batches = math.ceil(len(self.train_records) / self.training["batch_size"])
        if not 0 <= self.position["batch_position"] <= batches:
            raise ValueError("Invalid checkpoint batch position")
        if self.selection["score"] is not None:
            _, digest = load_checkpoint(self.run_dir / "checkpoints" / "best", self.identity)
            if digest != self.selection["checkpoint_sha256"]:
                raise ValueError("Best checkpoint does not match selection in latest")
        restore_rng(payload["rng"])
        # Keep only rows belonging to the restored optimizer boundary.
        # Preserve uncommitted rows as diagnostics before truncating the log.
        step_log = self.run_dir / "logs" / "steps.jsonl"
        if step_log.exists():
            entries = [json.loads(line) for line in step_log.read_text(encoding="utf-8").splitlines()]
            discarded = [row for row in entries if row["global_step"] > self.position["global_step"]]
            if discarded:
                atomic_json(step_log.with_name(f"discarded-after-step-{self.position['global_step']}-{time.time_ns()}.json"), discarded)
                kept = [row for row in entries if row["global_step"] <= self.position["global_step"]]
                step_log.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")

    def objective(self, indices):
        inputs = {k: v[indices].to(self.device) for k, v in self.encoded.items()}
        labels = {t: torch.tensor([self.train_records[i][t] for i in indices],
                                   dtype=torch.long, device=self.device) for t in self.model.tasks}
        with self.autocast():
            logits = self.model(**inputs)
            loss, _ = classification_loss(logits, labels, self.training["task_weights"])
            if self.config["mlm"]["enabled"]:
                masked = mask_tokens(inputs["input_ids"], inputs["attention_mask"],
                                     self.tokenizer.all_special_ids, self.tokenizer.mask_token_id,
                                     self.model.encoder.config.vocab_size,
                                     self.config["mlm"]["mask_rate"])
                if masked.selected.any():
                    mlm_logits = self.model.mlm_logits(masked.input_ids, inputs["attention_mask"])
                    loss = loss + self.config["mlm"]["weight"] * mlm_loss(
                        mlm_logits, masked.labels, self.config["mlm"]["denominator"])
            if self.config["smart"]["enabled"]:
                if self.config["method_mode"] == "reference_code":
                    regularizer = reference_smart_loss(self.model, inputs["input_ids"],
                                                     inputs["attention_mask"], logits, self.config["smart"])
                else:
                    # FP32 adversarial path prevents underflow for tiny perturbations under fp16.
                    with torch.autocast(self.device.type, enabled=False):
                        regularizer = smart_loss(self.model, inputs["input_ids"], inputs["attention_mask"],
                                                 self.tokenizer.all_special_ids, self.config["smart"])
                loss = loss + self.config["smart"]["weight"] * regularizer
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training objective")
        return loss

    def validate(self):
        predictions = prediction_records(self.model, self.tokenizer, self.validation_records,
                                         self.config["max_length"], self.training["eval_batch_size"],
                                         self.device)
        metrics = self.evaluator(predictions, self.validation_records,
                                 list(self.model.tasks), allow_invalid=False)
        score = sum(metrics["tasks"][t]["macro_f1"] for t in self.model.tasks) / len(self.model.tasks)
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation selection metric")
        return predictions, metrics, score

    def run(self, stop_after_steps=None):
        if stop_after_steps is not None and stop_after_steps <= self.position["global_step"]:
            raise ValueError("stop_after_steps must exceed saved global_step")
        batch_size = self.training["batch_size"]
        accumulation = self.training["gradient_accumulation_steps"]
        total_batches = math.ceil(len(self.train_records) / batch_size)
        self.model.train()
        while self.position["epoch"] < self.training["epochs"]:
            if not self.position["order"]:
                self.position["order"] = torch.randperm(len(self.train_records)).tolist()
            while self.position["batch_position"] < total_batches:
                start = self.position["batch_position"]
                end = min(start + accumulation, total_batches)
                groups = [self.position["order"][b * batch_size:(b + 1) * batch_size]
                          for b in range(start, end)]
                total_examples = sum(len(indices) for indices in groups)
                started = time.perf_counter()
                group_loss = 0.0
                self.optimizer.zero_grad(set_to_none=True)
                for indices in groups:
                    loss = self.objective(indices) * len(indices) / total_examples
                    group_loss += float(loss.detach())
                    self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                for parameter in self.model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Nonfinite model gradient; optimizer not advanced")
                if self.training["gradient_clipping"] is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training["gradient_clipping"],
                                                   error_if_nonfinite=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.position["batch_position"] = end
                self.position["global_step"] += 1
                step = self.position["global_step"]
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                log_dir = self.run_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / "steps.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"global_step": step, "epoch": self.position["epoch"] + 1,
                                             "batch_position": end, "examples": total_examples,
                                             "training_loss": group_loss,
                                             "learning_rate": self.optimizer.param_groups[0]["lr"],
                                             "seconds": time.perf_counter() - started,
                                             "peak_vram_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None}) + "\n")
                if step % self.training["checkpoint_every_steps"] == 0:
                    self.save_latest()
                if stop_after_steps is not None and step >= stop_after_steps:
                    self.save_latest()
                    return "INTERRUPTED"
            _, metrics, score = self.validate()
            self.scheduler.step()
            epoch = self.position["epoch"] + 1
            self.position.update(epoch=epoch, order=[], batch_position=0)
            if self.selection["score"] is None or score > self.selection["score"]:
                self.selection.update(score=score, best_epoch=epoch,
                                      best_step=self.position["global_step"])
                self.selection["checkpoint_sha256"] = save_checkpoint(
                    self.run_dir / "checkpoints" / "best", self.payload())
            atomic_json(self.run_dir / "selection.json", self.selection)
            self.save_latest()
            atomic_json(self.run_dir / "logs" / f"validation-epoch-{epoch:04d}.json",
                        {"epoch": epoch, "global_step": self.position["global_step"],
                         "selection_score": score, "metrics": metrics})
        return "COMPLETED"
