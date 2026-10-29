"""Group 5 improvement runner: a copy of neu_esc_kaggle.py (the baseline, left untouched) plus the
proposal's factorial ablation on the 2-task protocol (20 epochs, AdamW eps 5e-9, LinearLR to 0.1):

  head  sep | fac   separate heads (= baseline Linear2Head) | Factorized + Interaction joint head
  opt   sum | pcgrad | patgrad   plain loss sum | PCGrad | PaTGrad (pair-aware asymmetric surgery)
  loss  ce  | cbf   cross-entropy | class-balanced focal loss on the two marginals

Setup names are f"{head}_{opt}_{loss}"; sep_sum_ce is the baseline `mtl` setup run by this file.
Fixed, untuned choices (recorded in config.json under "improvement"): rank r=16 with the V branch
zero-initialised so training starts from c = 0; L = 0.5 L_s + 0.5 L_t + 0.5 L_joint; CB beta 0.999,
focal gamma 2; surgery per encoder layer group (embeddings, layer 0..N-1). PaTGrad uses epoch 1 as
warm-up (plain sum while probing per-layer cos(g_s, g_t)), then does surgery only on the top-k
layer groups with the lowest mean cosine. Urgency U_task = mean over the batch of
A[s,t] * R_c * D_c, with A the EMA pair conflict 0.5 * (1 - cos) of per-sample gradients at the
pooled [CLS] vector, R the CB effective-number weights and D the EMA class-wise NLL (normalised to
mean 1). On conflict, the task with lower urgency is projected onto the normal plane of the other.

Direction 1, CR-SD (setup `selfdistill`): R-Drop (two dropout passes, symmetric KL) + an EMA teacher
of the student's weights (decay 0.999) that the student distils from epoch 2 on; selection, test and
inference use the EMA weights. CR-SD + D (setup `crsd_ens`) adds a second, fixed teacher: an ensemble
of out-of-fold predictions on the training set, distilled from epoch 1 on. Those targets come from
`train --fold k --num-folds N` runs (each trains on the other folds and predicts its own fold) and
`ensemble-targets`, which averages their probabilities per training row.

Everything below the baseline docstring is unchanged except build_model, the loss/step code and
the extra outputs (joint logits and joint-argmax metrics for fac, surgery diagnostics per epoch).

Baseline docstring:
Lean NEU-ESC reproduction runner for Kaggle T4, following the authors' public code.

Reference: hung20gg/multi-task-bert @ b5637cc0 (copies in reports/method/reference/).
Deliberate deviations from that code, all recorded in each run's config.json:
  - fp16 autocast + GradScaler (T4 has no bf16; the authors trained in fp32).
  - Batch 128 is split into micro-batches with gradient accumulation, padded per micro-batch.
  - SMART skips the authors' first perturbed forward: its noise gradient is always None
    because `.long()` cuts the graph, so that forward never changes the result.
  - MTL selection uses the best mean validation macro-F1 over both tasks; the authors saved
    every epoch from 4 on and picked one by hand.
No checkpoints are written: the best weights stay in CPU RAM; each run saves logits, predictions
keyed by sample ID, metrics, and the SHA-256 of this file.

Subcommands:
  fetch     download the authors' CSV splits and check SHA-256
  prefetch  download pinned model/tokenizer revisions into the HF cache
  train     one run: --backbone {phobert,visobert} --setup {st_sentiment,st_topic,mtl,mtl_smart} --seed N
  ensemble-targets  average out-of-fold train predictions of fold runs into a crsd_ens target file
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# Kaggle images ship TensorFlow; keep transformers on PyTorch only. PhoBERT's pinned revision
# has only pytorch_model.bin, and without this flag transformers fetches a safetensors copy from
# an auto-conversion PR branch, which is a different revision than the pinned one.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

AUTHOR_COMMIT = "b5637cc0a8a7b5e88e794d180c11902405548eb5"
DATA_URL = f"https://raw.githubusercontent.com/hung20gg/multi-task-bert/{AUTHOR_COMMIT}/dataset/{{name}}"
# Hashes from reports/source/download_ledger.json.
DATA_FILES = {
    "train_set.csv": "8a18a73ca21105e0b424829cc41395719be98dd7f01bf3d962eca86e7a1501ec",
    "val_set.csv": "0b2567fe5374cf9b8d22de92ce71d3f9bfa036dffac46fa6f7a5650e85c81169",
    "test_set.csv": "cdd81b2c7b96752f9f45d05e231682fcd6c380f40273dac521425ceff71ab312",
    "train_set_processed.csv": "7ca23b39e8e917de5c81745b0be2f4bb0e085fdc6173882479ecb2ed8e9ebe96",
    "val_set_processed.csv": "948578e4b92df56117fd38d8edaee8d98f11adbd38611d6faee655008adea626",
    "test_set_processed.csv": "bf850a67db6f9dd48df2adc5aa181e63e052e526361a1e9f0e6179bf582dc2e4",
}
SPLIT_SIZES = {"train": 23048, "val": 3305, "test": 6613}
NUM_LABELS = {"sentiment": 4, "topic": 10}
TASKS = ("sentiment", "topic")

# Revisions pinned on 2026-10-11 (docs/FINDINGS.md). PhoBERT reads the authors' pre-segmented CSVs,
# exactly as train.py / trainmultihead.py do for any model name containing "vinai".
BACKBONES = {
    "phobert": {"model_id": "vinai/phobert-base-v2",
                "revision": "86cd7fd4c148980922ac11a2cf5e257f2ba639e1", "processed": True},
    "visobert": {"model_id": "uitnlp/visobert",
                 "revision": "196a62afad9cbe4f52a54aabad828b13f0eec59a", "processed": False},
}

# Hyperparameters from trainer/base_trainer.py + train.py (ST) and
# Trainer/head_trainer.py + trainmultihead.py (MTL).
SETUPS = {
    "st_sentiment": {"tasks": ("sentiment",), "epochs": 15, "adam_eps": 1e-8, "lr_end": 0.2,
                     "patience": 4, "smart": False, "micro_batch": 64},
    "st_topic": {"tasks": ("topic",), "epochs": 15, "adam_eps": 1e-8, "lr_end": 0.2,
                 "patience": 4, "smart": False, "micro_batch": 64},
    "mtl": {"tasks": TASKS, "epochs": 20, "adam_eps": 5e-9, "lr_end": 0.1,
            "patience": None, "smart": False, "micro_batch": 64},
    "mtl_smart": {"tasks": TASKS, "epochs": 20, "adam_eps": 5e-9, "lr_end": 0.1,
                  "patience": None, "smart": True, "micro_batch": 32},
}
for _head in ("sep", "fac"):
    for _opt in ("sum", "pcgrad", "patgrad"):
        for _loss in ("ce", "cbf"):
            SETUPS[f"{_head}_{_opt}_{_loss}"] = {**SETUPS["mtl"], "head": _head, "opt": _opt, "loss": _loss}
# Direction 1: R-Drop consistency + EMA-teacher self-distillation on the separate-head 2-task model.
# Micro-batch 32 because both dropout passes go through the encoder as one 64-row forward.
SETUPS["selfdistill"] = {**SETUPS["sep_sum_ce"], "micro_batch": 32, "self_distill": True}
SELF_DISTILL = {"rdrop_alpha": 1.0, "kd_beta": 1.0, "ema_decay": 0.999, "kd_start_epoch": 2,
                "ens_gamma": 1.0}
# CR-SD + D: the ensemble teacher is ready from the first step, so its KD term runs from epoch 1.
SETUPS["crsd_ens"] = {**SETUPS["selfdistill"], "ens_kd": True}
FOLD_SEED = 2026  # fold assignment is fixed across training seeds and backbones
IMPROVE = {"rank": 16, "joint_weight": 0.5, "cb_beta": 0.999, "focal_gamma": 2.0,
           "patgrad_topk": 6, "patgrad_warmup_epochs": 1, "ema": 0.9}
COMMON = {"lr": 2e-5, "weight_decay": 0.01, "batch_size": 128, "max_length": 128,
          "eval_batch_size": 128, "mtl_task_weight": 0.5}
# utils/loss_function.py SMARTLoss defaults as used by head_trainer (weight 0.02).
SMART = {"weight": 0.02, "noise_var": 1e-5, "norm_eps": 1e-6, "num_steps": 1}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance():
    """Hash of this file (the only code that changes results) and the git commit when there is one.

    Runs copied in from a Kaggle Dataset have no git metadata; runner_sha256 still identifies them.
    """
    here = Path(__file__).resolve()
    info = {"runner_sha256": sha256_file(here), "git_commit": None, "git_dirty": None}
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(here.parent), *args],
                                  capture_output=True, text=True, timeout=10)
        head = git("rev-parse", "HEAD")
        if head.returncode == 0:
            info["git_commit"] = head.stdout.strip()
            # Untracked or modified runner counts as dirty: the commit alone does not identify it.
            info["git_dirty"] = bool(git("status", "--porcelain", "--", str(here)).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


# ---------------------------------------------------------------- data

def fetch_data(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in DATA_FILES.items():
        target = data_dir / name
        if not target.exists() or sha256_file(target) != expected:
            log(f"download {name}")
            urllib.request.urlretrieve(DATA_URL.format(name=name), target)
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    log(f"data OK in {data_dir}")


def load_split(data_dir, split, processed):
    import pandas as pd
    name = f"{split}_set{'_processed' if processed else ''}.csv"
    path = Path(data_dir) / name
    if sha256_file(path) != DATA_FILES[name]:
        raise RuntimeError(f"{path} does not match the pinned author CSV; run fetch")
    frame = pd.read_csv(path)
    labels = {}
    for task, column in (("sentiment", "sentiment"), ("topic", "classification")):
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
            raise ValueError(f"{name}: non-integer {column} labels")
        labels[task] = values.astype(np.int64)
        if labels[task].min() < 0 or labels[task].max() >= NUM_LABELS[task]:
            raise ValueError(f"{name}: {column} label out of range")
    texts = frame["text"].astype(str).tolist()
    if len(texts) != SPLIT_SIZES[split]:
        raise ValueError(f"{name}: expected {SPLIT_SIZES[split]} rows, got {len(texts)}")
    return texts, labels, name


def sample_ids(split, count):
    """IDs in the format of the repo's data pipeline (docs/FINDINGS.md), e.g. test:000000:cdd81b2c7b96:
    split, zero-based CSV row, then the raw CSV's SHA-256 prefix. The authors' _processed.csv files are
    row-aligned with the raw ones, so PhoBERT and ViSoBERT runs share the same IDs."""
    prefix = DATA_FILES[f"{split}_set.csv"][:12]
    name = "validation" if split == "val" else split
    return np.array([f"{name}:{row:06d}:{prefix}" for row in range(count)])


def fold_rows(num_train, fold, num_folds):
    """Training rows held out in `fold`: every num_folds-th row of one fixed permutation."""
    return np.sort(np.random.RandomState(FOLD_SEED).permutation(num_train)[fold::num_folds])


def encode_texts(tokenizer, texts, max_length):
    # Same call shape as utils/dataloader.py (special tokens, truncation at 128), but without
    # padding: each micro-batch is padded to its own longest row. Logits do not change because
    # padded positions are masked out of attention.
    encoded = tokenizer(texts, add_special_tokens=True, truncation=True, max_length=max_length)
    return [np.asarray(ids, dtype=np.int64) for ids in encoded["input_ids"]]


def collate(sequences, indices, pad_id):
    import torch
    width = max(len(sequences[i]) for i in indices)
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(indices), width), dtype=torch.long)
    for row, index in enumerate(indices):
        seq = sequences[index]
        ids[row, :len(seq)] = torch.from_numpy(seq)
        mask[row, :len(seq)] = 1
    return ids, mask


# ---------------------------------------------------------------- model

def build_model(backbone_cfg, setup_cfg, model_path):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoModelForSequenceClassification

    source = model_path or backbone_cfg["model_id"]
    kwargs = {"output_loading_info": True}
    if not model_path:
        kwargs["revision"] = backbone_cfg["revision"]
    tasks = setup_cfg["tasks"]

    if len(tasks) == 1:
        encoder, info = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=NUM_LABELS[tasks[0]], **kwargs)

        class SingleTask(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder

            def forward(self, input_ids, attention_mask):
                return (self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits,)

        return SingleTask(), info

    encoder, info = AutoModel.from_pretrained(source, **kwargs)
    hidden = encoder.config.hidden_size

    factorized = setup_cfg.get("head") == "fac"
    n_s, n_t, rank = NUM_LABELS["sentiment"], NUM_LABELS["topic"], IMPROVE["rank"]

    class Linear2Head(nn.Module):
        """architecture/bert2head/model.py: BertLinear2HEAD. Its Dropout(0.2) layers are
        declared but never called in forward, so they are left out here.

        With factorized=True, a_s and b_t are the two baseline heads and the interaction
        c_st = <U_s(z), V_t(z)> is added: score(s, t) = a_s + b_t + c_st over the 40 pairs. Outputs are
        then (log P(s|x), log P(t|x), joint scores flattened as s * 10 + t)."""

        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.fc_input = nn.Linear(hidden, hidden)
            self.activation = nn.SiLU()
            self.ln1 = nn.LayerNorm(hidden)
            self.out_sent = nn.Linear(hidden, n_s)
            self.out_clas = nn.Linear(hidden, n_t)
            if factorized:
                self.inter_u = nn.Linear(hidden, n_s * rank)
                self.inter_v = nn.Linear(hidden, n_t * rank)
                # V = 0 at start, so c = 0 and the head starts as the two independent baseline heads.
                nn.init.zeros_(self.inter_v.weight)
                nn.init.zeros_(self.inter_v.bias)

        def forward(self, input_ids, attention_mask, return_hidden=False):
            cls = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state[:, 0, :]
            embedded = self.ln1(self.activation(self.fc_input(cls)))
            a, b = self.out_sent(embedded), self.out_clas(embedded)
            if not factorized:
                outputs = (a, b)
            else:
                u = self.inter_u(embedded).float().view(-1, n_s, rank)
                v = self.inter_v(embedded).float().view(-1, n_t, rank)
                joint = a.float()[:, :, None] + b.float()[:, None, :] + torch.einsum("bsr,btr->bst", u, v)
                log_z = torch.logsumexp(joint.flatten(1), dim=1)[:, None]
                outputs = (torch.logsumexp(joint, dim=2) - log_z, torch.logsumexp(joint, dim=1) - log_z,
                           joint.flatten(1))
            return outputs + (cls,) if return_hidden else outputs

    return Linear2Head(), info


def cb_weights(labels, num_classes):
    """Class-balanced weights from the effective number (1 - beta^n) / (1 - beta), mean 1 over classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = (1 - IMPROVE["cb_beta"]) / (1 - IMPROVE["cb_beta"] ** np.maximum(counts, 1))
    return weights * num_classes / weights.sum()


def per_sample_loss(logits, targets, kind, weights):
    """Returns (loss, nll) per row. ce: plain NLL; cbf: w_y * (1 - p_y)^gamma * NLL.
    Factorized heads pass log-marginals; log_softmax leaves them unchanged."""
    import torch.nn.functional as F
    nll = -F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    if kind == "ce":
        return nll, nll
    return weights[targets] * (1 - (-nll).exp()).pow(IMPROVE["focal_gamma"]) * nll, nll


def layer_group(name):
    if name.startswith("embeddings."):
        return "emb"
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "encoder" and parts[1] == "layer":
        return f"layer{int(parts[2]):02d}"
    return None  # pooler: unused by the [CLS] head


def sym_kl_sum(perturbed, clean):
    """utils/loss_function.py sym_kl_loss(input=perturbed, target=clean, reduction='sum')."""
    import torch.nn.functional as F
    perturbed, clean = perturbed.float(), clean.float()
    return (F.kl_div(F.log_softmax(perturbed, dim=-1), F.softmax(clean.detach(), dim=-1),
                     reduction="sum")
            + F.kl_div(F.log_softmax(clean, dim=-1), F.softmax(perturbed.detach(), dim=-1),
                       reduction="sum"))


def author_smart_ids(input_ids):
    """Perturbed IDs exactly as the authors' SMARTLoss produces them for num_steps=1.

    The noise is added to integer token IDs and cast back with `.long()`, so its gradient is
    None and the ascent step reduces to `noise / (inf_norm(noise) + eps)`. Values in (-1, 0)
    truncate to id - 1, which shifts about half of the tokens to the previous vocabulary ID.
    """
    import torch
    noise = torch.randn(input_ids.shape, device=input_ids.device) * SMART["noise_var"]
    noise = noise / (noise.abs().amax(dim=-1, keepdim=True) + SMART["norm_eps"])
    return (input_ids + noise).long()


# ---------------------------------------------------------------- metrics

def task_metrics(y_true, logits):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    y_pred = logits.argmax(axis=-1)
    labels = list(range(logits.shape[-1]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        # Trainer/head_trainer.py calls f1_score(pred, true): weighted F1 then uses predicted
        # supports. Accuracy and macro-F1 are symmetric, so only this one differs.
        "weighted_f1_author_mtl": float(f1_score(y_pred, y_true, average="weighted")),
        "per_class_f1": [float(x) for x in f1_score(y_true, y_pred, average=None, labels=labels,
                                                    zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def predict(model, sequences, pad_id, device, batch_size, use_amp):
    import torch
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    outputs = None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            ids, mask = collate(sequences, chunk, pad_id)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(ids.to(device), mask.to(device))
            if outputs is None:
                outputs = [np.zeros((len(sequences), head.shape[-1]), dtype=np.float32) for head in logits]
            for out, value in zip(outputs, logits):
                out[chunk] = value.float().cpu().numpy()
    model.train(was_training)
    return outputs


# ---------------------------------------------------------------- training

def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment(device):
    import torch
    import transformers
    import sklearn
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "sklearn": sklearn.__version__, "cuda": torch.version.cuda, "device": device.type,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import sentencepiece
        env["sentencepiece"] = sentencepiece.__version__
    except ImportError:
        env["sentencepiece"] = None
    return env


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def train(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    backbone = BACKBONES[args.backbone]
    setup = dict(SETUPS[args.setup])
    if args.epochs is not None:
        setup["epochs"] = args.epochs
    micro_batch = args.micro_batch or setup["micro_batch"]
    if COMMON["batch_size"] % micro_batch:
        raise ValueError("--micro-batch must divide 128")
    tasks = setup["tasks"]
    debug = bool(args.max_steps_per_epoch or args.limit_eval or args.epochs is not None
                 or args.model_path or args.fp32)

    if (args.fold is None) != (args.num_folds is None) or (args.fold is not None
                                                             and not 0 <= args.fold < args.num_folds):
        raise ValueError("--fold and --num-folds go together, with 0 <= fold < num_folds")
    if setup.get("ens_kd") and not args.ens_targets:
        raise ValueError(f"{args.setup} needs --ens-targets (see the ensemble-targets command)")
    run_name = f"{args.backbone}-{args.setup}-s{args.seed}"
    if args.fold is not None:
        run_name += f"-f{args.fold}of{args.num_folds}"
    run_dir = Path(args.out_dir) / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        status = json.loads(metrics_path.read_text(encoding="utf-8")).get("status")
        if status == "COMPLETED":
            log(f"{run_name} already COMPLETED, skipping (use --force to rerun)")
            return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("epochs.jsonl", "metrics.json", "logits.npz", "predictions.csv", "oof.npz"):
        (run_dir / stale).unlink(missing_ok=True)
    code = code_provenance()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.fp32
    seed_everything(args.seed)

    # ---- data
    processed = backbone["processed"]
    data = {split: load_split(args.data_dir, split, processed) for split in ("train", "val", "test")}
    token_source = args.model_path or backbone["model_id"]
    token_kwargs = {} if args.model_path else {"revision": backbone["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(token_source, use_fast=False, **token_kwargs)
    pad_id = tokenizer.pad_token_id
    started = time.perf_counter()
    sequences = {split: encode_texts(tokenizer, data[split][0], COMMON["max_length"]) for split in data}
    tokenize_seconds = time.perf_counter() - started
    labels = {split: data[split][1] for split in data}
    eval_index = {}
    for split in ("val", "test"):
        keep = list(range(len(sequences[split])))
        if args.limit_eval:
            keep = keep[:args.limit_eval]
        eval_index[split] = keep
    eval_ids = {split: sample_ids(split, len(sequences[split]))[eval_index[split]] for split in eval_index}
    held_out = (fold_rows(len(sequences["train"]), args.fold, args.num_folds)
                if args.fold is not None else np.array([], dtype=np.int64))
    train_rows = np.setdiff1d(np.arange(len(sequences["train"])), held_out)
    ens_probs = None
    if setup.get("ens_kd"):
        targets_file = np.load(args.ens_targets)
        if str(targets_file["train_csv_sha256"]) != DATA_FILES[data["train"][2]]:
            raise ValueError(f"{args.ens_targets} was built from another train CSV")
        ens_probs = {task: torch.tensor(targets_file[task], dtype=torch.float32) for task in tasks}
    lengths = np.array([len(s) for s in sequences["train"]])
    log(f"{run_name}: tokenized in {tokenize_seconds:.0f}s, train tokens mean {lengths.mean():.1f}, "
        f"truncated {(lengths >= COMMON['max_length']).mean():.1%}")

    # ---- model and optimizer (AdamW defaults otherwise: betas 0.9/0.999, weight_decay 0.01)
    model, loading_info = build_model(backbone, setup, args.model_path)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=COMMON["lr"], eps=setup["adam_eps"],
                                  weight_decay=COMMON["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                  end_factor=setup["lr_end"],
                                                  total_iters=setup["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    teacher = None
    if setup.get("self_distill"):
        # EMA of the student's weights: never trained, only averaged after each optimizer step.
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_params, student_params = list(teacher.parameters()), list(model.parameters())

    config = {
        "run_name": run_name, "backbone": args.backbone, "setup": args.setup, "seed": args.seed,
        "code": code, "model_id": backbone["model_id"], "revision": None if args.model_path else backbone["revision"],
        "model_path_override": args.model_path, "text_source": "author _processed.csv" if processed
        else "author raw .csv", "data_files": {data[s][2]: DATA_FILES[data[s][2]] for s in data},
        "tasks": list(tasks), "epochs": setup["epochs"], "adam_eps": setup["adam_eps"],
        "lr_end_factor": setup["lr_end"], "early_stopping_patience": setup["patience"],
        "smart": SMART if setup["smart"] else None, **COMMON, "micro_batch": micro_batch,
        "precision": "fp16_autocast" if use_amp else "fp32",
        "selection": "val macro-F1" if len(tasks) == 1 else "mean val macro-F1 over both tasks",
        "max_steps_per_epoch": args.max_steps_per_epoch, "limit_eval": args.limit_eval,
        "benchmark_eligible": not debug, "environment": environment(device),
        "loading_info": {k: [str(x) for x in v] for k, v in loading_info.items()},
        "tokenize_seconds": tokenize_seconds,
        "improvement": {"head": setup.get("head"), "opt": setup.get("opt"), "loss": setup.get("loss"),
                        **IMPROVE} if "head" in setup else None,
        "self_distill": SELF_DISTILL if teacher is not None else None,
        "ensemble_teacher": {"targets": args.ens_targets,
                             "sources": [str(x) for x in targets_file["sources"]]} if ens_probs else None,
        "fold": {"fold": args.fold, "num_folds": args.num_folds, "fold_seed": FOLD_SEED,
                 "held_out_rows": int(len(held_out))} if args.fold is not None else None,
        "inference_weights": "EMA teacher" if teacher is not None else "student",
    }
    write_json(run_dir / "config.json", config)

    # ---- improvement state (factorial setups only)
    new_setup = "head" in setup
    opt_kind = setup.get("opt", "sum")
    surgery = new_setup and opt_kind != "sum"
    n_s, n_t = NUM_LABELS["sentiment"], NUM_LABELS["topic"]
    class_weights = {task: torch.tensor(cb_weights(labels["train"][task], NUM_LABELS[task]),
                                        dtype=torch.float32, device=device) for task in tasks}
    if surgery:
        encoder_group = {id(p): layer_group(n) for n, p in model.encoder.named_parameters()}
        params = [p for p in model.parameters() if p.requires_grad]
        groups = {}
        for i, p in enumerate(params):
            if encoder_group.get(id(p)):
                groups.setdefault(encoder_group[id(p)], []).append(i)
        group_names = sorted(groups)
        encoder_index = {i for idx in groups.values() for i in idx}
        buf_s = {i: torch.zeros_like(params[i]) for i in encoder_index}
        buf_t = {i: torch.zeros_like(params[i]) for i in encoder_index}
        active = set(group_names) if opt_kind == "pcgrad" else set()
        pair_conflict = torch.full((n_s, n_t), 0.5, device=device)
        difficulty = {task: torch.ones(NUM_LABELS[task], device=device) for task in tasks}
    ema = IMPROVE["ema"]

    def ema_by_class(state, index, values, size):
        sums = torch.zeros(size, device=device).index_add_(0, index, values)
        counts = torch.zeros(size, device=device).index_add_(0, index, torch.ones_like(values))
        seen = counts > 0
        state[seen] = ema * state[seen] + (1 - ema) * sums[seen] / counts[seen]

    if ens_probs is not None:
        ens_log = {task: probs.clamp_min(1e-8).log().to(device) for task, probs in ens_probs.items()}

    def self_distill_loss(ids, mask, targets, batch_rows, epoch, chunk):
        """0.5 * sum over tasks of [CE averaged over two dropout passes + alpha * sym-KL(pass1, pass2)
        + beta * mean KL(teacher || pass)], the KD term from kd_start_epoch on. Row sums divided by
        the 128-row batch, so micro-batches add up to the batch mean."""
        both = model(torch.cat([ids, ids]), torch.cat([mask, mask]))
        teacher_out = None
        if epoch >= SELF_DISTILL["kd_start_epoch"]:
            with torch.no_grad():
                teacher_out = teacher(ids, mask)
        total = 0.0
        for k in range(len(tasks)):
            first, second = both[k].float().chunk(2)
            log1, log2 = F.log_softmax(first, dim=-1), F.log_softmax(second, dim=-1)
            ce = 0.5 * (F.nll_loss(log1, targets[k], reduction="sum")
                        + F.nll_loss(log2, targets[k], reduction="sum"))
            cons = 0.5 * (F.kl_div(log1, log2, log_target=True, reduction="sum")
                          + F.kl_div(log2, log1, log_target=True, reduction="sum"))
            task_loss = ce + SELF_DISTILL["rdrop_alpha"] * cons
            if teacher_out is not None:
                log_t = F.log_softmax(teacher_out[k].float(), dim=-1)
                task_loss = task_loss + SELF_DISTILL["kd_beta"] * 0.5 * (
                    F.kl_div(log1, log_t, log_target=True, reduction="sum")
                    + F.kl_div(log2, log_t, log_target=True, reduction="sum"))
            if ens_probs is not None:
                log_e = ens_log[tasks[k]][chunk]
                task_loss = task_loss + SELF_DISTILL["ens_gamma"] * 0.5 * (
                    F.kl_div(log1, log_e, log_target=True, reduction="sum")
                    + F.kl_div(log2, log_e, log_target=True, reduction="sum"))
            total = total + COMMON["mtl_task_weight"] * task_loss / batch_rows
        return total

    # ---- loop
    task_columns = {task: torch.from_numpy(labels["train"][task]) for task in tasks}
    n_train = len(train_rows)
    generator = torch.Generator().manual_seed(args.seed)
    best = {"score": -math.inf, "epoch": None, "state": None, "val_logits": None}
    best_student = {"score": -math.inf, "epoch": None, "state": None}
    selected = teacher if teacher is not None else model
    val_history = {task: [] for task in tasks}
    test_history = {task: [] for task in tasks}
    epochs_without_gain = 0
    step_times = []
    status = "COMPLETED"

    def evaluate(split, net=None):
        keep = eval_index[split]
        logits = predict(net or model, [sequences[split][i] for i in keep], pad_id, device,
                         COMMON["eval_batch_size"], use_amp)
        result = {task: task_metrics(labels[split][task][keep], logits[k])
                  for k, task in enumerate(tasks)}
        return logits, result

    model.train()
    for epoch in range(1, setup["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        order = train_rows[torch.randperm(n_train, generator=generator).numpy()].tolist()
        batches = [order[i:i + COMMON["batch_size"]] for i in range(0, n_train, COMMON["batch_size"])]
        if args.max_steps_per_epoch:
            batches = batches[:args.max_steps_per_epoch]
        loss_sum = 0.0
        epoch_cos = {g: [] for g in group_names} if surgery else {}
        epoch_global_cos, projected, steps_protect_sentiment = [], 0, 0
        for step, batch in enumerate(batches, start=1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            if surgery:
                for i in encoder_index:
                    buf_s[i].zero_()
                    buf_t[i].zero_()
                urgency = torch.zeros(2, device=device)
            for m in range(0, len(batch), micro_batch):
                chunk = batch[m:m + micro_batch]
                share = len(chunk) / len(batch)
                ids, mask = collate(sequences["train"], chunk, pad_id)
                ids, mask = ids.to(device), mask.to(device)
                targets = [task_columns[task][chunk].to(device) for task in tasks]
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    if teacher is None:
                        outputs = model(ids, mask, return_hidden=True) if opt_kind == "patgrad" else model(ids, mask)
                    if teacher is not None:
                        loss = self_distill_loss(ids, mask, targets, len(batch), epoch, chunk)
                    elif new_setup:
                        # Per-row losses summed over the micro-batch and divided by the 128-row batch,
                        # the same scaling as CE mean * share; ce on separate heads equals the mtl loss.
                        task_losses, nlls = [], []
                        for k, task in enumerate(tasks):
                            per_row, nll = per_sample_loss(outputs[k], targets[k], setup["loss"], class_weights[task])
                            task_losses.append(COMMON["mtl_task_weight"] * per_row.sum() / len(batch))
                            nlls.append(nll.detach())
                        joint_loss = None
                        if setup["head"] == "fac":
                            pair = targets[0] * n_t + targets[1]
                            joint_loss = IMPROVE["joint_weight"] * F.cross_entropy(
                                outputs[2], pair, reduction="sum") / len(batch)
                        loss = task_losses[0] + task_losses[1] + (joint_loss if joint_loss is not None else 0.0)
                    elif len(tasks) == 1:
                        # base_trainer.py: CrossEntropyLoss (mean) over the 128-row batch.
                        loss = F.cross_entropy(outputs[0].float(), targets[0]) * share
                    else:
                        # head_trainer.py: loss_i = CE_i + 0.02 * SMART_i, then 0.5/0.5.
                        # CE is a batch mean, so it is scaled by this micro-batch's share;
                        # sym-KL uses reduction='sum', so micro-batch sums add up unscaled.
                        loss = 0.0
                        for k, task in enumerate(tasks):
                            task_loss = F.cross_entropy(outputs[k].float(), targets[k]) * share
                            if setup["smart"]:
                                # The authors draw independent noise for each task's call.
                                perturbed = model(author_smart_ids(ids), mask)[k]
                                task_loss = task_loss + SMART["weight"] * sym_kl_sum(perturbed, outputs[k])
                            loss = loss + COMMON["mtl_task_weight"] * task_loss
                if not surgery:
                    scaler.scale(loss).backward()
                else:
                    # Separate gradients per task on the encoder; heads and g_joint use raw sums.
                    extra = [outputs[-1]] if opt_kind == "patgrad" else []
                    grads_s = torch.autograd.grad(scaler.scale(task_losses[0]), params + extra,
                                                  retain_graph=True, allow_unused=True)
                    grads_t = torch.autograd.grad(scaler.scale(task_losses[1]), params + extra,
                                                  retain_graph=joint_loss is not None, allow_unused=True)
                    grads_j = (torch.autograd.grad(scaler.scale(joint_loss), params, allow_unused=True)
                               if joint_loss is not None else [None] * len(params))
                    for i, p in enumerate(params):
                        if i in encoder_index:
                            if grads_s[i] is not None:
                                buf_s[i] += grads_s[i]
                            if grads_t[i] is not None:
                                buf_t[i] += grads_t[i]
                            direct = (grads_j[i],)
                        else:
                            direct = (grads_s[i], grads_t[i], grads_j[i])
                        for g in direct:
                            if g is not None:
                                p.grad = g if p.grad is None else p.grad.add_(g)
                    if opt_kind == "patgrad":
                        with torch.no_grad():
                            # Pair conflict from per-row gradients at the shared [CLS] vector.
                            cos = F.cosine_similarity(grads_s[-1].float(), grads_t[-1].float(), dim=1, eps=1e-12)
                            pair = targets[0] * n_t + targets[1]
                            ok = torch.isfinite(cos)
                            ema_by_class(pair_conflict.view(-1), pair[ok], 0.5 * (1 - cos[ok]), n_s * n_t)
                            for k, task in enumerate(tasks):
                                ema_by_class(difficulty[task], targets[k], nlls[k], NUM_LABELS[task])
                            a_rows = pair_conflict.view(-1)[pair]
                            for k, task in enumerate(tasks):
                                d = difficulty[task] / difficulty[task].mean()
                                urgency[k] += (a_rows * class_weights[task][targets[k]] * d[targets[k]]).sum()
                batch_loss += float(loss.detach())
            if surgery:
                with torch.no_grad():
                    stats = torch.stack([torch.stack([
                        sum((buf_s[i] * buf_t[i]).sum() for i in groups[g]),
                        sum(buf_s[i].pow(2).sum() for i in groups[g]),
                        sum(buf_t[i].pow(2).sum() for i in groups[g])]) for g in group_names]).double().tolist()
                    protect_sentiment = opt_kind == "patgrad" and float(urgency[0]) >= float(urgency[1])
                    steps_protect_sentiment += protect_sentiment
                    totals = np.sum(stats, axis=0)
                    if np.all(np.isfinite(totals)) and totals[1] > 0 and totals[2] > 0:
                        epoch_global_cos.append(totals[0] / math.sqrt(totals[1] * totals[2]))
                    for g, (dot, ns, nt) in zip(group_names, stats):
                        alpha = gamma = 1.0
                        valid = math.isfinite(dot) and math.isfinite(ns) and math.isfinite(nt) and ns > 0 and nt > 0
                        if valid:
                            epoch_cos[g].append(dot / math.sqrt(ns * nt))
                        if valid and dot < 0 and g in active:
                            projected += 1
                            if opt_kind == "pcgrad":
                                alpha, gamma = 1 - dot / ns, 1 - dot / nt
                            elif protect_sentiment:
                                alpha = 1 - dot / ns  # g_t loses its component along g_s
                            else:
                                gamma = 1 - dot / nt  # g_s loses its component along g_t
                        for i in groups[g]:
                            combined = buf_s[i].mul_(alpha).add_(buf_t[i], alpha=gamma)
                            params[i].grad = combined if params[i].grad is None else params[i].grad.add_(combined)
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                with torch.no_grad():
                    torch._foreach_lerp_(teacher_params, student_params, 1 - SELF_DISTILL["ema_decay"])
            loss_sum += batch_loss
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - step_start)
            if step % 50 == 0:
                log(f"{run_name} epoch {epoch} step {step}/{len(batches)} loss {batch_loss:.4f}")
        train_seconds = time.perf_counter() - epoch_start
        scheduler.step()  # base_trainer / head_trainer step LinearLR once per epoch

        eval_start = time.perf_counter()
        val_logits, val_metrics = evaluate("val")
        student_val = None
        if teacher is not None:
            # The student is scored too (an EMA-free ablation); selection and test use the teacher.
            student_val = {t: {k: v for k, v in val_metrics[t].items() if k != "confusion_matrix"}
                           for t in tasks}
            student_score = float(np.mean([val_metrics[t]["macro_f1"] for t in tasks]))
            if student_score > best_student["score"]:
                best_student.update(score=student_score, epoch=epoch, state={
                    k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
            val_logits, val_metrics = evaluate("val", teacher)
        val_seconds = time.perf_counter() - eval_start
        for k, task in enumerate(tasks):
            val_history[task].append(val_logits[k])
        score = float(np.mean([val_metrics[t]["macro_f1"] for t in tasks]))
        record = {"epoch": epoch, "train_loss": loss_sum / len(batches), "steps": len(batches),
                  "lr_next": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
                  "val_seconds": val_seconds, "selection_score": score,
                  "val": {t: {k: v for k, v in val_metrics[t].items() if k != "confusion_matrix"}
                          for t in tasks},
                  "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 2**30
                  if device.type == "cuda" else None}
        if student_val is not None:
            record["student_val"] = student_val
        if surgery:
            all_cos = [c for g in group_names for c in epoch_cos[g]]
            group_mean = {g: float(np.mean(epoch_cos[g])) if epoch_cos[g] else None for g in group_names}
            record["surgery"] = {
                "active_groups": sorted(active),
                "group_cos_mean": group_mean,
                "group_conflict_rate": float(np.mean([c < 0 for c in all_cos])) if all_cos else None,
                "global_cos_mean": float(np.mean(epoch_global_cos)) if epoch_global_cos else None,
                "global_conflict_rate": float(np.mean([c < 0 for c in epoch_global_cos])) if epoch_global_cos else None,
                "projections_per_step": projected / len(batches),
                "protect_sentiment_rate": steps_protect_sentiment / len(batches) if opt_kind == "patgrad" else None,
                "pair_conflict": pair_conflict.tolist() if opt_kind == "patgrad" else None,
            }
            if opt_kind == "patgrad" and epoch == IMPROVE["patgrad_warmup_epochs"]:
                # Layer probing: keep the top-k groups with the lowest mean cos(g_s, g_t) in warm-up.
                ranked = sorted((c, g) for g, c in group_mean.items() if c is not None)
                active = {g for _, g in ranked[:IMPROVE["patgrad_topk"]]}
                record["surgery"]["selected_groups"] = sorted(active)
                log(f"{run_name}: PaTGrad surgery on {sorted(active)}")
        if args.test_every_epoch:
            test_logits, test_metrics = evaluate("test", selected)
            for k, task in enumerate(tasks):
                test_history[task].append(test_logits[k])
            record["test"] = {t: {k: v for k, v in test_metrics[t].items() if k != "confusion_matrix"}
                              for t in tasks}
        with open(run_dir / "epochs.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        summary = " ".join(f"{t}: acc {val_metrics[t]['accuracy']:.4f} mF1 {val_metrics[t]['macro_f1']:.4f}"
                           for t in tasks)
        log(f"{run_name} epoch {epoch}/{setup['epochs']} train {train_seconds:.0f}s "
            f"val {val_seconds:.0f}s | {summary}")

        # base_trainer.py keeps a checkpoint only when the metric strictly improves.
        if score > best["score"]:
            best.update(score=score, epoch=epoch, val_logits=val_logits,
                        state={k: v.detach().to("cpu", copy=True) for k, v in selected.state_dict().items()})
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if setup["patience"] and epochs_without_gain >= setup["patience"]:
                log(f"{run_name}: early stop after epoch {epoch} (patience {setup['patience']})")
                break

    # ---- test with the selected weights
    student_result = None
    if teacher is not None:
        model.load_state_dict(best_student["state"])
        student_logits, student_test = evaluate("test")
        student_result = {"best_epoch": best_student["epoch"], "selection_score": best_student["score"],
                          "test": student_test}
        np.savez_compressed(run_dir / "student_logits.npz", **{
            f"test_{task}": student_logits[k] for k, task in enumerate(tasks)})
    model.load_state_dict(best["state"])
    test_start = time.perf_counter()
    test_logits, test_metrics = evaluate("test")
    test_seconds = time.perf_counter() - test_start
    val_metrics = {t: task_metrics(labels["val"][t][eval_index["val"]], best["val_logits"][k])
                   for k, t in enumerate(tasks)}

    arrays = {"val_ids": eval_ids["val"], "test_ids": eval_ids["test"]}
    for k, task in enumerate(tasks):
        arrays[f"val_{task}"] = best["val_logits"][k]
        arrays[f"test_{task}"] = test_logits[k]
        arrays[f"val_labels_{task}"] = labels["val"][task][eval_index["val"]]
        arrays[f"test_labels_{task}"] = labels["test"][task][eval_index["test"]]
        arrays[f"val_by_epoch_{task}"] = np.stack(val_history[task])
        if test_history[task]:
            arrays[f"test_by_epoch_{task}"] = np.stack(test_history[task])
    joint_decode = None
    if setup.get("head") == "fac":
        arrays["val_joint"], arrays["test_joint"] = best["val_logits"][2], test_logits[2]
        # Joint argmax (inference option 2): the max over the other task gives each task's label
        # of the single most likely (s, t) pair.
        joint_decode = {}
        for split, joint in (("val", best["val_logits"][2]), ("test", test_logits[2])):
            cube = joint.reshape(-1, n_s, n_t)
            joint_decode[split] = {
                "sentiment": task_metrics(labels[split]["sentiment"][eval_index[split]], cube.max(axis=2)),
                "topic": task_metrics(labels[split]["topic"][eval_index[split]], cube.max(axis=1))}
    np.savez_compressed(run_dir / "logits.npz", **arrays)
    if args.fold is not None:
        oof_logits = predict(model, [sequences["train"][i] for i in held_out], pad_id, device,
                             COMMON["eval_batch_size"], use_amp)
        np.savez_compressed(run_dir / "oof.npz", rows=held_out,
                            train_csv_sha256=DATA_FILES[data["train"][2]],
                            **{f"oof_{task}": oof_logits[k] for k, task in enumerate(tasks)})

    # The selected epoch's predictions keyed by sample ID; logits.npz holds the same rows in the same order.
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "split"] + [f"{t}_{kind}" for t in tasks for kind in ("true", "pred")])
        for split, logits in (("val", best["val_logits"]), ("test", test_logits)):
            for row, sample_id in enumerate(eval_ids[split]):
                cells = []
                for k, task in enumerate(tasks):
                    cells += [int(labels[split][task][eval_index[split][row]]), int(logits[k][row].argmax())]
                writer.writerow([sample_id, split] + cells)

    times = np.array(step_times[3:] or step_times)
    steps_full_epoch = math.ceil(n_train / COMMON["batch_size"])
    write_json(metrics_path, {
        "status": status, "run_name": run_name, "benchmark_eligible": not debug,
        "runner_sha256": code["runner_sha256"], "best_epoch": best["epoch"], "epochs_run": len(val_history[tasks[0]]),
        "selection_score": best["score"], "val": val_metrics, "test": test_metrics,
        "joint_decode": joint_decode, "student": student_result,
        "oof_accuracy": {task: float((oof_logits[k].argmax(-1) == labels["train"][task][held_out]).mean())
                         for k, task in enumerate(tasks)} if args.fold is not None else None,
        "timing": {"step_seconds_mean": float(times.mean()), "step_seconds_median": float(np.median(times)),
                   "steps_timed": int(len(times)),
                   "projected_train_epoch_seconds": float(times.mean() * steps_full_epoch),
                   "test_seconds": test_seconds,
                   "wall_seconds": time.perf_counter() - started},
    })
    log(f"{run_name}: best epoch {best['epoch']} | " + " ".join(
        f"test {t}: acc {test_metrics[t]['accuracy']:.4f} mF1 {test_metrics[t]['macro_f1']:.4f}"
        for t in tasks))
    return 0


def ensemble_targets(args):
    """Average softmax probabilities of out-of-fold predictions per training row (any mix of
    backbones and seeds). PhoBERT and ViSoBERT read row-aligned CSVs, so rows match across them."""
    sums, counts, sources, sha = {}, None, [], None
    for run_dir in args.runs:
        path = Path(run_dir) / "oof.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path}: not a finished --fold run")
        oof = np.load(path)
        if sha is None:
            sha = str(oof["train_csv_sha256"])
            counts = np.zeros(SPLIT_SIZES["train"], dtype=np.int64)
        elif str(oof["train_csv_sha256"]) != sha:
            raise ValueError(f"{path}: different train CSV")
        rows = oof["rows"]
        for task in TASKS:
            logits = oof[f"oof_{task}"].astype(np.float64)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            sums.setdefault(task, np.zeros((SPLIT_SIZES["train"], NUM_LABELS[task])))[rows] += probs
        counts[rows] += 1
        sources.append(str(Path(run_dir).name))
    if counts is None or counts.min() == 0:
        raise ValueError(f"{int((counts == 0).sum()) if counts is not None else 'all'} training rows have no "
                         "out-of-fold prediction; pass every fold of at least one fold set")
    out = {task: (sums[task] / counts[:, None]).astype(np.float32) for task in TASKS}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, train_csv_sha256=sha, sources=np.array(sources), counts=counts, **out)
    log(f"wrote {args.out}: {len(sources)} runs, {counts.min()}-{counts.max()} predictions per row")
    return 0


def prefetch(args):
    from transformers import AutoModel, AutoTokenizer
    for name in args.backbones:
        backbone = BACKBONES[name]
        AutoTokenizer.from_pretrained(backbone["model_id"], revision=backbone["revision"], use_fast=False)
        AutoModel.from_pretrained(backbone["model_id"], revision=backbone["revision"])
        log(f"cached {backbone['model_id']}@{backbone['revision'][:8]}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download and verify the authors' CSV splits")
    p_fetch.add_argument("--data-dir", required=True)

    p_prefetch = sub.add_parser("prefetch", help="cache pinned models/tokenizers")
    p_prefetch.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))

    p_ens = sub.add_parser("ensemble-targets", help="average out-of-fold train predictions for crsd_ens")
    p_ens.add_argument("runs", nargs="+", help="run directories of finished --fold runs")
    p_ens.add_argument("--out", required=True, help="output .npz")

    p_train = sub.add_parser("train", help="run one backbone/setup/seed")
    p_train.add_argument("--backbone", required=True, choices=list(BACKBONES))
    p_train.add_argument("--setup", required=True, choices=list(SETUPS))
    p_train.add_argument("--seed", required=True, type=int)
    p_train.add_argument("--data-dir", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--micro-batch", type=int, help="default 64, or 32 for mtl_smart")
    p_train.add_argument("--test-every-epoch", action="store_true",
                         help="also save test logits each epoch (post-hoc analysis only; never select on test)")
    p_train.add_argument("--force", action="store_true", help="rerun even if metrics.json is COMPLETED")
    p_train.add_argument("--fold", type=int, help="hold out this fold of the training set and save "
                         "its out-of-fold predictions (oof.npz)")
    p_train.add_argument("--num-folds", type=int)
    p_train.add_argument("--ens-targets", help="ensemble-targets output, required by crsd_ens")
    debug = p_train.add_argument_group("smoke/debug (marks the run benchmark_eligible=false)")
    debug.add_argument("--epochs", type=int)
    debug.add_argument("--max-steps-per-epoch", type=int)
    debug.add_argument("--limit-eval", type=int)
    debug.add_argument("--model-path", help="local model+tokenizer dir instead of the pinned HF revision")
    debug.add_argument("--fp32", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_data(args.data_dir)
        return 0
    if args.command == "prefetch":
        return prefetch(args)
    if args.command == "ensemble-targets":
        return ensemble_targets(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
