"""Explicitly opted-in Colab checks; never attempted by CPU/default tests."""
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from neu_esc.models import EncoderModel
from neu_esc.losses import classification_loss, mask_tokens, mlm_loss, smart_loss
from neu_esc.training.config import load_config


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("NEU_ESC_RUN_GPU_SMOKE") != "1" or not torch.cuda.is_available(),
                    reason="Requires explicit NEU_ESC_RUN_GPU_SMOKE=1 on verified Colab CUDA")
@pytest.mark.parametrize("backbone,variant", [
    (b, "st_sentiment") for b in ("mbert", "velectra", "xlm_r", "visobert", "phobert")
] + [(b, "mtl_mlm_smart") for b in ("xlm_r", "visobert", "phobert")])
def test_pinned_pretrained_gpu_forward_backward(backbone, variant):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / f"configs/experiments/{backbone}/{variant}.yaml")
    model, _ = EncoderModel.from_pretrained(config)
    model.cuda().train()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_id"],
                                             revision=config["tokenizer_revision"], use_fast=False)
    texts = ["Sinh_viên học_tập tốt .", "Dịch_vụ chưa tốt ."] if backbone == "phobert" else [
        "Sinh viên học tập tốt.", "Dịch vụ chưa tốt."]
    encoded = tokenizer(texts, max_length=24, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k: encoded[k].cuda() for k in ("input_ids", "attention_mask")}
    logits = model(**inputs)
    labels = {t: torch.tensor([1, 2], device="cuda") for t in model.tasks}
    loss, _ = classification_loss(logits, labels, config["training"]["task_weights"])
    if model.has_mlm:
        masked = mask_tokens(inputs["input_ids"], inputs["attention_mask"], tokenizer.all_special_ids,
                             tokenizer.mask_token_id, model.encoder.config.vocab_size, mask_rate=1.0)
        loss = loss + mlm_loss(model.mlm_logits(masked.input_ids, inputs["attention_mask"]), masked.labels)
        loss = loss + config["smart"]["weight"] * smart_loss(model, inputs["input_ids"],
                                                              inputs["attention_mask"], tokenizer.all_special_ids,
                                                              config["smart"])
    loss.backward()
    gradient = model.encoder.get_input_embeddings().weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    if not model.single_task:
        for head in model.heads.values():
            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    del model
    torch.cuda.empty_cache()
