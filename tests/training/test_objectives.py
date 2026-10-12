import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import (BertConfig, BertForSequenceClassification, ElectraConfig,
                          ElectraForSequenceClassification, RobertaConfig,
                          RobertaForSequenceClassification, XLMRobertaConfig,
                          XLMRobertaForSequenceClassification)
from transformers import RobertaModel, XLMRobertaModel

from neu_esc.models import EncoderModel
from neu_esc.losses import classification_loss, mask_tokens, mlm_loss, smart_loss
from neu_esc.losses.smart import kl, reference_smart_loss
from neu_esc.training.runtime import tiny_model_and_tokenizer
from neu_esc.training.engine import seed_everything
from conftest import variant_config


@pytest.mark.parametrize("variant", ["st_sentiment", "st_topic", "mtl", "mtl_smart", "mtl_mlm", "mtl_mlm_smart"])
def test_shapes_task_loss_and_shared_gradients(config, tmp_path, variant):
    config = variant_config(config, variant)
    model, tokenizer = tiny_model_and_tokenizer(config, tmp_path / variant)
    encoded = tokenizer(["word1 word2 word3", "word4 word5"], padding="max_length",
                        max_length=8, return_tensors="pt")
    logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    assert {t: list(v.shape) for t, v in logits.items()} == {
        t: [2, 4 if t == "sentiment" else 10] for t in config["tasks"]}
    labels = {t: torch.tensor([0, 1]) for t in model.tasks}
    loss, parts = classification_loss(logits, labels, config["training"]["task_weights"])
    expected = sum(F.cross_entropy(logits[t], labels[t]) for t in model.tasks)
    assert torch.equal(loss, expected)
    assert set(parts) == set(config["tasks"])
    loss.backward()
    assert model.encoder.get_input_embeddings().weight.grad.abs().sum() > 0
    if not model.single_task:
        assert model.shared[0].weight.grad.abs().sum() > 0
        for head in model.heads.values():
            assert head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("factory,config_type", [
    (BertForSequenceClassification, BertConfig), (ElectraForSequenceClassification, ElectraConfig),
    (RobertaForSequenceClassification, RobertaConfig), (XLMRobertaForSequenceClassification, XLMRobertaConfig)])
def test_native_st_families(factory, config_type):
    hf = config_type(vocab_size=32, hidden_size=16, embedding_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
                     num_labels=10, pad_token_id=0)
    model = EncoderModel(factory(hf), ["topic"], single_task=True)
    assert model(input_ids=torch.tensor([[2, 6, 7, 3]]), attention_mask=torch.ones(1, 4))["topic"].shape == (1, 10)


@pytest.mark.parametrize("factory,config_type", [(RobertaModel, RobertaConfig), (XLMRobertaModel, XLMRobertaConfig)])
def test_smart_preserves_roberta_padding_positions(factory, config_type):
    hf = config_type(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                     num_attention_heads=2, max_position_embeddings=32, pad_token_id=1)
    model = EncoderModel(factory(hf), ["sentiment", "topic"])
    model.eval()
    ids = torch.tensor([[0, 6, 1, 7, 2, 1]])
    mask = ids.ne(1).long()
    a = model(input_ids=ids, attention_mask=mask)
    b = model(inputs_embeds=model.embeddings(ids), attention_mask=mask,
              position_ids=model.embedding_position_ids(ids))
    for task in model.tasks:
        assert torch.equal(a[task], b[task])


def test_continuous_smart_gradients_and_special_mask(tiny, config):
    model, tokenizer = tiny
    inputs = tokenizer(["word1 word2 word3", "word4 word5"], padding="max_length",
                       max_length=8, return_tensors="pt")
    model.train()
    loss, diagnostic = smart_loss(model, inputs["input_ids"], inputs["attention_mask"],
                                  tokenizer.all_special_ids, config["smart"], return_diagnostics=True)
    assert model.training  # Restored after stable regularizer forwards.
    assert torch.isfinite(loss)
    for task in model.tasks:
        assert diagnostic[task]["noise_gradient_norms"][0] > 0
        noise, eligible = diagnostic[task]["noise"], diagnostic[task]["eligible_mask"]
        assert torch.equal(noise * ~eligible, torch.zeros_like(noise))
        assert noise.abs().max() <= config["smart"]["epsilon"]
    loss.backward()
    for parameter in [model.encoder.get_input_embeddings().weight, model.shared[0].weight,
                      model.heads["sentiment"].weight, model.heads["topic"].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_long_cast_allow_unused_and_real_id_call(tiny):
    model, tokenizer = tiny
    inputs = tokenizer(["word1 word2"], return_tensors="pt")
    ids = inputs["input_ids"]
    noise = torch.full_like(ids, -1e-5, dtype=torch.float32, requires_grad=True)
    clean = model(input_ids=ids, attention_mask=inputs["attention_mask"])
    cast_ids = (ids + noise).long()
    assert not cast_ids.requires_grad
    assert (cast_ids != ids).any()  # Even tiny negative noise can decrement positive integer IDs.
    logits = model(input_ids=cast_ids, attention_mask=inputs["attention_mask"])
    loss = kl(logits["sentiment"], clean["sentiment"].detach())
    assert torch.autograd.grad(loss, noise, allow_unused=True, retain_graph=True)[0] is None
    with pytest.raises(RuntimeError, match="not have been used"):
        torch.autograd.grad(loss, noise, allow_unused=False)


def test_reference_helper_matches_locked_source(tiny, config):
    root = Path(__file__).resolve().parents[2]
    path = root / "reports/method/reference/utils__loss_function.py"
    spec = importlib.util.spec_from_file_location("locked_reference_loss", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, tokenizer = tiny
    model.eval()
    inputs = tokenizer(["word1 word2", "word3 word4"], return_tensors="pt")
    states = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    def tuple_forward(ids, mask):
        output = model(input_ids=ids, attention_mask=mask)
        return output["sentiment"], output["topic"]

    source = module.SMARTLoss(tuple_forward, module.kl_loss, module.sym_kl_loss)
    torch.manual_seed(42)
    original = source(inputs["input_ids"], states["sentiment"], inputs["attention_mask"], sent=True)
    original += source(inputs["input_ids"], states["topic"], inputs["attention_mask"], sent=False)
    torch.manual_seed(42)
    ours = reference_smart_loss(model, inputs["input_ids"], inputs["attention_mask"], states, config["smart"])
    assert torch.equal(original, ours)


def test_mask_distribution_and_exclusions():
    seed_everything(42)
    ids = torch.full((1000, 104), 7, dtype=torch.long)
    ids[:, :4] = torch.tensor([0, 2, 3, 4])
    attention = torch.ones_like(ids)
    attention[:, 0] = 0
    masked = mask_tokens(ids, attention, [0, 1, 2, 3, 4], 4, 32)
    assert not masked.selected[:, :4].any()
    assert torch.equal(masked.input_ids[:, :4], ids[:, :4])
    assert masked.labels[~masked.selected].eq(-100).all()
    assert torch.equal(masked.labels[masked.selected], ids[masked.selected])
    count = int(masked.selected.sum())
    assert abs(count / 100000 - 0.3) < 0.01
    for code, expected in enumerate([0.8, 0.1, 0.1]):
        assert abs(int(masked.replacements.eq(code).sum()) / count - expected) < 0.01
    assert masked.input_ids[masked.replacements.eq(0)].eq(4).all()
    assert masked.input_ids[masked.replacements.eq(2)].eq(7).all()
    assert masked.input_ids.max() < 32
    assert masked.input_ids[masked.replacements.eq(1)].ge(5).all()


def test_mlm_normalization_no_double_division_empty_and_gradients(tiny):
    model, tokenizer = tiny
    encoded = tokenizer(["word1 word2 word3"], return_tensors="pt")
    mask = mask_tokens(encoded["input_ids"], encoded["attention_mask"], tokenizer.all_special_ids,
                       tokenizer.mask_token_id, 32, mask_rate=1.0)
    logits = model.mlm_logits(mask.input_ids, encoded["attention_mask"])
    loss = mlm_loss(logits, mask.labels)
    assert torch.allclose(loss, F.cross_entropy(logits.reshape(-1, 32), mask.labels.reshape(-1), ignore_index=-100))
    loss.backward()
    assert model.mlm_head[-1].weight.grad.abs().sum() > 0
    assert model.encoder.get_input_embeddings().weight.grad.abs().sum() > 0
    empty = torch.full_like(mask.labels, -100)
    zero = mlm_loss(logits.detach().requires_grad_(), empty)
    assert zero.item() == 0 and torch.isfinite(zero)
    zero.backward()
    special_ids = torch.tensor([[2, 3, 0]])
    special = mask_tokens(special_ids, torch.tensor([[1, 1, 0]]), [0, 1, 2, 3, 4], 4, 32)
    assert not special.selected.any()
