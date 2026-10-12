"""Executable CPU evidence for cast disconnection and continuous SMART gradients."""
import importlib.util
import json
from pathlib import Path

import torch

from neu_esc.losses.smart import kl, smart_loss
from neu_esc.training.config import load_config, file_digest
from neu_esc.training.engine import seed_everything
from neu_esc.training.runtime import tiny_model_and_tokenizer


def main():
    seed_everything(42)
    config = load_config("tests/training/fixtures/tiny.yaml")
    model, tokenizer = tiny_model_and_tokenizer(config, "runs/smoke/training/gradient-assets")
    model.eval()
    encoded = tokenizer(["word1 word2 word3", "word4 word5"], padding="max_length",
                        max_length=8, return_tensors="pt")
    ids, attention = encoded["input_ids"], encoded["attention_mask"]
    noise = torch.full_like(ids, -1e-5, dtype=torch.float32, requires_grad=True)
    clean = model(input_ids=ids, attention_mask=attention)
    cast = (ids + noise).long()
    perturbed = model(input_ids=cast, attention_mask=attention)
    loss = kl(perturbed["sentiment"], clean["sentiment"].detach())
    unused = torch.autograd.grad(loss, noise, allow_unused=True, retain_graph=True)[0]
    try:
        torch.autograd.grad(loss, noise, allow_unused=False)
        strict_error = None
    except RuntimeError as exc:
        strict_error = str(exc)
    reference_path = Path("reports/method/reference/utils__loss_function.py")
    spec = importlib.util.spec_from_file_location("locked_reference", reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def tuple_forward(input_ids, mask):
        output = model(input_ids=input_ids, attention_mask=mask)
        return output["sentiment"], output["topic"]

    original = module.SMARTLoss(tuple_forward, module.kl_loss, module.sym_kl_loss)
    reference_runtime = {"source_sha256": file_digest(reference_path)}
    try:
        original_loss = original(ids, clean["sentiment"], attention, sent=True)
        reference_runtime.update(status="executed", loss=float(original_loss.detach()), runtime_error=None)
    except Exception as exc:
        reference_runtime.update(status="runtime_error", runtime_error=f"{type(exc).__name__}: {exc}")
    model.zero_grad(set_to_none=True)
    continuous, diagnostics = smart_loss(model, ids, attention, tokenizer.all_special_ids,
                                         config["smart"], return_diagnostics=True)
    continuous.backward()
    evidence = {"fixture_only": True, "benchmark": False, "torch": torch.__version__,
                "token_ids": ids.tolist(), "cast_ids": cast.tolist(),
                "cast_has_grad_fn": cast.grad_fn is not None,
                "allow_unused_true_gradient_is_none": unused is None,
                "allow_unused_false_error": strict_error, "original_helper_runtime": reference_runtime,
                "continuous_loss": float(continuous.detach()),
                "tasks": {t: {"noise_gradient_norms": d["noise_gradient_norms"],
                              "masked_noise_is_zero": bool((d["noise"] * ~d["eligible_mask"]).eq(0).all()),
                              "max_abs_noise": float(d["noise"].abs().max())} for t, d in diagnostics.items()},
                "model_gradients": {name: {"not_none": parameter.grad is not None,
                                           "finite": bool(torch.isfinite(parameter.grad).all()) if parameter.grad is not None else False,
                                           "norm": float(parameter.grad.norm()) if parameter.grad is not None else None}
                                    for name, parameter in model.named_parameters() if parameter.grad is not None},
                "figure3_cause_proven": False}
    Path("reports/method/gradient_audit.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cast_noise_gradient": "None" if unused is None else "present",
                      "original_helper": reference_runtime["status"],
                      "continuous_noise_gradient_norms": {t: d["noise_gradient_norms"] for t, d in diagnostics.items()}}))


if __name__ == "__main__":
    main()
