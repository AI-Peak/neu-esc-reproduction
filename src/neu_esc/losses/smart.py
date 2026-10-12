"""Continuous embedding perturbations; token-ID reference path is diagnostic only."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .mlm import eligible_tokens


def kl(a, b, reduction="batchmean"):
    return F.kl_div(F.log_softmax(a.float(), dim=-1), F.softmax(b.float(), dim=-1),
                    reduction=reduction)


def symmetric_kl(a, b):
    # Detach the target of each directed KL, as in the public SMART helper.
    return kl(a, b.detach()) + kl(b, a.detach())


@contextmanager
def stable_dropout(model):
    """SMART should measure perturbations rather than independent dropout samples."""
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def smart_loss(model, input_ids, attention_mask, special_token_ids, config,
               *, return_diagnostics=False):
    if config["steps"] < 1 or config["epsilon"] <= 0:
        raise ValueError("SMART requires steps >= 1 and epsilon > 0")
    mask = eligible_tokens(input_ids, attention_mask, special_token_ids).unsqueeze(-1)
    with stable_dropout(model):
        embeddings = model.embeddings(input_ids)
        positions = model.embedding_position_ids(input_ids)
        clean = model(inputs_embeds=embeddings, attention_mask=attention_mask, position_ids=positions)
        total = embeddings.sum() * 0.0
        diagnostics = {}
        # Independent adversaries per task, following the public two-call design.
        for task in model.tasks:
            noise = (torch.randn_like(embeddings.float()) * config["noise_std"] * mask).detach()
            gradient_norms = []
            for _ in range(config["steps"]):
                noise.requires_grad_()
                perturbed = model(inputs_embeds=embeddings.detach() + noise * mask,
                                  attention_mask=attention_mask, position_ids=positions)[task]
                inner = kl(perturbed, clean[task].detach())
                gradient, = torch.autograd.grad(inner, noise, allow_unused=False)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite SMART noise gradient")
                gradient_norms.append(float(gradient.detach().norm()))
                # L-infinity normalized ascent with explicit radius projection.
                norm = gradient.abs().amax(dim=-1, keepdim=True).clamp_min(config["gradient_eps"])
                noise = (noise.detach() + config["step_size"] * gradient.detach() / norm)
                noise = (noise.clamp(-config["epsilon"], config["epsilon"]) * mask).detach()
            adversarial = model(inputs_embeds=embeddings + noise, attention_mask=attention_mask,
                                position_ids=positions)[task]
            total = total + symmetric_kl(adversarial, clean[task])
            diagnostics[task] = {"noise_gradient_norms": gradient_norms,
                                 "noise": noise.detach(), "eligible_mask": mask.detach()}
    return (total, diagnostics) if return_diagnostics else total


def reference_smart_loss(model, input_ids, attention_mask, states, config):
    """Exact two-task helper operations from the locked repo, including .long().

    No clipping, masking, dropout repair, or None-gradient repair beyond its own
    fallback. Runtime errors (e.g. negative padding IDs) deliberately propagate.
    """
    if model.tasks != ("sentiment", "topic"):
        raise ValueError("Reference diagnostic implements the public two-task SMART only")
    total = input_ids.float().sum() * 0.0
    for task in model.tasks:
        noise = torch.randn_like(input_ids.float(), requires_grad=True) * config["noise_std"]
        for step_index in range(config["steps"] + 1):
            perturbed = model(input_ids=(input_ids + noise).long(),
                              attention_mask=attention_mask)[task]
            if step_index == config["steps"]:
                total = total + kl(perturbed, states[task].detach(), "sum")
                total = total + kl(states[task], perturbed.detach(), "sum")
                break
            loss = kl(perturbed, states[task].detach())
            gradient, = torch.autograd.grad(loss, noise, allow_unused=True)
            updated = noise if gradient is None else noise + config["step_size"] * gradient
            noise = (updated / (updated.abs().amax(dim=-1, keepdim=True)
                                 + config["gradient_eps"])).detach().requires_grad_()
    return total
