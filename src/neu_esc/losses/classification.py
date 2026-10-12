import torch.nn.functional as F


def classification_loss(logits, labels, weights):
    """Mean CE per task, then explicit weighted sum (paper default weights = 1)."""
    if not logits or set(logits) != set(weights):
        raise ValueError("Task loss weights must match model tasks exactly")
    losses = {t: F.cross_entropy(value.float(), labels[t], reduction="mean")
              for t, value in logits.items()}
    return sum(weights[t] * value for t, value in losses.items()), losses
