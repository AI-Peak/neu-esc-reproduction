from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MaskedTokens:
    input_ids: torch.Tensor
    labels: torch.Tensor
    selected: torch.Tensor
    replacements: torch.Tensor  # -1 unselected, 0 MASK, 1 random, 2 unchanged


def eligible_tokens(input_ids, attention_mask, special_token_ids):
    eligible = attention_mask.bool().clone()
    for token_id in special_token_ids:
        eligible &= input_ids.ne(token_id)
    return eligible


def mask_tokens(input_ids, attention_mask, special_token_ids, mask_token_id,
                vocab_size, mask_rate=0.3):
    if mask_token_id is None or not 0 <= mask_token_id < vocab_size:
        raise ValueError("Tokenizer must have a valid MASK token")
    if not 0 <= mask_rate <= 1:
        raise ValueError("Invalid mask rate")
    eligible = eligible_tokens(input_ids, attention_mask, special_token_ids)
    selected = (torch.rand(input_ids.shape, device=input_ids.device) < mask_rate) & eligible
    labels = input_ids.clone()
    labels[~selected] = -100
    result = input_ids.clone()
    choice = torch.rand(input_ids.shape, device=input_ids.device)
    masked = selected & (choice < 0.8)
    random = selected & (choice >= 0.8) & (choice < 0.9)
    result[masked] = mask_token_id
    # Sample actual vocabulary IDs; exclude special IDs from random replacements as well.
    vocabulary = torch.ones(vocab_size, dtype=torch.bool, device=input_ids.device)
    vocabulary[list(special_token_ids)] = False
    candidates = vocabulary.nonzero().flatten()
    if len(candidates) == 0:
        raise ValueError("No non-special vocabulary tokens")
    result[random] = candidates[torch.randint(len(candidates), (int(random.sum()),),
                                              device=input_ids.device)]
    replacements = torch.full_like(input_ids, -1)
    replacements[masked] = 0
    replacements[random] = 1
    replacements[selected & (choice >= 0.9)] = 2
    return MaskedTokens(result, labels, selected, replacements)


def mlm_loss(logits, labels, denominator="selected_tokens"):
    """(sum token CE / batch size) / (selected count / batch size) = sum / count.

    The sum-per-sentence interpretation avoids dividing a mean CE twice.
    """
    if denominator != "selected_tokens":
        raise ValueError("Supported MLM denominator: selected_tokens")
    count = int(labels.ne(-100).sum())
    if count == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction="sum") / count
