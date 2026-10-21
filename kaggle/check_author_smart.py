"""Check that the runner's SMART matches the authors' SMARTLoss bit for bit (CPU, ~5 s).

    python kaggle/check_author_smart.py

Runs reports/method/reference/utils__loss_function.py SMARTLoss and the runner's
author_smart_ids/sym_kl_sum on the same RNG stream, for both task branches.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn

import neu_esc_kaggle as runner

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "author_loss", ROOT / "reports/method/reference/utils__loss_function.py")
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)

torch.manual_seed(0)
VOCAB = 64001
embedding = nn.Embedding(VOCAB, 8)
head_s, head_t = nn.Linear(8, 4), nn.Linear(8, 10)
calls = []


def eval_fn(ids, mask):
    calls.append(ids.clone())
    pooled = (embedding(ids) * mask.unsqueeze(-1)).sum(1)
    return head_s(pooled), head_t(pooled)


ids = torch.randint(0, VOCAB, (16, 128))
ids[:, 0] = 0
ids[:, -10:] = 1
mask = (ids != 1).long()
clean_s, clean_t = eval_fn(ids, mask)
smart = author.SMARTLoss(eval_fn=eval_fn, loss_fn=author.kl_loss, loss_last_fn=author.sym_kl_loss)

for sent in (True, False):
    clean = clean_s if sent else clean_t
    torch.manual_seed(123)
    calls.clear()
    reference = smart(ids, clean, mask, sent=sent)
    reference_ids = calls[-1]  # the final (i == num_steps) forward

    torch.manual_seed(123)
    ours_ids = runner.author_smart_ids(ids)
    ours = runner.sym_kl_sum(eval_fn(ours_ids, mask)[0 if sent else 1], clean)

    assert torch.equal(reference_ids, ours_ids), "perturbed IDs differ from the authors' SMARTLoss"
    assert torch.allclose(reference, ours, rtol=1e-6, atol=1e-6), (reference, ours)
    shifted = ((ids - ours_ids) == 1).float().mean().item()
    print(f"{'sentiment' if sent else 'topic'}: identical IDs and loss "
          f"({float(ours.detach()):.4f}); {shifted:.1%} of tokens shifted to id - 1")
print("OK: runner SMART == authors' SMARTLoss")
