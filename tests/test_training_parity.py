"""End-to-end check: swapping in fused attention does not change training.

`test_attention_equivalence.py` proves the module computes the same function.
This goes one level up and trains two identical models -- one using the fused
attention from `model.py`, one with every attention block swapped back to the
original Python-loop implementation -- on byte-for-byte identical batches, then
compares the loss curves step by step.

This is the check that answers "did the refactor change learning behaviour".
Starting both models from the *same* weights is the whole point: a fresh
`Transformer` of each kind would differ simply because the two modules draw
their random initialisation in a different order.
"""

import copy

import torch
from reference_attention import LoopedMultiHeadedSelfAttention, copy_weights_to_looped

from model import Transformer

VOCAB = 65
STEPS = 30
LOSS_TOLERANCE = 1e-4


def _make_looped_twin(model, n_heads, dropout):
    """Deep-copy `model`, replacing each fused attention block with the loop."""
    twin = copy.deepcopy(model)
    for block in twin.decoder_blocks:
        fused = block.mhsa
        looped = LoopedMultiHeadedSelfAttention(
            fused.n_embd,
            n_heads,
            dropout=dropout,
            use_flash_attention=fused.use_flash_attention,
        )
        copy_weights_to_looped(fused, looped)
        block.mhsa = looped
    return twin


def _train_curve(model, batches, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for xb, yb in batches:
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_fused_and_looped_training_curves_match():
    torch.manual_seed(1337)
    model = Transformer(
        vocab_size=VOCAB,
        block_size=32,
        n_embd=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        use_flash_attention=True,
    )
    twin = _make_looped_twin(model, n_heads=4, dropout=0.0)

    torch.manual_seed(4242)
    batches = [
        (torch.randint(0, VOCAB, (8, 32)), torch.randint(0, VOCAB, (8, 32)))
        for _ in range(STEPS)
    ]

    fused_losses = _train_curve(model, batches, lr=1e-3)
    looped_losses = _train_curve(twin, batches, lr=1e-3)

    deltas = [abs(a - b) for a, b in zip(fused_losses, looped_losses)]
    max_delta = max(deltas)

    print(
        f"\n  fused  first/last loss: {fused_losses[0]:.6f} / {fused_losses[-1]:.6f}"
        f"\n  looped first/last loss: {looped_losses[0]:.6f} / {looped_losses[-1]:.6f}"
        f"\n  max |delta| over {STEPS} steps: {max_delta:.3e}"
    )

    assert max_delta < LOSS_TOLERANCE, (
        f"loss curves diverged: max delta {max_delta:.3e} over {STEPS} steps"
    )
