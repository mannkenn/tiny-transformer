"""Proof that fused multi-head attention == the original per-head Python loop.

The refactor in ``model.py`` replaced ``n_heads`` independent Q/K/V projections
and ``n_heads`` attention calls with one fused QKV GEMM and one batched
attention call. That is a performance change only; these tests assert it is not
a numerical one.

Dropout is held at 0.0 throughout. The two implementations draw a different
number of RNG samples (the loop calls dropout once per head, the fused path once
in total), so with dropout active they are equal only in distribution, not
elementwise. Every training config sets dropout at the block level, and the
comparison that matters -- "does this compute the same function" -- is the
deterministic one.
"""

import pytest
import torch

from model import MultiHeadedSelfAttention
from reference_attention import LoopedMultiHeadedSelfAttention, copy_weights_to_fused

TOLERANCE = 1e-5


def _build_pair(n_embd, n_heads, use_flash_attention, seed=0):
    torch.manual_seed(seed)
    looped = LoopedMultiHeadedSelfAttention(
        n_embd, n_heads, dropout=0.0, use_flash_attention=use_flash_attention
    )
    fused = MultiHeadedSelfAttention(
        n_embd, n_heads, dropout=0.0, use_flash_attention=use_flash_attention
    )
    copy_weights_to_fused(looped, fused)
    looped.eval()
    fused.eval()
    return looped, fused


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


@pytest.mark.parametrize("use_flash_attention", [True, False], ids=["sdpa", "manual"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "bidirectional"])
def test_fused_matches_looped_forward(use_flash_attention, is_causal, capsys):
    """Both attention paths, causal and not, must match elementwise."""
    B, T, n_embd, n_heads = 4, 32, 96, 6
    looped, fused = _build_pair(n_embd, n_heads, use_flash_attention)

    torch.manual_seed(1234)
    x = torch.randn(B, T, n_embd)

    with torch.no_grad():
        expected = looped(x, is_causal=is_causal)
        actual = fused(x, is_causal=is_causal)

    diff = _max_abs_diff(expected, actual)
    with capsys.disabled():
        path = "sdpa" if use_flash_attention else "manual"
        mask = "causal" if is_causal else "bidirectional"
        print(f"\n  [{path:6s} / {mask:14s}] max |fused - looped| = {diff:.3e}")

    assert expected.shape == actual.shape
    assert torch.allclose(expected, actual, atol=TOLERANCE), (
        f"max abs diff {diff:.3e} exceeds atol {TOLERANCE}"
    )


@pytest.mark.parametrize("use_flash_attention", [True, False], ids=["sdpa", "manual"])
def test_fused_matches_looped_backward(use_flash_attention):
    """Gradients w.r.t. the input must match too, not just the forward value."""
    B, T, n_embd, n_heads = 2, 16, 96, 6
    looped, fused = _build_pair(n_embd, n_heads, use_flash_attention)

    torch.manual_seed(99)
    x = torch.randn(B, T, n_embd)

    x_looped = x.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)

    looped(x_looped, is_causal=True).sum().backward()
    fused(x_fused, is_causal=True).sum().backward()

    diff = _max_abs_diff(x_looped.grad, x_fused.grad)
    assert torch.allclose(x_looped.grad, x_fused.grad, atol=TOLERANCE), (
        f"input-gradient max abs diff {diff:.3e} exceeds atol {TOLERANCE}"
    )


def test_sdpa_and_manual_paths_agree():
    """The two paths inside the fused module must also agree with each other.

    This is what makes ``use_flash_attention`` a fair A/B toggle: flipping it
    changes which kernel runs, not what is computed.
    """
    B, T, n_embd, n_heads = 4, 32, 96, 6

    torch.manual_seed(7)
    sdpa = MultiHeadedSelfAttention(n_embd, n_heads, dropout=0.0, use_flash_attention=True)
    manual = MultiHeadedSelfAttention(n_embd, n_heads, dropout=0.0, use_flash_attention=False)
    manual.load_state_dict(sdpa.state_dict())
    sdpa.eval()
    manual.eval()

    torch.manual_seed(21)
    x = torch.randn(B, T, n_embd)

    with torch.no_grad():
        out_sdpa = sdpa(x, is_causal=True)
        out_manual = manual(x, is_causal=True)

    diff = _max_abs_diff(out_sdpa, out_manual)
    assert torch.allclose(out_sdpa, out_manual, atol=TOLERANCE), (
        f"sdpa vs manual max abs diff {diff:.3e} exceeds atol {TOLERANCE}"
    )


def test_causal_mask_actually_masks():
    """A causal model's output at position t must not depend on tokens after t.

    Guards against the failure mode where ``is_causal`` silently stops being
    threaded through and the model quietly starts seeing the future.
    """
    B, T, n_embd, n_heads = 1, 16, 96, 6

    torch.manual_seed(5)
    attn = MultiHeadedSelfAttention(n_embd, n_heads, dropout=0.0, use_flash_attention=True)
    attn.eval()

    torch.manual_seed(11)
    x = torch.randn(B, T, n_embd)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0  # change only the final position

    with torch.no_grad():
        out = attn(x, is_causal=True)
        out_perturbed = attn(x_perturbed, is_causal=True)

    # Everything before the last position must be untouched.
    assert torch.allclose(out[:, :-1], out_perturbed[:, :-1], atol=TOLERANCE)
    # ...and the last position must actually have changed, or the test is vacuous.
    assert not torch.allclose(out[:, -1], out_perturbed[:, -1], atol=1e-3)
