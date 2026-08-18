"""Shape, masking and learning-behaviour checks on the full model."""

import pytest
import torch

from model import Transformer

VOCAB = 65


def _model(**overrides):
    kwargs = {
        "vocab_size": VOCAB,
        "block_size": 32,
        "n_embd": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "use_flash_attention": True,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return Transformer(**kwargs)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("seq_len", [1, 32])
def test_forward_shapes(batch_size, seq_len):
    model = _model().eval()
    idx = torch.randint(0, VOCAB, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(idx)
    assert logits.shape == (batch_size, seq_len, VOCAB)
    assert loss is None

    with torch.no_grad():
        logits, loss = model(idx, idx)
    assert logits.shape == (batch_size, seq_len, VOCAB)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_forward_rejects_sequences_longer_than_block_size():
    """Positional embeddings are only defined up to block_size."""
    model = _model(block_size=16).eval()
    idx = torch.randint(0, VOCAB, (2, 17))
    with pytest.raises(AssertionError):
        model(idx)


def test_untrained_loss_is_near_uniform():
    """A fresh model should sit close to -log(1/vocab_size).

    Catches gross initialisation or label-alignment mistakes: if the loss starts
    far from chance, something is wrong before a single step is taken.
    """
    model = _model().eval()
    torch.manual_seed(3)
    idx = torch.randint(0, VOCAB, (16, 32))

    with torch.no_grad():
        _, loss = model(idx, idx)

    expected = torch.log(torch.tensor(float(VOCAB)))
    assert abs(loss.item() - expected.item()) < 0.5, (
        f"initial loss {loss.item():.3f} is far from chance {expected.item():.3f}"
    )


@pytest.mark.parametrize("norm_first", [False, True], ids=["post_norm", "pre_norm"])
@pytest.mark.parametrize("use_flash_attention", [True, False], ids=["sdpa", "manual"])
def test_overfits_a_single_fixed_batch(norm_first, use_flash_attention):
    """The loop must be able to drive loss down on one memorised batch.

    This is the cheapest end-to-end signal that forward, backward, the optimizer
    and the target alignment all agree with each other. A model that cannot
    overfit 8x32 tokens has a real bug somewhere.
    """
    model = _model(norm_first=norm_first, use_flash_attention=use_flash_attention)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    torch.manual_seed(11)
    xb = torch.randint(0, VOCAB, (8, 32))
    yb = torch.randint(0, VOCAB, (8, 32))

    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    # Random targets on a fixed batch are pure memorisation, so a working loop
    # should get well below the chance level of log(65) ~= 4.17.
    assert losses[-1] < 1.0, (
        f"loss only reached {losses[-1]:.4f} after 60 steps (started {losses[0]:.4f})"
    )


def test_pre_norm_and_post_norm_are_different_architectures():
    """Guards the default: norm_first must actually change the computation."""
    post = _model(norm_first=False).eval()
    pre = _model(norm_first=True).eval()
    pre.load_state_dict(post.state_dict())

    torch.manual_seed(17)
    idx = torch.randint(0, VOCAB, (4, 32))

    with torch.no_grad():
        post_logits, _ = post(idx)
        pre_logits, _ = pre(idx)

    assert not torch.allclose(post_logits, pre_logits, atol=1e-4)
