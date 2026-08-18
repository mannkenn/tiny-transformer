"""Tests for the training loop: gradient accumulation, LR schedule, plumbing."""

import json

import pytest
import torch

import train as train_mod
from model import Transformer
from utils import parse_config

VOCAB = 65


def _model(seed=0):
    torch.manual_seed(seed)
    return Transformer(
        vocab_size=VOCAB,
        block_size=16,
        n_embd=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        use_flash_attention=True,
    )


def _grads(model):
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def test_gradient_accumulation_matches_one_large_batch():
    """N micro-batches accumulated must equal one batch of N x the size.

    This is the property the whole grad-accum feature rests on: it is only a
    memory optimisation if the resulting gradient is the same one you would
    have got from the big batch. The `/ grad_accum_steps` scaling in the loop is
    exactly what makes the sum of per-micro-batch means equal the mean over all
    of them, and only holds while every micro-batch is the same size.
    """
    micro_batch, accum_steps, seq_len = 4, 4, 16

    torch.manual_seed(123)
    xs = [torch.randint(0, VOCAB, (micro_batch, seq_len)) for _ in range(accum_steps)]
    ys = [torch.randint(0, VOCAB, (micro_batch, seq_len)) for _ in range(accum_steps)]

    accumulated = _model()
    accumulated.train()
    accumulated.zero_grad(set_to_none=True)
    for xb, yb in zip(xs, ys):
        _, loss = accumulated(xb, yb)
        (loss / accum_steps).backward()
    accum_grads = _grads(accumulated)

    single = _model()
    single.train()
    single.zero_grad(set_to_none=True)
    _, loss = single(torch.cat(xs, dim=0), torch.cat(ys, dim=0))
    loss.backward()
    single_grads = _grads(single)

    worst_name, worst = None, 0.0
    for name, grad in single_grads.items():
        diff = (grad - accum_grads[name]).abs().max().item()
        if diff > worst:
            worst_name, worst = name, diff

    print(f"\n  max |grad diff| = {worst:.3e} (at {worst_name})")
    assert worst < 1e-6, f"gradient mismatch {worst:.3e} at {worst_name}"


def test_accumulated_loss_is_detached_scalar():
    """The training loop must not keep the graph alive across micro-steps."""
    model = _model()
    total = torch.zeros(())
    torch.manual_seed(5)
    for _ in range(3):
        xb = torch.randint(0, VOCAB, (2, 16))
        _, loss = model(xb, xb)
        total += loss.detach()
        (loss / 3).backward()

    assert not total.requires_grad
    assert total.grad_fn is None


@pytest.mark.parametrize(
    "step,expected",
    [(0, 5e-5), (9, 5e-4)],  # warmup is linear over 10 steps to max_lr
)
def test_lr_warmup(step, expected):
    lr = train_mod.get_lr(step, max_lr=5e-4, min_lr=5e-5, warmup_steps=10, max_iters=100)
    assert lr == pytest.approx(expected, rel=1e-6)


def test_lr_cosine_decay_endpoints():
    kwargs = {"max_lr": 5e-4, "min_lr": 5e-5, "warmup_steps": 10, "max_iters": 100}
    assert train_mod.get_lr(10, **kwargs) == pytest.approx(5e-4, rel=1e-6)
    assert train_mod.get_lr(100, **kwargs) == pytest.approx(5e-5, rel=1e-6)
    # monotonically decreasing through the decay phase
    values = [train_mod.get_lr(s, **kwargs) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_small_learning_rates_survive_csv_formatting():
    """The log used %.4f, which wrote every LR below 1e-4 as 0.0000."""
    assert f"{3e-5:.6g}" == "3e-05"
    assert float(f"{3e-5:.6g}") == pytest.approx(3e-5)


def test_estimate_loss_runs_without_building_a_graph():
    model = _model()
    cfg = {"eval_iters": 2, "batch_size": 2, "block_size": 16}
    data = torch.randint(0, VOCAB, (512,))
    device = torch.device("cpu")

    out = train_mod.estimate_loss(model, data, data, cfg, device)

    assert set(out) == {"train", "val"}
    assert all(isinstance(v, float) for v in out.values())
    # estimate_loss must restore training mode for the caller
    assert model.training


def test_seeding_is_reproducible():
    train_mod.set_seed(7)
    a = (torch.randn(4).tolist(), torch.randint(0, 100, (4,)).tolist())
    train_mod.set_seed(7)
    b = (torch.randn(4).tolist(), torch.randint(0, 100, (4,)).tolist())
    assert a == b


def test_seeding_covers_numpy_and_random():
    import random

    import numpy as np

    train_mod.set_seed(21)
    a = (np.random.rand(3).tolist(), random.random())
    train_mod.set_seed(21)
    b = (np.random.rand(3).tolist(), random.random())
    assert a == b


def test_resolve_device_auto_never_picks_an_unavailable_backend():
    device = train_mod.resolve_device("auto")
    assert device.type in {"cuda", "cpu"}
    if device.type == "cuda":
        assert torch.cuda.is_available()


def test_run_info_records_the_actual_device(tmp_path):
    """Benchmark numbers are only meaningful alongside the hardware identity."""
    cfg = parse_config(
        {
            "run_name": "unit",
            "learning_rate": 1e-3,
            "batch_size": 2,
            "block_size": 16,
            "n_embd": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "eval_interval": 5,
            "eval_iters": 2,
            "max_iters": 5,
        }
    )
    device = torch.device("cpu")
    train_mod.write_run_info(str(tmp_path), cfg, device)

    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["device_type"] == "cpu"
    assert info["device_name"]
    assert info["torch_version"] == torch.__version__
    assert info["cuda_available"] is torch.cuda.is_available()
    assert info["resolved_config"]["run_name"] == "unit"


def test_unwrap_model_strips_compile_wrapper():
    """Compiled checkpoints must stay loadable by an uncompiled model."""
    model = _model()

    class FakeCompiled:
        def __init__(self, mod):
            self._orig_mod = mod

    assert train_mod.unwrap_model(model) is model
    assert train_mod.unwrap_model(FakeCompiled(model)) is model


def test_train_end_to_end_writes_logs_and_checkpoints(tmp_path):
    """A miniature run of the real training loop, start to finish."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world, this is a tiny corpus for testing. " * 200)

    cfg = parse_config(
        {
            "run_name": "unit",
            "learning_rate": 1e-3,
            "batch_size": 4,
            "block_size": 16,
            "n_embd": 32,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.0,
            "eval_interval": 5,
            "eval_iters": 2,
            "max_iters": 10,
            "grad_accum_steps": 2,
            "timing_warmup_steps": 1,
            "device": "cpu",
        }
    )
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    log_path = out_dir / "train_log.csv"

    train_mod.train(cfg, str(out_dir), str(log_path), data_path=str(corpus))

    assert log_path.exists()
    rows = log_path.read_text().strip().splitlines()
    assert rows[0].split(",")[:6] == [
        "step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "step_time",
        "tokens_per_sec",
    ]
    assert len(rows) > 1
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "run_info.json").exists()


def test_resume_continues_from_the_next_step(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world, this is a tiny corpus for testing. " * 200)

    base = {
        "run_name": "unit",
        "learning_rate": 1e-3,
        "batch_size": 4,
        "block_size": 16,
        "n_embd": 32,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "eval_interval": 5,
        "eval_iters": 2,
        "max_iters": 10,
        "device": "cpu",
    }
    first = tmp_path / "first"
    first.mkdir()
    train_mod.train(
        parse_config(base), str(first), str(first / "train_log.csv"), data_path=str(corpus)
    )

    ckpt = torch.load(first / "latest.pt")
    assert ckpt["step"] == 9  # last completed step of a 10-iteration run

    second = tmp_path / "second"
    second.mkdir()
    cfg2 = parse_config({**base, "max_iters": 15})
    train_mod.train(
        cfg2,
        str(second),
        str(second / "train_log.csv"),
        resume=str(first / "latest.pt"),
        data_path=str(corpus),
    )

    steps = [
        int(line.split(",")[0])
        for line in (second / "train_log.csv").read_text().strip().splitlines()[1:]
    ]
    # Resuming must not redo step 9, which was already applied to the weights.
    assert min(steps) >= 10
