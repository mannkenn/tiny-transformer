"""Measure fused vs per-head-loop attention at a given config shape.

This is the measurement that was missing when the summary table concluded that
flash attention, bf16 and torch.compile were all no-ops. Those experiments
varied the kernel and the dtype while the model was issuing 3 * n_heads small
GEMMs and n_heads attention calls per layer, so per-op overhead dominated and
none of them could show a difference.

Every number printed is labelled with the device it was produced on. Do not
quote a result from this script without that label.

    python scripts/bench_attention.py                    # base config shape
    python scripts/bench_attention.py --device mps
    python scripts/bench_attention.py --batch-size 16 --iters 100
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MultiHeadedSelfAttention  # noqa: E402
from tests.reference_attention import (  # noqa: E402
    LoopedMultiHeadedSelfAttention,
    copy_weights_to_fused,
)
from train import device_description, synchronize  # noqa: E402


def bench(module, x, iters, warmup, device):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(x, is_causal=True)
        synchronize(device)

        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            module(x, is_causal=True)
            synchronize(device)
            samples.append(time.perf_counter() - start)

    return statistics.median(samples) * 1e3  # ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6, help="for the per-op counts")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    print(f"device:  {device_description(device)}  ({device.type})")
    print(f"torch:   {torch.__version__}")
    print(
        f"shape:   batch={args.batch_size} block={args.block_size} "
        f"n_embd={args.n_embd} n_heads={args.n_heads}"
    )
    print(f"timing:  median of {args.iters} iters, {args.warmup} warmup, single layer\n")

    torch.manual_seed(0)
    looped = LoopedMultiHeadedSelfAttention(args.n_embd, args.n_heads, 0.0).to(device)
    x = torch.randn(args.batch_size, args.block_size, args.n_embd, device=device)

    rows = []
    for use_flash in (True, False):
        looped.eval()
        for head in looped.heads:
            head.use_flash_attention = use_flash

        fused = MultiHeadedSelfAttention(
            args.n_embd, args.n_heads, 0.0, use_flash_attention=use_flash
        ).to(device)
        copy_weights_to_fused(looped, fused)

        looped_ms = bench(looped, x, args.iters, args.warmup, device)
        fused_ms = bench(fused, x, args.iters, args.warmup, device)
        rows.append((("sdpa" if use_flash else "manual"), looped_ms, fused_ms))

    width = max(len(r[0]) for r in rows)
    print(f"{'path':<{width}}  {'looped':>10}  {'fused':>10}  {'speedup':>8}")
    for name, looped_ms, fused_ms in rows:
        print(
            f"{name:<{width}}  {looped_ms:>9.3f}ms  {fused_ms:>9.3f}ms  "
            f"{looped_ms / fused_ms:>7.2f}x"
        )

    gemms_looped = 3 * args.n_heads * args.n_layers
    print(
        f"\nper forward pass at n_layers={args.n_layers}: "
        f"{gemms_looped} QKV GEMMs + {args.n_heads * args.n_layers} attention calls "
        f"(looped)  vs  {args.n_layers} + {args.n_layers} (fused)"
    )
    print(f"\nAll timings above were measured on: {device_description(device)}")


if __name__ == "__main__":
    main()
