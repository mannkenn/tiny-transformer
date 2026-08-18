# Experiment Summary

> ## Status: every performance number below needs re-running
>
> The table in this file was produced before the bugs fixed in
> `fix/attention-fusion-and-eval-bugs`. Specifically, every run below was
> measured with:
>
> - **attention split across `n_heads` separate Python-loop iterations**, each
>   with its own Q/K/V projection — 108 small GEMMs and 36 attention calls per
>   forward pass at the base config, instead of 6 and 6;
> - **step timing taken without a device synchronize**, so the clock was
>   measuring how long it took to *enqueue* work, not to run it;
> - **the first steps of each run included in the throughput average**, which
>   folds allocator warmup and `torch.compile` compilation into the steady-state
>   number;
> - **a `.item()` call inside the gradient accumulation loop**, which forced a
>   host-device sync every micro-step and broke the `torch.compile` graph.
>
> The **loss** figures are still meaningful — the refactor is numerically
> equivalent (proof below), so the optimization behaviour is unchanged. The
> **throughput and memory** figures are not, and should not be quoted.
>
> **The hardware these runs used is not recorded anywhere in this repository.**
> The throughput rules out CPU, but nothing here establishes which GPU it was.
> Runs made from now on write a `run_info.json` recording device name, torch
> version and CUDA version alongside the log.
>
> Re-running requires a CUDA machine. The owner's current machine is Apple
> Silicon with no CUDA. See "What needs re-measuring" at the bottom.

## Results (pre-refactor, hardware unrecorded)

Preserved exactly as generated. Do not read the Tokens/sec, Step Time or Max
Mem columns as current.

| Run Name | Change | Batch | Grad Accum | Eff Batch | Precision | Compile | Tokens/sec | Step Time (s) | Max Mem (GB) | Val Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smoke Test | debug run on CPU with tiny model | 4 | 1 | 4 | fp32 | no | 14k | 0.008 | 0.02 | 2.9837 |
| Baseline | reference hyperparameters | 64 | 1 | 64 | fp32 | no | 291k | 0.0569 | 3.6 | 1.6111 |
| Batch32 | batch size 32 | 32 | 1 | 32 | fp32 | no | 227k | 0.0377 | 1.898 | 1.5159 |
| Batch128 | batch size 128 | 128 | 1 | 128 | fp32 | no | 275k | 0.1194 | 7.048 | 1.8073 |
| Batch256 | batch size 256 | 256 | 1 | 256 | fp32 | no | 240k | 0.2732 | 13.942 | 2.0223 |
| Grad Accum16X4 | grad accum x4 | 16 | 4 | 64 | fp32 | no | 124k | 0.1363 | 1.068 | 1.6227 |
| Grad Accum32X2 | grad accum x2 | 32 | 2 | 64 | fp32 | no | 245k | 0.0678 | 1.944 | 1.6044 |
| Mp Bf16 | bf16 mixed precision | 64 | 1 | 64 | bf16 | no | 291k | 0.0558 | 3.6 | 1.6111 |
| Flash Att | flash attention | 64 | 1 | 64 | fp32 | no | 293k | 0.057 | 3.597 | 1.6112 |
| Torch Compile | torch.compile | 64 | 1 | 64 | fp32 | yes | 291k | 0.0551 | 3.6 | 1.6111 |

Only one of these runs has a surviving CSV (`results/baseline_20260423_204805/`,
recovered from commit `d89a39d`). The other eight GPU runs cannot be audited or
recomputed — `outputs/` is gitignored and their logs are gone.

## Corrections to the previous conclusions

### "Training is memory-bandwidth limited" — not supported

The previous version of this file concluded from the batch sweep that training
was memory-bandwidth limited. The evidence given was that throughput did not
scale proportionally with batch size. That observation rules out being *purely*
compute-bound, but it does not distinguish bandwidth limits from fixed per-step
overhead, and no bandwidth measurement was ever taken.

The more likely explanation is per-operation overhead. The model was issuing
`3 * n_heads * n_layers` = 108 separate QKV GEMMs and `n_heads * n_layers` = 36
separate attention calls per forward pass, on tensors small enough that each one
is dominated by dispatch rather than arithmetic. Larger batches make each
individual GEMM more efficient without reducing how many there are, which
produces exactly the sublinear scaling observed.

This is a hypothesis, not a measurement. Confirming it needs a profiler trace
(`torch.profiler`, or Nsight) on a CUDA machine showing where the step time
actually goes. **Do not state either explanation as fact without that trace.**

### "Flash attention doesn't help at this size" — measured, but on the wrong code

The +0.7% result is real, but it was measured against an implementation that
called `scaled_dot_product_attention` once per head on a `(B, T, head_size)`
tensor. Fusing the softmax into the attention kernel cannot pay for itself when
the tensor is that small and there are 36 launches of it. The experiment tested
"does flash attention help this particular looped implementation", which is a
narrower question than the one the table implies.

### "BF16 is free" — the null result is too clean to trust

The bf16 row is identical to baseline in *all four* reported columns: 291k
tokens/sec, 3.6 GB, val loss 1.6111, to every digit recorded. Throughput and
loss matching is believable. **Peak memory matching exactly is not** — bf16
autocast halves the bytes of the cached activations that dominate this model's
memory, so 3.6 GB should have moved even if nothing else did.

The most likely reading is that autocast was not actually in effect for that
run. The config is correct (`mixed_precision: True`, `dtype: bf16`), so if it
was inactive the cause is in the training loop or the environment, not the
config. This needs to be re-run with an assertion that tensors inside the
autocast region actually have dtype bfloat16 before any claim about bf16 is
made.

The same caution applies to the torch.compile row, also identical to baseline
at 291k / 3.6 GB / 1.6111.

### The comparison metric hides overfitting

Every run is scored by validation loss at step 5000. The one surviving CSV shows
that is well past the optimum:

| step | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 4999 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val loss | 1.5600 | 1.5079 | **1.4986** | 1.5084 | 1.5094 | 1.5449 | 1.5846 | 1.6111 |

Validation loss bottoms out around step 2500 at 1.4986 and climbs for the
remaining half of the run. The "baseline val loss 1.6111" quoted throughout is
the value after 2500 steps of overfitting on a 1.1 MB corpus.

This undercuts the batch-size finding. "Smaller batches generalize better" and
"smaller batches overfit more slowly, so they are less far past their optimum at
step 5000" predict the same table, and the current experiment cannot tell them
apart. `summarize.py` now reports best validation loss and the step it occurred
at, alongside the final value, so a re-run will distinguish them.

## Verification performed after the refactor

All of the following was measured on an **Apple Silicon Mac (no CUDA)**. These
numbers are not comparable to the GPU table above and are not offered as
replacements for it.

### Numerical equivalence of the attention refactor

Built the original looped module and the fused one, copied weights so they are
mathematically identical, compared outputs elementwise
(`tests/test_attention_equivalence.py`):

| path | masking | max abs difference |
| --- | --- | --- |
| manual | causal | **0.000e+00** (bit-exact) |
| manual | bidirectional | **0.000e+00** (bit-exact) |
| SDPA | causal | 1.490e-07 |
| SDPA | bidirectional | 7.451e-08 |

The manual path is bit-exact, which confirms the QKV weight-layout mapping is
exactly right. The SDPA differences are float accumulation order inside the
fused kernel. Input gradients match to the same tolerance.

Training 30 optimizer steps from identical weights on identical batches, fused
vs looped: **max loss difference 4.768e-07**
(`tests/test_training_parity.py`).

Gradient accumulation over 4 micro-batches vs one batch of 4x the size: **max
gradient difference 2.794e-08** (`tests/test_train.py`).

### Attention microbenchmark (CPU and MPS, not CUDA)

`python scripts/bench_attention.py`, base config shape (batch 64, block 256,
n_embd 384, n_heads 6), single layer, median of 50 iterations. Run twice, both
results shown, because absolute timings on this machine move by several percent
between invocations while the ratios hold:

| device | path | looped | fused | speedup |
| --- | --- | --- | --- | --- |
| CPU (Apple Silicon) | SDPA | 40.116 / 43.125 ms | 18.494 / 19.268 ms | **2.17x / 2.24x** |
| CPU (Apple Silicon) | manual | 38.274 / 44.437 ms | 34.990 / 40.928 ms | 1.09x / 1.09x |
| MPS (Apple Silicon) | SDPA | 11.202 / 11.230 ms | 10.268 / 10.261 ms | 1.09x / 1.09x |
| MPS (Apple Silicon) | manual | 11.252 / 11.671 ms | 11.974 / 11.931 ms | 0.94x / 0.98x |

Treat the ratios as the result and the absolute milliseconds as indicative only;
this is an unloaded laptop, not a benchmarking rig.

Note how device-dependent the effect is: ~2.2x on CPU, ~1.09x on MPS, and
slightly *negative* on the MPS manual path. **This does not predict the CUDA
speedup.** The refactor is correct and removes real overhead, but anyone
claiming a specific GPU improvement needs to measure it on the GPU.

### End-to-end training (CPU, 4 layers / 6 heads / 192 embd / 128 ctx / batch 16)

150 steps on `input.txt`. SDPA and manual paths produce the same loss curve to
~4 decimals, confirming `use_flash_attention` is a fair A/B toggle:

| step | 25 | 50 | 75 | 100 | 125 | 149 |
| --- | --- | --- | --- | --- | --- | --- |
| SDPA val loss | 2.6962 | 2.5555 | 2.5249 | 2.4908 | 2.4445 | 2.4411 |
| manual val loss | 2.6963 | 2.5553 | 2.5251 | 2.4908 | 2.4445 | 2.4409 |

Throughput on that config, **CPU**: 58,002 tok/s (SDPA) vs 43,673 tok/s
(manual). The same config on **MPS**: 128,769 tok/s.

### Why the warmup exclusion matters

The MPS run's first step took **2.969 s** against a steady state of **0.0156 s**
— first-call kernel compilation. With the old 20-step rolling average and no
warmup exclusion, the reported throughput for the first 20 steps would have been
roughly 12.5k tok/s instead of the actual 128k tok/s, a 10x understatement.

## What needs re-measuring

Requires a CUDA machine. Nothing here can be filled in on Apple Silicon.

1. **The whole throughput and memory table**, re-run with fused attention, the
   synchronized timer and warmup exclusion.
2. **Flash attention**, which is now a meaningful comparison: one batched
   `scaled_dot_product_attention` over `(B, n_heads, T, head_size)` against the
   equivalent explicit matmul-softmax-matmul, rather than 36 tiny calls either way.
3. **torch.compile**, now that the `.item()` graph break in the accumulation
   loop is gone.
4. **bf16**, with an explicit assertion that autocast is active, to establish
   whether the identical-to-baseline result was real or an artifact.
5. **A profiler trace** to settle whether the sublinear batch scaling is
   per-step overhead or bandwidth. This is the measurement that was missing all
   along.
6. **The two LR-scaling configs** (`batch128_scaled_lr`, `batch256_scaled_lr`),
   which have never been run at all.
7. **Best-validation-loss comparisons** rather than final-step, so the batch
   sweep measures optimization quality instead of overfitting rate.
