# Experiment Summary

> ## Status: two rows are invalid, the rest need re-running
>
> **Invalid — these two experiments never ran.** `mp_bf16` and `torch_compile`
> did not do what their names say. A config-parsing bug meant the settings were
> discarded before `train.py` saw them, so both executed as byte-identical
> repeats of the baseline config. Root cause and evidence below. Their numbers
> are not stale measurements to be refreshed; they are measurements of something
> else entirely.
>
> **Unaffected.** `use_flash_attention` and `grad_accum_steps` *were* being
> parsed when these runs happened, so the flash attention and gradient
> accumulation rows measured what they claim. Their conclusions stand, subject
> only to the general staleness below.
>
> **Stale — every remaining performance number needs re-running.** All rows were
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
> The **loss** figures remain meaningful — the attention refactor is numerically
> equivalent (proof below), so optimization behaviour is unchanged. The
> **throughput and memory** figures are not, and should not be quoted.
>
> **The hardware these runs used is not recorded anywhere in this repository.**
> The throughput rules out CPU, but nothing here establishes which GPU it was.
> Runs made from now on write a `run_info.json` recording device name, torch
> version, CUDA version, and whether mixed precision and `torch.compile` were
> actually active.
>
> Re-running requires a CUDA machine. The owner's current machine is Apple
> Silicon with no CUDA. See "What needs re-measuring" at the bottom.

## Results (pre-refactor, hardware unrecorded)

Preserved exactly as generated, with a status column added. Do not read the
Tokens/sec, Step Time or Max Mem columns as current.

| Status | Run Name | Change | Batch | Grad Accum | Eff Batch | Precision | Compile | Tokens/sec | Step Time (s) | Max Mem (GB) | Val Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stale | Smoke Test | debug run on CPU with tiny model | 4 | 1 | 4 | fp32 | no | 14k | 0.008 | 0.02 | 2.9837 |
| stale | Baseline | reference hyperparameters | 64 | 1 | 64 | fp32 | no | 291k | 0.0569 | 3.6 | 1.6111 |
| stale | Batch32 | batch size 32 | 32 | 1 | 32 | fp32 | no | 227k | 0.0377 | 1.898 | 1.5159 |
| stale | Batch128 | batch size 128 | 128 | 1 | 128 | fp32 | no | 275k | 0.1194 | 7.048 | 1.8073 |
| stale | Batch256 | batch size 256 | 256 | 1 | 256 | fp32 | no | 240k | 0.2732 | 13.942 | 2.0223 |
| stale | Grad Accum16X4 | grad accum x4 | 16 | 4 | 64 | fp32 | no | 124k | 0.1363 | 1.068 | 1.6227 |
| stale | Grad Accum32X2 | grad accum x2 | 32 | 2 | 64 | fp32 | no | 245k | 0.0678 | 1.944 | 1.6044 |
| **INVALID** | Mp Bf16 | bf16 mixed precision — **ran in fp32** | 64 | 1 | 64 | ~~bf16~~ fp32 | no | 291k | 0.0558 | 3.6 | 1.6111 |
| stale | Flash Att | flash attention | 64 | 1 | 64 | fp32 | no | 293k | 0.057 | 3.597 | 1.6112 |
| **INVALID** | Torch Compile | torch.compile — **never compiled** | 64 | 1 | 64 | fp32 | ~~yes~~ no | 291k | 0.0551 | 3.6 | 1.6111 |

The two INVALID rows are the baseline run repeated. Note that they agree with
the baseline row to four decimals on validation loss *and* on peak memory
(3.6 GB), and differ only in step-time jitter (0.0569 / 0.0558 / 0.0551) — the
residual you get from running the same work three times.

Only one of these runs has a surviving CSV (`results/baseline_20260423_204805/`,
recovered from commit `d89a39d`). The other eight GPU runs cannot be audited or
recomputed — `outputs/` is gitignored and their logs are gone.

## Root cause of the two invalid rows

`parse_config` built its output by reaching into the raw YAML with
`cfg.get(name, default)` for each key it knew about, and **silently discarding
every key it did not**. `train.py` then read the *parsed* dict:

```python
use_amp = cfg.get("mixed_precision", False) and torch.cuda.is_available()
if cfg.get("torch_compile", False):
    model = torch.compile(model)
```

A config key that `parse_config` had not been taught about did not raise, did
not warn, and did not take effect. It read as `False`.

The timeline, from `git log -S` on `utils.py`:

| commit | date | what changed |
| --- | --- | --- |
| `c15e431` | 2026-05-01 | `configs/mp/mp_bf16.yaml` and `configs/compile/torch_compile.yaml` created, setting `mixed_precision`, `dtype`, `torch_compile` |
| `1b6bc95` | 2026-05-01 | `parse_config` gains `use_flash_attention` and `grad_accum_steps` |
| `9edd69a` | 2026-05-02 | **the results table is written** |
| `ddbbf2c` | 2026-05-25 | `parse_config` finally gains `mixed_precision`, `dtype`, `torch_compile` — three weeks after the runs |

At `9edd69a`, the commit that produced the table, `parse_config` returned exactly:

```
run_name, learning_rate, batch_size, block_size, n_embd, n_layers, n_heads,
dropout, eval_interval, eval_iters, max_iters, grad_accum_steps,
use_flash_attention, min_lr, warmup_steps, use_lr_scheduler
```

No `mixed_precision`. No `dtype`. No `torch_compile`. `train.py` at that same
commit already had the `torch.compile` branch at line 180 — the code existed,
but the condition guarding it could never be true.

Therefore:

- **`mp_bf16` ran in pure fp32.** `torch.autocast(..., enabled=False)`.
- **`torch_compile` never called `torch.compile`.**
- Both were byte-identical repeats of the baseline config: same seed (1337),
  same data order, same kernels.

**`use_flash_attention` and `grad_accum_steps` were parsed at `1b6bc95`, before
the runs.** The flash attention row and both gradient accumulation rows are
genuine measurements. Two of the five claimed optimizations were never
exercised; three were.

### What prevents a recurrence

`parse_config` is now a declarative `SCHEMA` and **raises on any key it does not
recognise**, naming the key and suggesting the intended one. An option cannot be
declared without being threaded through, because the schema entry *is* the
threading. Additionally:

- `train.py` indexes the parsed config directly instead of using `.get()`, so a
  missing key raises rather than defaulting to "feature off".
- Requesting `mixed_precision` on a non-CUDA device prints an explicit warning
  instead of quietly running fp32.
- When mixed precision is active, a forward pass is checked to confirm the
  logits really come back in the requested dtype; if not, the run aborts.
- `run_info.json` records `mixed_precision_active` and `torch_compile_active`
  separately from what was requested, so the artifact states what the run did.
- `summarize.py` renders a row that requested bf16 but ran fp32 as
  `fp32 (bf16 requested, INACTIVE)`.

Had any one of these existed in May, the two invalid rows would have been
self-evident instead of being published as null results.

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

### "BF16 is effectively a free optimization" — withdrawn, the run was fp32

This claim came from a run that executed in fp32. See the root cause above.
Peak memory matching the baseline exactly (3.6 GB) was the tell: bf16 autocast
halves the bytes of the cached activations that dominate this model's memory,
so that number should have moved even if throughput did not.

**Nothing is known about bf16 on this model.** Not that it helps, not that it is
free, not that it is neutral. The experiment has not been run. `configs/mp/
mp_bf16.yaml` is correct and will now do what it says, but it needs CUDA.

### "torch.compile shows no measurable speedup" — withdrawn, it never compiled

Same root cause. `torch.compile` was never called; the model ran eager. The
previous explanation offered for the null result ("small models spend more time
in Python overhead and compilation warmup than saved compute") is a plausible
statement about small models in general and was not tested by this experiment.

Worth noting how the accounting works out: on this machine a compiled CPU smoke
run took **12.49 s** on step 0 against a **0.0034 s** steady state. Compilation
cost is real and large, and with the old 20-step rolling average and no warmup
exclusion it would have dragged the reported throughput down by roughly two
orders of magnitude — the opposite of a null result. Getting an exactly-baseline
number was only possible because compilation never happened at all.

### "Smaller batches generalize better" — confounded, and the ranking inverts

Every run is scored by validation loss at step 5000. The one surviving CSV shows
that is deep in the overfit regime:

| step | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 4999 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train loss | 1.3118 | 1.2279 | 1.1517 | 1.0929 | 1.0297 | 0.9680 | 0.9054 | 0.8362 |
| val loss | 1.5600 | 1.5079 | **1.4986** | 1.5084 | 1.5094 | 1.5449 | 1.5846 | 1.6111 |

Training loss falls monotonically from 1.3118 to 0.8362 across that span while
validation loss climbs from its minimum. Validation bottoms out at **1.4986
around step 2500**; the "baseline val loss 1.6111" quoted throughout is the
value after 2500 further steps of overfitting a 1.1 MB corpus.

**This inverts the headline conclusion.** The previous version of this file
named Batch32 the best config in the sweep at 1.5159. But the baseline's *best*
validation loss is **1.4986**, which beats Batch32's final-step 1.5159. The
ranking was an artifact of where the ruler was held.

The two competing explanations are:

1. Smaller batches produce noisier gradients that generalize better.
2. Smaller batches overfit a small corpus more slowly, so at a fixed late step
   they are less far past their own optimum.

These predict the same table. **The experiment as run cannot distinguish them**,
and the second is the more mundane. The original write-up asserted the first.

What can and cannot be concluded from what survives:

- **Can:** the baseline overfits, with a clear minimum at step 2500, and
  final-step scoring is the wrong metric for this corpus and step budget.
- **Cannot:** any ranking among batch32 / batch128 / batch256. Their CSVs are
  gone, so their best-val losses and the steps they occurred at are
  unrecoverable. The only figures that survive for them are final-step values,
  which is precisely the metric now known to be misleading. **No re-ranking is
  computed here**, because doing so would require numbers that no longer exist.

A correct comparison needs either early stopping or best-val scoring, on a
re-run. `summarize.py` now reports best validation loss and the step it occurred
at as the primary convergence metric, with final-step loss secondary, and emits
an explicit warning whenever a run's best step precedes its last step.

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

`torch.compile` is worse. A compiled CPU smoke run took **12.4916 s** on step 0
against a **0.0034 s** steady state, a factor of ~3,700. Any throughput average
including that step is meaningless.

### The config bug is caught now

Running a config with a key `parse_config` does not recognise:

```
utils.ConfigError: unknown config key(s): 'mixed_precission' (did you mean
'mixed_precision'?). Valid keys are: batch_size, block_size, deterministic,
device, dropout, dtype, ... An unrecognised key is an error rather than a
no-op because silently ignoring one is what made the mp_bf16 and
torch_compile experiments rerun the baseline instead.
```

Requesting mixed precision where it cannot run:

```
WARNING: mixed_precision=True but device is cpu; autocast is CUDA-only in this
project, so this run is fp32. Any bf16 claim from it would be false.
```

and the resulting `run_info.json` records the truth rather than the request:

```json
"mixed_precision_requested": true,
"mixed_precision_active": false,
"mixed_precision_status": "requested but unsupported on cpu"
```

which `summarize.py` renders in the table as
`fp32 (bf16 requested, INACTIVE)`.

## What needs re-measuring

Requires a CUDA machine. Nothing here can be filled in on Apple Silicon.

1. **The whole throughput and memory table**, re-run with fused attention, the
   synchronized timer and warmup exclusion.
2. **Flash attention**, which is now a meaningful comparison: one batched
   `scaled_dot_product_attention` over `(B, n_heads, T, head_size)` against the
   equivalent explicit matmul-softmax-matmul, rather than 36 tiny calls either way.
   The existing +0.7% row is a real measurement, just of the old implementation.
3. **torch.compile — for the first time.** Not a re-measurement. The previous
   run never called it.
4. **bf16 — for the first time.** Not a re-measurement. The previous run was
   fp32. `verify_autocast()` now aborts the run if the logits do not come back
   in the requested dtype, so the failure mode cannot repeat silently.
5. **A profiler trace** to settle whether the sublinear batch scaling is
   per-step overhead or bandwidth. This is the measurement that was missing all
   along.
6. **The two LR-scaling configs** (`batch128_scaled_lr`, `batch256_scaled_lr`),
   which have never been run at all.
7. **Best-validation-loss comparisons** rather than final-step, so the batch
   sweep measures optimization quality instead of overfitting rate.
