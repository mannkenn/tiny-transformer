# tiny-transformer

A small decoder-only transformer trained on character-level Shakespeare text
(`input.txt`) to study how systems-level choices (batch size, memory, kernel
efficiency) interact with optimization behavior (validation loss, convergence).

> **On the benchmark numbers in this repo.** Two rows of the results table —
> `mp_bf16` and `torch_compile` — are **invalid**: a config-parsing bug meant
> those settings were discarded before `train.py` saw them, so both ran as
> repeats of the baseline. bf16 and `torch.compile` have never actually been
> measured on this model. The flash attention and gradient accumulation rows are
> unaffected. Everything else was measured on a rented GPU that is no longer
> accessible, on hardware that was never recorded, and before the attention and
> timing bugs were fixed — loss figures remain valid, throughput and memory need
> re-running on CUDA. Full accounting in
> [experiment_summary.md](experiment_summary.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions this repo was developed against. On a CUDA
machine, install torch from the matching index rather than the default wheel.

## Running experiments

Train a single config:

```bash
python train.py --config base
python train.py --config batch/batch128
python train.py --config configs/mp/mp_bf16.yaml
```

Run all configs sequentially (logs to `logs/`):

```bash
bash run_experiments.sh
```

Each run writes checkpoints, a CSV log and a `run_info.json` to
`outputs/<run_name>_<config>_<timestamp>/`.

Override the device (defaults to CUDA when available, else CPU):

```bash
python train.py --config smoke_test --device mps
```

Summarize completed runs:

```bash
python summarize.py                                  # print a report
python summarize.py --out experiment_summary.md      # write it (refuses to clobber)
```

`summarize.py` prints by default and only writes with `--out`. It will not
overwrite an existing file without `--force`, because `experiment_summary.md` is
the only surviving record of several runs whose CSVs are gone.

## Tests

```bash
pytest
```

34 CPU-only tests, ~3 seconds. The important ones:

| Test | What it pins down |
| --- | --- |
| `test_attention_equivalence.py` | Fused attention is numerically identical to the original per-head loop, on both the SDPA and manual paths, forward and backward |
| `test_training_parity.py` | 30 optimizer steps from identical weights produce the same loss curve either way |
| `test_train.py::test_gradient_accumulation_matches_one_large_batch` | N accumulated micro-batches give the same gradient as one batch of N x the size |
| `test_model.py::test_overfits_a_single_fixed_batch` | The loop can drive loss down on a memorised batch — the cheapest end-to-end signal that forward, backward, optimizer and target alignment agree |

`tests/reference_attention.py` is a frozen copy of the pre-refactor attention
implementation. It exists only so the equivalence proof stays runnable; it is
deliberately not modernised.

## Model and training defaults

| Setting | Value |
| --- | --- |
| Architecture | 6-layer decoder, 384-dim, 6 heads (10.79M params) |
| Context length | 256 characters |
| Optimizer | AdamW, lr = 3e-4 |
| Training steps | 5,000 |
| Data split | 90% train / 10% val |
| Normalization | post-LayerNorm (see below) |

The baseline config is `configs/base.yaml`. Other configs vary one axis at a
time: batch size, gradient accumulation, mixed precision, flash attention, or
`torch.compile`.

### Attention

Multi-head attention uses a single `nn.Linear(n_embd, 3 * n_embd)` producing Q,
K and V in one GEMM, reshaped to `(B, n_heads, T, head_size)`, followed by one
batched attention call per layer.

An earlier version held an `nn.ModuleList` of `n_heads` separate head modules,
each with its own Q/K/V projection, and looped over them in Python. At the base
config that meant 108 small GEMMs and 36 attention calls per forward pass. The
two are mathematically identical — see the test table above — but the looped
version spent most of its time on per-operation overhead rather than arithmetic,
which is the most likely reason flash attention, bf16 and `torch.compile` all
measured as no-ops.

`use_flash_attention: False` switches to an explicit
matmul → mask → softmax → matmul implementation. Both paths compute the same
thing (asserted in the tests), so the flag is a fair A/B comparison of kernels.

### Pre-norm vs post-norm

`DecoderBlock` defaults to **post-LayerNorm**, the original "Attention Is All
You Need" arrangement:

```python
x = self.ln1(x + self.mhsa(x, is_causal=True))
x = self.ln2(x + self.ff(x))
```

GPT-2 and nanoGPT use **pre-LayerNorm**, where the normalization moves inside
the residual branch:

```python
x = x + self.mhsa(self.ln1(x))
x = x + self.ff(self.ln2(x))
```

The difference matters because pre-norm leaves an unnormalized path from input
to output. Every residual branch adds into a stream that is never rescaled, so
gradients reach early layers without passing through a LayerNorm each time. That
is what makes deep transformers trainable without a long learning-rate warmup;
post-norm models get progressively harder to train as depth grows. At 6 layers
either works, which is why the original choice was never a problem here.

Set `norm_first: True` in a config to use pre-norm. The default stays post-norm
so the existing results table remains a valid comparison — switching it would
silently invalidate every number in it.

## Reproducibility

- `seed` (default 1337) seeds torch, numpy and `random`.
- `deterministic: True` additionally selects deterministic kernels, at some cost
  in throughput.
- `timing_warmup_steps` (default 5) excludes the first steps of a run from the
  rolling throughput average, so allocator warmup and `torch.compile`
  compilation do not contaminate it. Runs shorter than the warmup report `nan`
  rather than a number that would be wrong.
- Step timing calls `torch.cuda.synchronize()` / `torch.mps.synchronize()`
  before reading the clock. Without it the timer measures how long it took to
  queue the work, not to do it.

## Project layout

```
configs/          experiment configs grouped by axis (batch/, attention/, mp/, etc.)
train.py          training loop with logging, checkpointing and provenance
model.py          decoder-only transformer
summarize.py      aggregate run logs into a markdown report
utils.py          config loading and validation
tests/            pytest suite (CPU-only, ~3s)
scripts/          bench_attention.py, prepare_fineweb.py (not wired into training)
results/          committed run artifacts, see results/README.md
outputs/          scratch dir for new runs (gitignored)
input.txt         Shakespeare character-level corpus
```

## Experiment results

Full table, corrections and the list of what still needs re-measuring:
[experiment_summary.md](experiment_summary.md). Committed raw artifacts and
their caveats: [results/README.md](results/README.md).

Summary of where things stand:

- **Two experiments never ran.** `mp_bf16` executed in fp32 and `torch_compile`
  never called `torch.compile`, because `parse_config` silently discarded config
  keys it had not been taught about. Both were baseline repeats, which is why
  three rows agree to four decimals. Nothing is known about bf16 or
  `torch.compile` on this model — not that they help, not that they are free.
- **Flash attention and gradient accumulation are unaffected.** Those keys were
  parsed when the runs happened, so those rows measured what they claim.
- **Loss and convergence results are valid.** The attention refactor is proven
  numerically equivalent, so nothing about optimization behaviour changed.
- **Throughput and memory results are stale.** They were measured with the
  looped attention and an unsynchronized timer, on unrecorded hardware.
- **The batch-size conclusion is confounded, and the ranking inverts.** Every run
  is scored at step 5000, but the surviving CSV shows validation loss bottoming
  out at **1.4986 at step 2500** and rising to 1.6111 by the end. The baseline's
  best beats Batch32's reported 1.5159, so "smaller batches generalize better"
  cannot be distinguished from "smaller batches overfit more slowly". The other
  runs' CSVs are gone, so no corrected ranking can be computed.
