# tiny-transformer

A small decoder-only transformer trained on character-level Shakespeare text (`input.txt`) to study how systems-level choices (batch size, memory, kernel efficiency) interact with optimization behavior (validation loss, convergence).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

Each run writes checkpoints and a CSV log to `outputs/<run_name>_<config>_<timestamp>/`. Summarize all completed runs:

```bash
python summarize.py
```

## Model and training defaults

| Setting | Value |
| --- | --- |
| Architecture | 6-layer decoder, 384-dim, 6 heads |
| Context length | 256 characters |
| Optimizer | AdamW, lr = 3e-4 |
| Training steps | 5,000 |
| Data split | 90% train / 10% val |

The baseline config is `configs/base.yaml`. Other configs vary one axis at a time: batch size, gradient accumulation, mixed precision, flash attention, or `torch.compile`.

## Experiment results

Full metrics table and auto-generated analysis: [experiment_summary.md](experiment_summary.md).

### Baseline (batch 64, fp32)

Reference point for all comparisons: **291k tokens/sec**, **3.6 GB** peak memory, validation loss **1.6111** after 5k steps.

### Batch size sweep

| Config | Batch | Throughput | Memory | Val Loss | Takeaway |
| --- | --- | --- | --- | --- | --- |
| batch32 | 32 | 227k | 1.9 GB | **1.5159** | Best validation loss; ~half the memory of baseline |
| baseline | 64 | 291k | 3.6 GB | 1.6111 | Best balance of speed and quality |
| batch128 | 128 | 275k | 7.0 GB | 1.8073 | Slightly faster per step but worse loss without LR scaling |
| batch256 | 256 | 240k | 13.9 GB | 2.0223 | Highest memory, worst loss — large-batch penalty |

**Finding:** Throughput does not scale linearly with batch size on this model; training is memory-bandwidth limited. Smaller batches produce noisier gradients that generalize better on this small dataset, while very large batches degrade validation loss unless learning rate is scaled.

### Gradient accumulation

Both configs target an effective batch of 64:

| Config | Micro-batch | Accum steps | Throughput | Memory | Val Loss |
| --- | --- | --- | --- | --- | --- |
| grad_accum32x2 | 32 | 2 | 245k | 1.9 GB | 1.6044 |
| grad_accum16x4 | 16 | 4 | 124k | 1.1 GB | 1.6227 |

**Finding:** 2× accumulation recovers near-baseline speed and loss at half the memory. 4× accumulation minimizes memory but cuts throughput in half — useful only when memory is the hard constraint.

### Mixed precision (BF16)

Identical to baseline on throughput, memory, and validation loss. Enable with `mixed_precision: True` and `dtype: bf16` in config — no accuracy trade-off observed at this scale.

### Flash attention

293k vs 291k tokens/sec — effectively no change. At 256 context length and 384 embedding dim, the manual attention path is already fast enough that fused kernels do not matter much.

### torch.compile

No measurable speedup (291k vs 291k). Compilation overhead and Python dispatch dominate for a ~10M parameter model.

### LR scaling (not yet run)

Configs `batch128_scaled_lr` and `batch256_scaled_lr` apply the linear scaling rule (lr × batch/64). These are set up to test whether scaled LR recovers the validation loss gap seen in the unscaled large-batch runs.

## Project layout

```
configs/          experiment configs grouped by axis (batch/, attention/, mp/, etc.)
train.py          training loop with logging and checkpointing
model.py          decoder-only transformer
summarize.py      aggregate outputs/*/train_log.csv into experiment_summary.md
dataset.py        optional FineWeb-Edu tokenization script (not used by train.py yet)
input.txt         Shakespeare character-level corpus
```
