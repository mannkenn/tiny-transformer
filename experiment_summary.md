## Experiment Summary

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

## Findings

### Smoke test
- Quick sanity check on CPU: 14k tok/s, val loss 2.9837. Used to verify the training loop before GPU experiments.

### Baseline
- Throughput: **291k tokens/sec** at batch 64, fp32, manual attention.
- Memory: **3.6 GB** peak allocated.
- Validation loss after 5k steps: **1.6111**.

### Batch size sweep
- Larger batches increase memory roughly linearly but do not scale throughput proportionally on this model — training appears memory-bandwidth limited.
- **Batch32**: 227k tok/s, 1.898 GB, val loss 1.5159 (−47% memory vs baseline, +5.9% val loss improvement).
- **Batch128**: 275k tok/s, 7.048 GB, val loss 1.8073 (+96% memory vs baseline, −12.2% val loss vs baseline).
- **Batch256**: 240k tok/s, 13.942 GB, val loss 2.0223 (+287% memory vs baseline, −25.5% val loss vs baseline).
- Best optimization quality in this sweep: **Batch32** (val loss 1.5159). Fastest among batch variants: **Batch128** (275k tok/s).
- Without LR scaling, larger batches hurt convergence — batch 256 reaches the worst validation loss despite the highest per-step compute.

### Gradient accumulation
- Gradient accumulation simulates a larger effective batch while keeping per-step activation memory low.
- **Grad Accum16X4** (eff batch 64): 124k tok/s, 1.068 GB, val loss 1.6227.
- **Grad Accum32X2** (eff batch 64): 245k tok/s, 1.944 GB, val loss 1.6044.
- **Grad Accum32X2** is the better trade-off: near-baseline throughput and validation loss at roughly half the memory footprint. Four accumulation steps cut memory further but roughly halve throughput.

### Mixed precision (BF16)
- **BF16** matches baseline on throughput (291k), memory (3.6 GB), and validation loss (1.6111). On supported GPUs this is effectively a free optimization.

### Flash attention
- Flash attention yields a negligible throughput change (293k vs 291k) at this model size. Kernel fusion benefits grow with sequence length and head count.

### torch.compile
- **torch.compile** shows no measurable speedup here (291k vs 291k). Small models often spend more time in Python overhead and compilation warmup than saved compute.

### LR scaling (linear rule)
- Configs exist for batch 128/256 with linearly scaled learning rates but have not been run yet. Compare against the unscaled batch128/batch256 runs to see whether LR scaling closes the validation gap.
