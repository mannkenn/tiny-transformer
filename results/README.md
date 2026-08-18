# results/

Committed run artifacts, so the numbers quoted in the README and
`experiment_summary.md` are inspectable from a clone.

`outputs/` stays gitignored: it is the scratch directory new runs write to.
When a run is worth keeping, copy its `train_log.csv`, `config.yaml` and
`run_info.json` here. Checkpoints (`*.pt`) stay out of git.

## What is here

| Directory | Source | Device | Status |
| --- | --- | --- | --- |
| `baseline_20260423_204805/` | recovered from commit `d89a39d` | unrecorded | pre-refactor |
| `smoke_test_20260525_130924/` | committed from `outputs/` | CPU (unrecorded) | pre-refactor |

## What is *not* here, and why

The results table in `experiment_summary.md` covers nine GPU runs. **The raw
CSVs for eight of them no longer exist.** `outputs/` is gitignored, only one CSV
was ever committed, and a later commit deleted it. The table is currently the
only surviving record of those runs, and it cannot be recomputed or audited.

`baseline_20260423_204805/train_log.csv` is that one CSV, recovered with
`git show d89a39d:outputs/baseline_20260423_204805/train_log.csv`.

## Caveats on the recovered baseline

Read these before quoting anything from it.

**The hardware is not recorded.** The CSV contains no device information, and
neither does its `config.yaml`. The throughput (~287k tokens/sec at 10.8M
params, batch 64, context 256) is far beyond what a CPU does, so it ran on some
GPU, but *which* GPU is not established by anything in this repository. Runs
made after this branch write a `run_info.json` next to the log recording the
device name, torch version and CUDA version, so this gap does not recur.

**It is not the run in the summary table.** Mean throughput excluding step 0 is
287,220 tokens/sec; the table reports 291k for the baseline row. Final
validation loss matches exactly (1.6111), which is expected given the fixed
seed. So the table was generated from a later baseline run whose CSV is gone.

**The schema is older.** Columns are `step,train_loss,val_loss,lr,elapsed_time,
tokens_per_sec` — no memory columns, and `lr`/`elapsed_time` were later renamed
to `learning_rate`/`step_time`. `summarize.py` maps the old names onto the new
ones so this file stays readable.

**The timing method was wrong when it was recorded.** Step time was measured
without a CUDA synchronize (see the commit fixing this). The numbers are in the
right ballpark because a `.item()` call happened to force a sync, but they were
never a clean measurement.

**The headline validation loss is the post-overfitting one.** The full curve:

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 4999 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3.7405 | 1.7939 | 1.4477 | 1.3118 | 1.2279 | 1.1517 | 1.0929 | 1.0297 | 0.9680 | 0.9054 | 0.8362 |
| val | 3.7688 | 1.9217 | 1.6383 | 1.5600 | 1.5079 | **1.4986** | 1.5084 | 1.5094 | 1.5449 | 1.5846 | 1.6111 |

Validation loss bottoms out at **1.4986 around step 2500** and then climbs for
the rest of the run while training loss keeps falling — textbook overfitting on
a 1.1 MB corpus. The 1.6111 quoted everywhere as "the baseline" is the value
*after* 2500 steps of overfitting.

This matters for how the comparison table reads. Every run is scored at step
5000, so the table ranks configs by how much they had overfit by then, not by
how well they train. That is the most likely explanation for the batch-size
sweep's headline finding (smaller batch = better validation loss): smaller
batches are noisier and overfit a small corpus more slowly. Scoring each run at
its own best validation loss, or adding early stopping, would test that.

## Re-running

Every throughput and memory number in this repo predates the attention fusion
and the timing fix, and none of it can be reproduced on the owner's current
machine (Apple Silicon, no CUDA). See the top of `experiment_summary.md` for
what specifically needs to be re-measured.
