import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import yaml

from utils import parse_config

# Older runs used different column names. Keep reading them rather than
# stranding the only surviving pre-refactor CSV.
LEGACY_COLUMNS = {
    "lr": "learning_rate",
    "elapsed_time": "step_time",
}

BASELINE = {
    "batch_size": 64,
    "grad_accum_steps": 1,
    "learning_rate": 3e-4,
    "use_flash_attention": False,
    "mixed_precision": False,
    "torch_compile": False,
}

RUN_ORDER = {
    "smoke_test": 0,
    "baseline": 1,
    "batch32": 2,
    "batch128": 3,
    "batch256": 4,
    "batch128_scaled_lr": 5,
    "batch256_scaled_lr": 6,
    "grad_accum16x4": 7,
    "grad_accum32x2": 8,
    "mp_bf16": 9,
    "flash_att": 10,
    "torch_compile": 11,
}

CATEGORY_MAP = {
    "smoke_test": "Sanity check",
    "baseline": "Baseline",
    "batch32": "Batch size sweep",
    "batch128": "Batch size sweep",
    "batch256": "Batch size sweep",
    "batch128_scaled_lr": "LR scaling",
    "batch256_scaled_lr": "LR scaling",
    "grad_accum16x4": "Gradient accumulation",
    "grad_accum32x2": "Gradient accumulation",
    "mp_bf16": "Mixed precision",
    "flash_att": "Attention kernel",
    "torch_compile": "Compilation",
}


def dataframe_to_markdown(df):
    if df.empty:
        return "|  |\n|---|"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def format_tokens(x):
    if pd.isna(x):
        return ""

    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"

    if x >= 1_000:
        return f"{x / 1_000:.0f}k"

    return str(round(x, 0))


def load_run_config(csv_path):
    config_path = csv_path.parent / "config.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return parse_config(raw)


def load_run_info(csv_path):
    """Read the hardware/software provenance written alongside a run."""
    info_path = csv_path.parent / "run_info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        return json.load(f)


def device_label(info):
    """Never guess. A run with no recorded device is reported as unknown."""
    if info is None:
        return "unrecorded"
    return info.get("device_name") or info.get("device_type") or "unrecorded"


def describe_change(cfg):
    if cfg is None:
        return ""

    run_name = cfg["run_name"]
    if run_name == "smoke_test":
        return "debug run on CPU with tiny model"
    if run_name == "baseline":
        return "reference hyperparameters"

    parts = []

    if cfg["batch_size"] != BASELINE["batch_size"]:
        parts.append(f"batch size {cfg['batch_size']}")

    if cfg["grad_accum_steps"] != BASELINE["grad_accum_steps"]:
        parts.append(f"grad accum x{cfg['grad_accum_steps']}")

    if cfg["learning_rate"] != BASELINE["learning_rate"]:
        parts.append(f"lr {cfg['learning_rate']:.1e}")

    if cfg["use_flash_attention"] != BASELINE["use_flash_attention"]:
        parts.append("flash attention")

    if cfg["mixed_precision"] != BASELINE["mixed_precision"]:
        parts.append(f"{cfg['dtype']} mixed precision")

    if cfg["torch_compile"] != BASELINE["torch_compile"]:
        parts.append("torch.compile")

    return ", ".join(parts) if parts else "config variant"


def precision_label(cfg, info=None):
    """Report the precision the run actually used, not the one it requested.

    When run_info.json records that autocast was requested but inactive, say so
    in the table. A row reading plain "bf16" for a run that executed in fp32 is
    how the mp_bf16 result got published as a null finding.
    """
    if cfg is None:
        return ""

    requested = cfg["mixed_precision"]
    active = info.get("mixed_precision_active") if info else None

    if active is False and requested:
        return f"fp32 ({cfg['dtype']} requested, INACTIVE)"
    if active or (active is None and requested):
        return cfg["dtype"]
    return "fp32"


def compile_label(cfg, info=None):
    if cfg is None:
        return ""

    requested = cfg["torch_compile"]
    active = info.get("torch_compile_active") if info else None

    if active is False and requested:
        return "no (requested, INACTIVE)"
    if active or (active is None and requested):
        return "yes"
    return "no"


def summarize_run(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={k: v for k, v in LEGACY_COLUMNS.items() if k in df.columns})

    numeric_cols = [
        "step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "step_time",
        "tokens_per_sec",
        "allocated_gb",
        "reserved_gb",
        "max_allocated_gb",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ignore step 0 for speed averages because warmup can distort results
    steady_df = df[df["step"] > 0] if "step" in df.columns else df
    final_row = df.iloc[-1]
    cfg = load_run_config(csv_path)
    info = load_run_info(csv_path)

    run_name = cfg["run_name"] if cfg else csv_path.parent.name

    # Validation loss at the last step is not the same thing as the best the run
    # ever reached. On this corpus the model overfits well before the final
    # step, so reporting only the final value ranks runs by how far past their
    # optimum they went.
    best_idx = df["val_loss"].idxmin()

    return {
        "run_name": run_name,
        "Run Name": run_name.replace("_", " ").title(),
        "Change": describe_change(cfg),
        "Device": device_label(info),
        "Batch": cfg["batch_size"] if cfg else "",
        "Grad Accum": cfg["grad_accum_steps"] if cfg else "",
        "Eff Batch": cfg["effective_batch_size"] if cfg else "",
        "Precision": precision_label(cfg, info),
        "Compile": compile_label(cfg, info),
        "Tokens/sec": format_tokens(steady_df["tokens_per_sec"].mean()),
        "Step Time (s)": round(steady_df["step_time"].mean(), 4),
        "Max Mem (GB)": round(df["max_allocated_gb"].max(), 3)
        if "max_allocated_gb" in df.columns
        else "",
        "Final Val Loss": round(final_row["val_loss"], 4),
        "Best Val Loss": round(df.loc[best_idx, "val_loss"], 4),
        "Best @ Step": int(df.loc[best_idx, "step"]),
        "category": CATEGORY_MAP.get(run_name, "Other"),
        "_sort_key": RUN_ORDER.get(run_name, 99),
        "_tokens_per_sec_raw": steady_df["tokens_per_sec"].mean(),
        "_last_step": int(final_row["step"]),
        # None means "the run predates provenance logging", which is not the
        # same as False and must not be reported as though it were.
        "_mixed_precision_active": (info or {}).get("mixed_precision_active"),
        "_mixed_precision_status": (info or {}).get("mixed_precision_status"),
        "_torch_compile_active": (info or {}).get("torch_compile_active"),
    }


def build_findings(rows):
    by_name = {row["run_name"]: row for row in rows}
    baseline = by_name.get("baseline")
    sections = []

    def pct_delta(current, reference, higher_is_better=False):
        if not current or not reference or pd.isna(current) or pd.isna(reference):
            return None
        delta = (current - reference) / reference * 100
        if not higher_is_better:
            delta = -delta
        return delta

    if baseline:
        # Memory is only instrumented on CUDA; say so rather than printing "** GB**".
        mem = baseline["Max Mem (GB)"]
        mem_line = (
            f"- Memory: **{mem} GB** peak allocated.\n"
            if mem not in ("", None) and not pd.isna(mem)
            else "- Memory: not recorded (only instrumented on CUDA).\n"
        )
        sections.append(
            "### Baseline\n"
            f"- Device: **{baseline['Device']}**.\n"
            f"- Throughput: **{baseline['Tokens/sec']} tokens/sec** at batch 64, fp32, manual attention.\n"
            f"{mem_line}"
            f"- Validation loss: **{baseline['Best Val Loss']}** (best, at step "
            f"{baseline['Best @ Step']}). Final step: {baseline['Final Val Loss']}."
            + (
                f"\n- Validation loss bottomed out at step {baseline['Best @ Step']} of "
                f"{baseline['_last_step']} and rose afterwards, so the final-step figure "
                "measures the run past its own optimum. Best-val is the primary metric here."
                if baseline["Best @ Step"] < baseline["_last_step"]
                else ""
            )
        )

    batch_runs = [by_name[k] for k in ("batch32", "batch128", "batch256") if k in by_name]
    if batch_runs and baseline:
        best = min(batch_runs, key=lambda r: r["Best Val Loss"])
        fastest = max(batch_runs, key=lambda r: r["_tokens_per_sec_raw"])
        lines = [
            "### Batch size sweep",
            "- Larger batches increase memory roughly linearly but do not scale throughput "
            "proportionally. Doubling the batch should roughly double tokens/sec if the GPU "
            "were compute- or bandwidth-bound; it does not, which points at fixed per-step "
            "overhead (kernel launches, Python dispatch, the optimizer step) dominating.",
        ]
        for run in batch_runs:
            mem_delta = pct_delta(run["Max Mem (GB)"], baseline["Max Mem (GB)"], higher_is_better=False)
            loss_delta = pct_delta(run["Best Val Loss"], baseline["Best Val Loss"], higher_is_better=False)
            mem_note = f"{mem_delta:+.0f}% memory vs baseline" if mem_delta is not None else "memory n/a"
            loss_note = f"{loss_delta:+.1f}% best val loss vs baseline" if loss_delta is not None else "loss n/a"
            lines.append(
                f"- **{run['Run Name']}**: {run['Tokens/sec']} tok/s, {run['Max Mem (GB)']} GB, "
                f"best val loss {run['Best Val Loss']} @ step {run['Best @ Step']} "
                f"({mem_note}, {loss_note})."
            )
        lines.append(
            f"- Best optimization quality in this sweep: **{best['Run Name']}** "
            f"(best val loss {best['Best Val Loss']}). "
            f"Fastest among batch variants: **{fastest['Run Name']}** ({fastest['Tokens/sec']} tok/s)."
        )

        # Ranking on the final step is only meaningful if the runs are still
        # improving there. Say so with the runs' own numbers rather than
        # asserting a conclusion the data may not support.
        overfit = [r for r in [baseline, *batch_runs] if r["Best @ Step"] < r["_last_step"]]
        if overfit:
            lines.append(
                "- **Caution: this ranking may be confounded.** "
                + ", ".join(
                    f"{r['Run Name']} peaked at step {r['Best @ Step']} of {r['_last_step']} "
                    f"({r['Best Val Loss']} -> {r['Final Val Loss']})"
                    for r in overfit
                )
                + ". Runs scored after their own optimum are being ranked by how fast they "
                "overfit, not by how well they generalize. Smaller batches are noisier and "
                "overfit a 1.1 MB corpus more slowly, which produces the same ordering as "
                "'smaller batches generalize better' and cannot be distinguished from it "
                "without early stopping or best-val scoring."
            )
        sections.append("\n".join(lines))

    grad_runs = [by_name[k] for k in ("grad_accum16x4", "grad_accum32x2") if k in by_name]
    if grad_runs and baseline:
        lines = [
            "### Gradient accumulation",
            "- Gradient accumulation simulates a larger effective batch while keeping per-step activation memory low.",
        ]
        for run in grad_runs:
            lines.append(
                f"- **{run['Run Name']}** (eff batch {run['Eff Batch']}): {run['Tokens/sec']} tok/s, "
                f"{run['Max Mem (GB)']} GB, best val loss {run['Best Val Loss']}."
            )
        best_grad = max(grad_runs, key=lambda r: r["_tokens_per_sec_raw"])
        lines.append(
            f"- **{best_grad['Run Name']}** is the better trade-off: near-baseline throughput and validation loss at roughly half the memory footprint."
        )
        sections.append("\n".join(lines))

    mp_run = by_name.get("mp_bf16")
    if mp_run and baseline:
        if mp_run["_mixed_precision_active"] is False:
            # Refuse to describe a fp32 run as a mixed-precision result. This is
            # the exact reporting failure that published "bf16 is free".
            sections.append(
                "### Mixed precision (BF16)\n"
                "- **This run did not use mixed precision.** Its `run_info.json` records "
                f"`mixed_precision_active: false` ({mp_run['_mixed_precision_status']}), "
                "so its numbers describe an fp32 run and say nothing about bf16.\n"
                "- No conclusion is drawn. Re-run on a device where autocast is active."
            )
        else:
            sections.append(
                "### Mixed precision (BF16)\n"
                f"- **BF16** measured {mp_run['Tokens/sec']} tok/s, "
                f"{mp_run['Max Mem (GB)']} GB, best val loss {mp_run['Best Val Loss']}, "
                f"against a baseline of {baseline['Tokens/sec']} tok/s, "
                f"{baseline['Max Mem (GB)']} GB, {baseline['Best Val Loss']}.\n"
                "- Sanity check before drawing a conclusion: bf16 halves activation bytes, "
                "so peak memory should differ from the fp32 baseline. If it does not, "
                "suspect the measurement rather than reporting a null result."
            )

    flash_run = by_name.get("flash_att")
    if flash_run and baseline:
        sections.append(
            "### Flash attention\n"
            f"- Flash attention yields a negligible throughput change "
            f"({flash_run['Tokens/sec']} vs {baseline['Tokens/sec']}).\n"
            "- These numbers were measured with the pre-fusion attention, which ran each "
            "head as a separate SDPA call on a (B, T, head_size) tensor. Fusing kernels "
            "cannot help when the bottleneck is the number of launches rather than the "
            "cost of each one, so this result says more about the old implementation than "
            "about flash attention."
        )

    compile_run = by_name.get("torch_compile")
    if compile_run and baseline:
        if compile_run["_torch_compile_active"] is False:
            sections.append(
                "### torch.compile\n"
                "- **This run never called `torch.compile`.** Its `run_info.json` records "
                "`torch_compile_active: false`, so its numbers describe an eager run.\n"
                "- No conclusion is drawn."
            )
        else:
            sections.append(
                "### torch.compile\n"
                f"- **torch.compile** measured {compile_run['Tokens/sec']} tok/s against "
                f"a baseline of {baseline['Tokens/sec']} tok/s.\n"
                "- Check that compilation warmup is excluded from the average before "
                "reading this: a compiled first step can cost thousands of times a "
                "steady-state step."
            )

    lr_runs = [by_name[k] for k in ("batch128_scaled_lr", "batch256_scaled_lr") if k in by_name]
    if lr_runs:
        lines = [
            "### LR scaling (linear rule)",
            "- These runs scale learning rate proportionally with batch size to recover optimization quality at larger batches.",
        ]
        for run in lr_runs:
            lines.append(
                f"- **{run['Run Name']}**: lr scaled to match batch, best val loss "
                f"{run['Best Val Loss']}, {run['Tokens/sec']} tok/s."
            )
        sections.append("\n".join(lines))
    elif any(k in by_name for k in ("batch128_scaled_lr", "batch256_scaled_lr")) is False:
        sections.append(
            "### LR scaling (linear rule)\n"
            "- Configs exist for batch 128/256 with linearly scaled learning rates but have not been run yet. "
            "Compare against the unscaled batch128/batch256 runs to see whether LR scaling closes the validation gap."
        )

    smoke = by_name.get("smoke_test")
    if smoke:
        sections.insert(
            0,
            "### Smoke test\n"
            f"- Quick sanity check on {smoke['Device']}: {smoke['Tokens/sec']} tok/s, "
            f"best val loss {smoke['Best Val Loss']}. "
            "Used to verify the training loop before GPU experiments.",
        )

    return "\n\n".join(sections)


DISPLAY_COLS = [
    "Run Name",
    "Change",
    "Device",
    "Batch",
    "Grad Accum",
    "Eff Batch",
    "Precision",
    "Compile",
    "Tokens/sec",
    "Step Time (s)",
    "Max Mem (GB)",
    "Best Val Loss",
    "Best @ Step",
    "Final Val Loss",
]


def build_report(runs):
    rows = [summarize_run(run) for run in runs]
    rows.sort(key=lambda r: (r["_sort_key"], r["run_name"]))

    summary_df = pd.DataFrame(rows)[DISPLAY_COLS]
    return "\n".join(
        [
            "## Experiment Summary",
            "",
            dataframe_to_markdown(summary_df),
            "",
            "## Findings",
            "",
            build_findings(rows),
            "",
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate run logs into a markdown summary."
    )
    parser.add_argument(
        "--runs",
        default=["outputs/*/train_log.csv", "results/*/train_log.csv"],
        nargs="+",
        help="glob(s) matching train_log.csv files (default: outputs/ and results/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the report to this file (default: print only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow --out to overwrite an existing file",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    runs = sorted({path for pattern in args.runs for path in glob.glob(pattern)})

    if not runs:
        print(
            f"No runs found matching {args.runs}. "
            "Run experiments first with train.py or run_experiments.sh."
        )
        return

    report = build_report(runs)
    print(report)

    if args.out is None:
        return

    # Writing used to be unconditional, straight over experiment_summary.md.
    # That file is the only surviving record of nine GPU runs whose CSVs are
    # gone, so a single `python summarize.py` on a machine with one smoke test
    # in outputs/ would have destroyed it.
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(
            f"\nRefusing to overwrite existing {out_path} ({len(runs)} run(s) found). "
            "Pass --force if that is what you want."
        )
        return

    out_path.write_text(report)
    print(f"\nWrote {out_path} from {len(runs)} run(s).")


if __name__ == "__main__":
    main()
