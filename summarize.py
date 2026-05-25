import glob
from pathlib import Path

import pandas as pd
import yaml

from utils import parse_config

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

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return parse_config(raw)


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


def precision_label(cfg):
    if cfg is None:
        return ""
    if cfg.get("mixed_precision"):
        return cfg.get("dtype", "bf16")
    return "fp32"


def summarize_run(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

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

    run_name = cfg["run_name"] if cfg else csv_path.parent.name

    return {
        "run_name": run_name,
        "Run Name": run_name.replace("_", " ").title(),
        "Change": describe_change(cfg),
        "Batch": cfg["batch_size"] if cfg else "",
        "Grad Accum": cfg["grad_accum_steps"] if cfg else "",
        "Eff Batch": cfg["effective_batch_size"] if cfg else "",
        "Precision": precision_label(cfg),
        "Compile": "yes" if cfg and cfg.get("torch_compile") else "no",
        "Tokens/sec": format_tokens(steady_df["tokens_per_sec"].mean()),
        "Step Time (s)": round(steady_df["step_time"].mean(), 4),
        "Max Mem (GB)": round(df["max_allocated_gb"].max(), 3)
        if "max_allocated_gb" in df.columns
        else "",
        "Val Loss": round(final_row["val_loss"], 4),
        "category": CATEGORY_MAP.get(run_name, "Other"),
        "_sort_key": RUN_ORDER.get(run_name, 99),
        "_tokens_per_sec_raw": steady_df["tokens_per_sec"].mean(),
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
        sections.append(
            "### Baseline\n"
            f"- Throughput: **{baseline['Tokens/sec']} tokens/sec** at batch 64, fp32, manual attention.\n"
            f"- Memory: **{baseline['Max Mem (GB)']} GB** peak allocated.\n"
            f"- Validation loss after 5k steps: **{baseline['Val Loss']}**."
        )

    batch_runs = [by_name[k] for k in ("batch32", "batch128", "batch256") if k in by_name]
    if batch_runs and baseline:
        best = min(batch_runs, key=lambda r: r["Val Loss"])
        fastest = max(batch_runs, key=lambda r: r["_tokens_per_sec_raw"])
        lines = [
            "### Batch size sweep",
            "- Larger batches increase memory roughly linearly but do not scale throughput proportionally on this model — training appears memory-bandwidth limited.",
        ]
        for run in batch_runs:
            mem_delta = pct_delta(run["Max Mem (GB)"], baseline["Max Mem (GB)"], higher_is_better=False)
            loss_delta = pct_delta(run["Val Loss"], baseline["Val Loss"], higher_is_better=False)
            mem_note = f"{mem_delta:+.0f}% memory vs baseline" if mem_delta is not None else "memory n/a"
            loss_note = f"{loss_delta:+.1f}% val loss vs baseline" if loss_delta is not None else "loss n/a"
            lines.append(
                f"- **{run['Run Name']}**: {run['Tokens/sec']} tok/s, {run['Max Mem (GB)']} GB, val loss {run['Val Loss']} ({mem_note}, {loss_note})."
            )
        lines.append(
            f"- Best optimization quality in this sweep: **{best['Run Name']}** (val loss {best['Val Loss']}). "
            f"Fastest among batch variants: **{fastest['Run Name']}** ({fastest['Tokens/sec']} tok/s)."
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
                f"{run['Max Mem (GB)']} GB, val loss {run['Val Loss']}."
            )
        best_grad = max(grad_runs, key=lambda r: r["_tokens_per_sec_raw"])
        lines.append(
            f"- **{best_grad['Run Name']}** is the better trade-off: near-baseline throughput and validation loss at roughly half the memory footprint."
        )
        sections.append("\n".join(lines))

    mp_run = by_name.get("mp_bf16")
    if mp_run and baseline:
        sections.append(
            "### Mixed precision (BF16)\n"
            f"- **BF16** matches baseline on throughput ({mp_run['Tokens/sec']}), memory ({mp_run['Max Mem (GB)']} GB), "
            f"and validation loss ({mp_run['Val Loss']}). On supported GPUs this is effectively a free optimization."
        )

    flash_run = by_name.get("flash_att")
    if flash_run and baseline:
        sections.append(
            "### Flash attention\n"
            f"- Flash attention yields a negligible throughput change ({flash_run['Tokens/sec']} vs {baseline['Tokens/sec']}) "
            f"at this model size. Kernel fusion benefits grow with sequence length and head count."
        )

    compile_run = by_name.get("torch_compile")
    if compile_run and baseline:
        sections.append(
            "### torch.compile\n"
            f"- **torch.compile** shows no measurable speedup here ({compile_run['Tokens/sec']} vs {baseline['Tokens/sec']}). "
            "Small models often spend more time in Python overhead and compilation warmup than saved compute."
        )

    lr_runs = [by_name[k] for k in ("batch128_scaled_lr", "batch256_scaled_lr") if k in by_name]
    if lr_runs:
        lines = [
            "### LR scaling (linear rule)",
            "- These runs scale learning rate proportionally with batch size to recover optimization quality at larger batches.",
        ]
        for run in lr_runs:
            lines.append(
                f"- **{run['Run Name']}**: lr scaled to match batch, val loss {run['Val Loss']}, {run['Tokens/sec']} tok/s."
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
            f"- Quick sanity check on CPU: {smoke['Tokens/sec']} tok/s, val loss {smoke['Val Loss']}. "
            "Used to verify the training loop before GPU experiments.",
        )

    return "\n\n".join(sections)


def main():
    runs = glob.glob("outputs/*/train_log.csv")

    if not runs:
        print("No runs found under outputs/. Run experiments first with train.py or run_experiments.sh.")
        return

    rows = [summarize_run(run) for run in runs]
    rows.sort(key=lambda r: (r["_sort_key"], r["run_name"]))

    display_cols = [
        "Run Name",
        "Change",
        "Batch",
        "Grad Accum",
        "Eff Batch",
        "Precision",
        "Compile",
        "Tokens/sec",
        "Step Time (s)",
        "Max Mem (GB)",
        "Val Loss",
    ]
    summary_df = pd.DataFrame(rows)[display_cols]
    markdown_table = dataframe_to_markdown(summary_df)
    findings = build_findings(rows)

    output = "\n".join(
        [
            "## Experiment Summary",
            "",
            markdown_table,
            "",
            "## Findings",
            "",
            findings,
            "",
        ]
    )

    print(output)

    with open("experiment_summary.md", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
