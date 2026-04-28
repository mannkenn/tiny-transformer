import glob
from pathlib import Path
import pandas as pd


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


def summarize_run(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Convert numeric columns
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

    run_name = csv_path.parent.name

    summary = {
        "Run Name": run_name,
        "Change": "",
        "Batch": "",
        "Grad Accum": "",
        "Eff Batch": "",
        "Precision": "",
        "Compile": "",
        "Tokens/sec": format_tokens(steady_df["tokens_per_sec"].mean()),
        "Step Time (s)": round(steady_df["step_time"].mean(), 4),
        "Max Mem (GB)": round(df["max_allocated_gb"].max(), 3)
        if "max_allocated_gb" in df.columns
        else "",
        "Val Loss": round(final_row["val_loss"], 4),
    }

    return summary

def format_tokens(x):
    if pd.isna(x):
        return ""

    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"

    if x >= 1_000:
        return f"{x / 1_000:.0f}k"

    return str(round(x, 0))


runs = glob.glob("outputs/*/train_log.csv")

rows = [summarize_run(run) for run in runs]

summary_df = pd.DataFrame(rows)

# Save markdown table
markdown_table = dataframe_to_markdown(summary_df)

print(markdown_table)

with open("experiment_summary.md", "w") as f:
    f.write("## Experiment Summary\n\n")
    f.write(markdown_table)
    f.write("\n")