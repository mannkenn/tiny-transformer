"""Tests for the reporting layer.

The reporting layer is where the false claims actually got published. The
config bug made mp_bf16 run in fp32; summarize.py is what then labelled that row
"bf16" and wrote "on supported GPUs this is effectively a free optimization".
Both halves needed fixing, so both halves are tested.
"""

import json

import pandas as pd
import pytest
import yaml

import summarize

BASE_CONFIG = {
    "run_name": "baseline",
    "learning_rate": 3e-4,
    "batch_size": 64,
    "block_size": 256,
    "n_embd": 384,
    "n_layers": 6,
    "n_heads": 6,
    "dropout": 0.1,
    "eval_interval": 500,
    "eval_iters": 200,
    "max_iters": 5000,
}

# A run that improves, bottoms out, then overfits -- the shape of the real
# baseline CSV.
OVERFITTING_LOG = """step,train_loss,val_loss,learning_rate,step_time,tokens_per_sec
0,3.7405,3.7688,0.0003,0.4215,38870.5490
500,1.7939,1.9217,0.0003,0.0610,287523.7777
1000,1.4477,1.6383,0.0003,0.0556,287052.1923
2500,1.1517,1.4986,0.0003,0.0597,285859.5444
4999,0.8362,1.6111,0.0003,0.0531,287478.6120
"""


def make_run(directory, name, log=OVERFITTING_LOG, config=None, info=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train_log.csv").write_text(log)
    cfg = {**BASE_CONFIG, "run_name": name, **(config or {})}
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
    if info is not None:
        (directory / "run_info.json").write_text(json.dumps(info))
    return directory / "train_log.csv"


# --- reporting what actually happened ---------------------------------------


def test_inactive_mixed_precision_row_is_labelled_inactive(tmp_path):
    """A fp32 run must not be labelled 'bf16' just because the config asked."""
    path = make_run(
        tmp_path / "mp_bf16",
        "mp_bf16",
        config={"mixed_precision": True, "dtype": "bf16"},
        info={"mixed_precision_active": False, "device_name": "test-gpu"},
    )
    row = summarize.summarize_run(path)
    assert row["Precision"] == "fp32 (bf16 requested, INACTIVE)"


def test_inactive_compile_row_is_labelled_inactive(tmp_path):
    path = make_run(
        tmp_path / "torch_compile",
        "torch_compile",
        config={"torch_compile": True},
        info={"torch_compile_active": False, "device_name": "test-gpu"},
    )
    row = summarize.summarize_run(path)
    assert "INACTIVE" in row["Compile"]


def test_active_mixed_precision_row_is_labelled_normally(tmp_path):
    path = make_run(
        tmp_path / "mp_bf16",
        "mp_bf16",
        config={"mixed_precision": True, "dtype": "bf16"},
        info={"mixed_precision_active": True, "device_name": "test-gpu"},
    )
    assert summarize.summarize_run(path)["Precision"] == "bf16"


def test_findings_refuse_to_conclude_from_an_inactive_run(tmp_path):
    """The published claim was drawn from a run that never used bf16."""
    rows = [
        summarize.summarize_run(make_run(tmp_path / "baseline", "baseline")),
        summarize.summarize_run(
            make_run(
                tmp_path / "mp_bf16",
                "mp_bf16",
                config={"mixed_precision": True, "dtype": "bf16"},
                info={
                    "mixed_precision_active": False,
                    "mixed_precision_status": "config key was silently dropped",
                },
            )
        ),
    ]
    findings = summarize.build_findings(rows)

    assert "did not use mixed precision" in findings
    assert "No conclusion is drawn" in findings
    assert "free optimization" not in findings


def test_findings_refuse_to_conclude_from_an_uncompiled_run(tmp_path):
    rows = [
        summarize.summarize_run(make_run(tmp_path / "baseline", "baseline")),
        summarize.summarize_run(
            make_run(
                tmp_path / "torch_compile",
                "torch_compile",
                config={"torch_compile": True},
                info={"torch_compile_active": False},
            )
        ),
    ]
    findings = summarize.build_findings(rows)

    assert "never called `torch.compile`" in findings
    assert "no measurable speedup" not in findings


def test_missing_run_info_is_unrecorded_not_assumed(tmp_path):
    """Absent provenance must read as unknown, never as a guessed device."""
    path = make_run(tmp_path / "baseline", "baseline", info=None)
    row = summarize.summarize_run(path)
    assert row["Device"] == "unrecorded"
    # A run predating provenance logging is unknown, not proven-inactive.
    assert row["_mixed_precision_active"] is None
    assert row["Precision"] == "fp32"


# --- scoring on best val loss ------------------------------------------------


def test_best_val_loss_is_reported_with_its_step(tmp_path):
    path = make_run(tmp_path / "baseline", "baseline")
    row = summarize.summarize_run(path)

    assert row["Best Val Loss"] == pytest.approx(1.4986)
    assert row["Best @ Step"] == 2500
    assert row["Final Val Loss"] == pytest.approx(1.6111)


def test_best_val_loss_column_precedes_final(tmp_path):
    """Best-val is the primary convergence metric; final is secondary."""
    cols = summarize.DISPLAY_COLS
    assert cols.index("Best Val Loss") < cols.index("Final Val Loss")


def test_overfitting_run_triggers_an_explicit_warning(tmp_path):
    rows = [
        summarize.summarize_run(make_run(tmp_path / "baseline", "baseline")),
        summarize.summarize_run(make_run(tmp_path / "batch32", "batch32", config={"batch_size": 32})),
    ]
    findings = summarize.build_findings(rows)

    assert "may be confounded" in findings
    assert "overfit" in findings


def test_no_overfitting_warning_when_the_best_step_is_the_last(tmp_path):
    """Do not cry wolf on a run that never turned over."""
    monotonic = """step,train_loss,val_loss,learning_rate,step_time,tokens_per_sec
0,3.7,3.8,0.0003,0.4,38870.0
500,1.8,1.9,0.0003,0.06,287523.0
4999,0.9,1.2,0.0003,0.05,287478.0
"""
    rows = [
        summarize.summarize_run(make_run(tmp_path / "baseline", "baseline", log=monotonic)),
        summarize.summarize_run(
            make_run(tmp_path / "batch32", "batch32", log=monotonic, config={"batch_size": 32})
        ),
    ]
    findings = summarize.build_findings(rows)
    assert "may be confounded" not in findings


# --- legacy compatibility ----------------------------------------------------


def test_legacy_column_names_are_understood(tmp_path):
    """The only surviving pre-refactor CSV uses lr / elapsed_time."""
    legacy = """step,train_loss,val_loss,lr,elapsed_time,tokens_per_sec
0,3.7405,3.7688,0.0003,0.4215,38870.5490
2500,1.1517,1.4986,0.0003,0.0597,285859.5444
4999,0.8362,1.6111,0.0003,0.0531,287478.6120
"""
    path = make_run(tmp_path / "baseline", "baseline", log=legacy)
    row = summarize.summarize_run(path)

    assert row["Best Val Loss"] == pytest.approx(1.4986)
    assert not pd.isna(row["Step Time (s)"])


def test_report_does_not_overwrite_without_force(tmp_path, capsys):
    make_run(tmp_path / "baseline", "baseline")
    target = tmp_path / "summary.md"
    target.write_text("PRECIOUS")

    summarize.main(["--runs", f"{tmp_path}/*/train_log.csv", "--out", str(target)])

    assert target.read_text() == "PRECIOUS"
    assert "Refusing to overwrite" in capsys.readouterr().out


def test_report_writes_with_force(tmp_path):
    make_run(tmp_path / "baseline", "baseline")
    target = tmp_path / "summary.md"
    target.write_text("PRECIOUS")

    summarize.main(["--runs", f"{tmp_path}/*/train_log.csv", "--out", str(target), "--force"])

    assert "Experiment Summary" in target.read_text()
