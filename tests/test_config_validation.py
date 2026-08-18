"""Tests for strict config validation.

The regression these exist for: `parse_config` used to silently drop any key it
did not recognise, and `train.py` read the parsed dict with
`cfg.get("mixed_precision", False)`. A config key that had not been threaded
through therefore read as False, and the mp_bf16 and torch_compile experiments
ran as unmodified baseline repeats without anything raising or warning.
"""

import pytest
import yaml

from utils import SCHEMA, ConfigError, load_config, parse_config

MINIMAL = {
    "run_name": "unit",
    "learning_rate": 3e-4,
    "batch_size": 4,
    "block_size": 16,
    "n_embd": 32,
    "n_layers": 2,
    "n_heads": 4,
    "dropout": 0.0,
    "eval_interval": 5,
    "eval_iters": 2,
    "max_iters": 10,
}


def cfg(**overrides):
    return {**MINIMAL, **overrides}


# --- the core regression ----------------------------------------------------


def test_unknown_key_raises():
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(mixed_precission=True))

    message = str(exc.value)
    assert "mixed_precission" in message
    assert "unknown config key" in message


def test_unknown_key_error_lists_valid_keys():
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(totally_made_up=1))

    message = str(exc.value)
    assert "mixed_precision" in message and "torch_compile" in message


def test_unknown_key_error_suggests_the_intended_key():
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(torch_compil=True))
    assert "did you mean 'torch_compile'?" in str(exc.value)


def test_all_unknown_keys_reported_at_once():
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(bogus_one=1, bogus_two=2))
    message = str(exc.value)
    assert "bogus_one" in message and "bogus_two" in message


@pytest.mark.parametrize(
    "key,value",
    [
        ("mixed_precision", True),
        ("torch_compile", True),
        ("dtype", "fp16"),
        ("use_flash_attention", False),
        ("grad_accum_steps", 4),
        ("norm_first", True),
        ("seed", 99),
        ("deterministic", True),
        ("timing_warmup_steps", 12),
        ("device", "cpu"),
        ("use_lr_scheduler", True),
    ],
)
def test_known_key_reaches_the_parsed_output(key, value):
    """The other half of the bug: a key must actually arrive, not just parse.

    Silently dropping any of these is precisely what produced three identical
    rows in the results table.
    """
    parsed = parse_config(cfg(**{key: value}))
    assert parsed[key] == value


def test_every_schema_key_is_present_in_output():
    """No option can be declared and then not emitted."""
    parsed = parse_config(cfg())
    for name in SCHEMA:
        assert name in parsed, f"{name} is in SCHEMA but missing from parse_config output"


def test_defaults_match_the_historical_behaviour():
    """Defaults must not silently change what an existing config means."""
    parsed = parse_config(cfg())
    assert parsed["mixed_precision"] is False
    assert parsed["torch_compile"] is False
    assert parsed["use_flash_attention"] is True
    assert parsed["grad_accum_steps"] == 1
    assert parsed["norm_first"] is False  # post-norm, as every recorded run used
    assert parsed["seed"] == 1337
    assert parsed["device"] == "auto"


# --- required keys and types ------------------------------------------------


@pytest.mark.parametrize("missing", sorted(MINIMAL))
def test_missing_required_key_raises(missing):
    incomplete = {k: v for k, v in MINIMAL.items() if k != missing}
    with pytest.raises(ConfigError) as exc:
        parse_config(incomplete)
    assert missing in str(exc.value)


def test_quoted_boolean_is_rejected_rather_than_coerced():
    """bool("false") is True, so this would otherwise invert the setting."""
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(mixed_precision="false"))
    assert "boolean" in str(exc.value)


def test_non_integral_float_is_rejected_rather_than_truncated():
    with pytest.raises(ConfigError):
        parse_config(cfg(warmup_steps=0.5))


def test_integral_float_is_accepted():
    assert parse_config(cfg(max_iters=5e3))["max_iters"] == 5000


def test_scientific_notation_string_parses():
    """PyYAML reads unquoted 3e-4 as a string, so this path matters."""
    assert parse_config(cfg(learning_rate="3e-4"))["learning_rate"] == pytest.approx(3e-4)


def test_bad_dtype_is_rejected_not_defaulted():
    """dtype_map.get(..., torch.bfloat16) would have silently accepted this."""
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(dtype="fp8"))
    assert "fp8" in str(exc.value)


def test_bad_device_is_rejected():
    with pytest.raises(ConfigError):
        parse_config(cfg(device="tpu"))


# --- semantic consistency ---------------------------------------------------


def test_n_embd_must_divide_by_n_heads():
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(n_embd=30, n_heads=4))
    assert "divisible" in str(exc.value)


@pytest.mark.parametrize("key", ["batch_size", "block_size", "n_layers", "max_iters"])
def test_non_positive_values_rejected(key):
    with pytest.raises(ConfigError):
        parse_config(cfg(**{key: 0}))


def test_dropout_out_of_range_rejected():
    with pytest.raises(ConfigError):
        parse_config(cfg(dropout=1.0))


def test_warmup_longer_than_training_rejected():
    with pytest.raises(ConfigError) as exc:
        parse_config(cfg(use_lr_scheduler=True, warmup_steps=100, max_iters=10))
    assert "warmup_steps" in str(exc.value)


def test_min_lr_above_max_lr_rejected():
    with pytest.raises(ConfigError):
        parse_config(cfg(use_lr_scheduler=True, learning_rate=1e-4, min_lr=1e-3))


def test_effective_batch_size_is_derived():
    parsed = parse_config(cfg(batch_size=8, grad_accum_steps=4))
    assert parsed["effective_batch_size"] == 32


def test_effective_batch_size_cannot_be_set_by_config():
    """It is derived; accepting it would let a config contradict itself."""
    with pytest.raises(ConfigError):
        parse_config(cfg(effective_batch_size=999))


# --- the shipped configs ----------------------------------------------------


def test_every_repo_config_validates(repo_config_path):
    """Every config in configs/ must survive strict validation.

    Parameterised over the real files, so adding a config with a typo fails CI
    instead of quietly running something other than the experiment named.
    """
    parse_config(load_config(str(repo_config_path)))


def test_config_files_are_mappings(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(ConfigError):
        load_config(str(bad))


def test_empty_config_file_raises(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ConfigError):
        load_config(str(empty))


def test_historical_run_configs_still_parse(repo_result_config_path):
    """Committed run artifacts must stay readable by summarize.py."""
    raw = yaml.safe_load(repo_result_config_path.read_text())
    parse_config(raw)
