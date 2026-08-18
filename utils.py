"""Config loading and strict validation.

The reason this file is strict
------------------------------

`parse_config` used to build its output by reaching into the raw YAML with
`cfg.get(name, default)` for each key it knew about, and silently dropping
anything else. `train.py` then read the *parsed* dict:

    use_amp = cfg.get("mixed_precision", False) and torch.cuda.is_available()
    if cfg.get("torch_compile", False):
        model = torch.compile(model)

So a config key that `parse_config` had not been taught about did not raise, did
not warn, and did not take effect. It read as `False`.

That is exactly what happened. At commit 9edd69a -- the commit that generated the
results table -- `parse_config` returned no `mixed_precision`, `dtype` or
`torch_compile` key, while `configs/mp/mp_bf16.yaml` and
`configs/compile/torch_compile.yaml` had been setting them since c15e431. Those
two experiments ran as byte-identical repeats of the baseline config: same seed,
same data order, autocast disabled, `torch.compile` never called. The three
matching rows in the results table are one run reported three times.

The fix is structural rather than a patch. SCHEMA below is the single source of
truth: a key exists for the loop that parses it, so it cannot be declared
without being threaded through. Anything in the YAML that is not in SCHEMA is an
error, not a silent no-op.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config is unparseable, unknown or internally inconsistent."""


_REQUIRED = object()


def _strict_bool(value):
    # bool("false") is True, so a quoted YAML boolean would silently invert.
    if isinstance(value, bool):
        return value
    raise ConfigError(f"expected a boolean (true/false), got {value!r}")


def _strict_int(value):
    if isinstance(value, bool):
        raise ConfigError(f"expected an integer, got boolean {value!r}")
    if isinstance(value, int):
        return value
    # Allow 5e3 from YAML, reject 0.5 rather than truncating it to 0.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ConfigError(f"expected an integer, got {value!r}")


def _strict_float(value):
    if isinstance(value, bool):
        raise ConfigError(f"expected a number, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # PyYAML reads unquoted 3e-4 as a string, so this path is load-bearing.
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"expected a number, got {value!r}") from None
    raise ConfigError(f"expected a number, got {value!r}")


@dataclass(frozen=True)
class Option:
    cast: Callable[[Any], Any]
    default: Any = _REQUIRED
    choices: tuple = ()
    doc: str = ""

    @property
    def required(self):
        return self.default is _REQUIRED


SCHEMA = {
    "run_name": Option(str, doc="label for the run, used in the output directory"),
    "learning_rate": Option(_strict_float, doc="peak learning rate"),
    "batch_size": Option(_strict_int, doc="micro-batch size"),
    "block_size": Option(_strict_int, doc="context length in tokens"),
    "n_embd": Option(_strict_int, doc="embedding dimension"),
    "n_layers": Option(_strict_int, doc="number of decoder blocks"),
    "n_heads": Option(_strict_int, doc="attention heads per block"),
    "dropout": Option(_strict_float, doc="dropout probability"),
    "eval_interval": Option(_strict_int, doc="steps between evaluations"),
    "eval_iters": Option(_strict_int, doc="batches averaged per evaluation"),
    "max_iters": Option(_strict_int, doc="total optimizer steps"),
    "grad_accum_steps": Option(_strict_int, 1, doc="micro-batches per optimizer step"),
    "use_flash_attention": Option(
        _strict_bool, True, doc="use scaled_dot_product_attention instead of explicit matmuls"
    ),
    "mixed_precision": Option(_strict_bool, False, doc="enable autocast (CUDA only)"),
    "dtype": Option(str, "bf16", choices=("bf16", "fp16"), doc="autocast dtype"),
    "torch_compile": Option(_strict_bool, False, doc="wrap the model in torch.compile"),
    "min_lr": Option(_strict_float, 0.0, doc="floor for the cosine schedule"),
    "warmup_steps": Option(_strict_int, 0, doc="linear warmup steps for the LR schedule"),
    "use_lr_scheduler": Option(_strict_bool, False, doc="enable warmup + cosine decay"),
    "norm_first": Option(
        _strict_bool, False, doc="pre-norm (GPT-2 style) instead of post-norm"
    ),
    "seed": Option(_strict_int, 1337, doc="seed for torch, numpy and random"),
    "deterministic": Option(_strict_bool, False, doc="force deterministic kernels"),
    "timing_warmup_steps": Option(
        _strict_int, 5, doc="steps excluded from the rolling throughput average"
    ),
    "device": Option(
        str, "auto", choices=("auto", "cpu", "cuda", "mps"), doc="device override"
    ),
}

# Keys that parse_config computes rather than reads. Listed so callers can tell
# derived values apart from things a config is allowed to set.
DERIVED_KEYS = ("effective_batch_size",)


def load_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} must contain a mapping, got {type(cfg).__name__}")
    return cfg


def _suggest(unknown_key):
    """Point at the intended key when someone misspells one."""
    import difflib

    matches = difflib.get_close_matches(unknown_key, SCHEMA, n=1, cutoff=0.7)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def _validate_semantics(parsed):
    """Cross-field checks that a per-key cast cannot catch."""
    positive = (
        "batch_size",
        "block_size",
        "n_embd",
        "n_layers",
        "n_heads",
        "eval_interval",
        "eval_iters",
        "max_iters",
        "grad_accum_steps",
    )
    for key in positive:
        if parsed[key] <= 0:
            raise ConfigError(f"{key} must be positive, got {parsed[key]}")

    if parsed["n_embd"] % parsed["n_heads"] != 0:
        raise ConfigError(
            f"n_embd ({parsed['n_embd']}) must be divisible by n_heads "
            f"({parsed['n_heads']}); head_size would be "
            f"{parsed['n_embd'] / parsed['n_heads']}"
        )

    if not 0.0 <= parsed["dropout"] < 1.0:
        raise ConfigError(f"dropout must be in [0, 1), got {parsed['dropout']}")

    if parsed["learning_rate"] <= 0:
        raise ConfigError(f"learning_rate must be positive, got {parsed['learning_rate']}")

    if parsed["timing_warmup_steps"] < 0:
        raise ConfigError(
            f"timing_warmup_steps must be >= 0, got {parsed['timing_warmup_steps']}"
        )

    if parsed["use_lr_scheduler"]:
        if parsed["warmup_steps"] >= parsed["max_iters"]:
            raise ConfigError(
                f"warmup_steps ({parsed['warmup_steps']}) must be less than max_iters "
                f"({parsed['max_iters']}), or the schedule never leaves warmup"
            )
        if parsed["min_lr"] > parsed["learning_rate"]:
            raise ConfigError(
                f"min_lr ({parsed['min_lr']}) must not exceed learning_rate "
                f"({parsed['learning_rate']})"
            )


def parse_config(cfg):
    """Validate and type-cast a raw config mapping.

    Raises ConfigError on unknown keys, missing required keys, values of the
    wrong type, values outside an option's choices, and inconsistent
    combinations. It never silently drops or defaults a key the caller supplied.
    """
    if not isinstance(cfg, dict):
        raise ConfigError(f"config must be a mapping, got {type(cfg).__name__}")

    unknown = sorted(set(cfg) - set(SCHEMA))
    if unknown:
        details = ", ".join(f"{k!r}{_suggest(k)}" for k in unknown)
        raise ConfigError(
            f"unknown config key(s): {details}. "
            f"Valid keys are: {', '.join(sorted(SCHEMA))}. "
            "An unrecognised key is an error rather than a no-op because "
            "silently ignoring one is what made the mp_bf16 and torch_compile "
            "experiments rerun the baseline instead."
        )

    missing = sorted(k for k, opt in SCHEMA.items() if opt.required and k not in cfg)
    if missing:
        raise ConfigError(f"missing required config key(s): {', '.join(missing)}")

    parsed = {}
    for name, opt in SCHEMA.items():
        if name not in cfg:
            parsed[name] = opt.default
            continue
        try:
            value = opt.cast(cfg[name])
        except ConfigError as exc:
            raise ConfigError(f"config key {name!r}: {exc}") from None
        if opt.choices and value not in opt.choices:
            raise ConfigError(
                f"config key {name!r}: expected one of {opt.choices}, got {value!r}"
            )
        parsed[name] = value

    parsed["effective_batch_size"] = parsed["batch_size"] * parsed["grad_accum_steps"]

    _validate_semantics(parsed)
    return parsed
