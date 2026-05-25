import yaml


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_config(cfg):
    batch_size = int(cfg["batch_size"])
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))

    return {
        "run_name": str(cfg["run_name"]),
        "learning_rate": float(cfg["learning_rate"]),
        "batch_size": batch_size,
        "block_size": int(cfg["block_size"]),
        "n_embd": int(cfg["n_embd"]),
        "n_layers": int(cfg["n_layers"]),
        "n_heads": int(cfg["n_heads"]),
        "dropout": float(cfg["dropout"]),
        "eval_interval": int(cfg["eval_interval"]),
        "eval_iters": int(cfg["eval_iters"]),
        "max_iters": int(cfg["max_iters"]),
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "use_flash_attention": bool(cfg.get("use_flash_attention", True)),
        "mixed_precision": bool(cfg.get("mixed_precision", False)),
        "dtype": str(cfg.get("dtype", "bf16")),
        "torch_compile": bool(cfg.get("torch_compile", False)),
        "min_lr": float(cfg.get("min_lr", 0)),
        "warmup_steps": int(cfg.get("warmup_steps", 0)),
        "use_lr_scheduler": bool(cfg.get("use_lr_scheduler", False)),
    }
