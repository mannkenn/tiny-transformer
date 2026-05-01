import yaml


def load_config(path: str):
	with open(path, "r") as f:
		cfg = yaml.safe_load(f)
	return cfg


def parse_config(cfg):
	# cast config values to correct types
	return {
		"run_name": str(cfg["run_name"]),
		"learning_rate": float(cfg["learning_rate"]),
		"batch_size": int(cfg["batch_size"]),
		"block_size": int(cfg["block_size"]),
		"n_embd": int(cfg["n_embd"]),
		"n_layers": int(cfg["n_layers"]),
		"n_heads": int(cfg["n_heads"]),
		"dropout": float(cfg["dropout"]),
		"eval_interval": int(cfg["eval_interval"]),
		"eval_iters": int(cfg["eval_iters"]),
		"max_iters": int(cfg["max_iters"]),
		"grad_accum_steps": int(cfg.get("grad_accum_steps", 1)),
		"use_flash_attention": bool(cfg.get("use_flash_attention", True)),
	}
