import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch

from model import Transformer
from utils import load_config, parse_config

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_device(requested="auto"):
    """Map a config/CLI device string to a concrete torch.device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed, deterministic=False):
    """Seed every RNG the training loop can touch.

    torch.manual_seed alone leaves numpy and the stdlib `random` module
    unseeded, so anything that later reaches for them (data shuffling,
    augmentation, sampling) silently varies run to run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Trades throughput for reproducibility: picks deterministic kernels and
        # errors out if an op has no deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def synchronize(device):
    """Block until queued work on `device` has finished.

    CUDA and MPS kernel launches are asynchronous, so `time.time()` around a
    training step measures how long it took to *enqueue* the work, not to run
    it, unless the device is explicitly synchronised first.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def device_description(device):
    """Human-readable identity of the hardware a run actually executed on."""
    if device.type == "cuda":
        index = device.index or 0
        return torch.cuda.get_device_name(index)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def write_run_info(out_dir, cfg, device):
    """Record what hardware and software produced a run, next to its results.

    Benchmark numbers are meaningless without this. Any claim about throughput
    or memory should be traceable to the device string written here.
    """
    info = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "device_type": device.type,
        "device_name": device_description(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "git_commit": git_commit(),
        "resolved_config": cfg,
    }

    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    print(f"device: {info['device_name']} ({device.type}) | torch {info['torch_version']}")
    return info


def prepare_run_from_file(config_path, device_override=None):
    """Load a config and create the output directory for a single run."""
    raw = load_config(config_path)
    cfg = parse_config(raw)
    if device_override is not None:
        cfg["device"] = device_override
    print(cfg)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # include config filename to distinguish runs from the same run_name
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    out_dir = f"outputs/{cfg['run_name']}_{cfg_name}_{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    # save config used
    shutil.copy(config_path, f"{out_dir}/config.yaml")

    return cfg, out_dir, f"{out_dir}/train_log.csv"


def load_data(path="input.txt"):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data, len(chars)


def get_batch(split, train_data, val_data, cfg, device):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - cfg["block_size"], (cfg["batch_size"],))
    x = torch.stack([source[i : i + cfg["block_size"]] for i in ix])
    y = torch.stack([source[i + 1 : i + cfg["block_size"] + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        # Accumulate on-device and read back once. Calling .item() per iteration
        # forces eval_iters host-device syncs per split for no benefit.
        losses = torch.zeros(cfg["eval_iters"], device=device)
        for k in range(cfg["eval_iters"]):
            xb, yb = get_batch(split, train_data, val_data, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out


# logging
def log_headers(device):
    headers = [
        "step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "step_time",
        "tokens_per_sec",
    ]
    if device.type == "cuda":
        headers.extend(["allocated_gb", "reserved_gb", "max_allocated_gb"])
    return headers


def ensure_log_file(path, device):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(log_headers(device))


def append_log(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# model checkpointing
def unwrap_model(model):
    """Return the eager module behind a torch.compile wrapper.

    torch.compile returns an OptimizedModule whose state_dict keys are all
    prefixed with `_orig_mod.`. Saving that prefixed dict makes the checkpoint
    unloadable by an uncompiled model, so compiled runs could not be resumed
    without compilation.
    """
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path, model, optimizer, step, best_val_loss, cfg):
    checkpoint = {
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint["step"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    cfg = checkpoint.get("config", None)

    return step, best_val_loss, cfg


def get_lr(step, max_lr, min_lr, warmup_steps, max_iters):
    # linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2. after training ends, stay at min_lr
    if step >= max_iters:
        return min_lr

    # 3. cosine decay
    decay_ratio = (step - warmup_steps) / (max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def build_model(cfg, vocab_size, device):
    model = Transformer(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        use_flash_attention=cfg["use_flash_attention"],
        norm_first=cfg["norm_first"],
    ).to(device)

    if cfg.get("torch_compile", False):
        print("compiling model with torch.compile...")
        model = torch.compile(model)

    return model


def train(cfg, out_dir, log_path, resume=None, data_path="input.txt"):
    set_seed(cfg["seed"], cfg["deterministic"])

    device = resolve_device(cfg["device"])
    write_run_info(out_dir, cfg, device)

    # mixed precision: autocast is only wired up for cuda in this project
    use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"
    if cfg.get("mixed_precision", False) and not use_amp:
        print(f"warning: mixed_precision requested but device is {device.type}; running fp32")
    amp_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)

    train_data, val_data, vocab_size = load_data(data_path)
    model = build_model(cfg, vocab_size, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    best_val_loss = float("inf")

    start_step = 0
    if resume is not None:
        last_step, best_val_loss, _ = load_checkpoint(
            resume, model, optimizer, map_location=device
        )
        # The checkpoint records the last *completed* step, so training resumes
        # at the next one. Starting at last_step re-ran a step that had already
        # been applied to the weights.
        start_step = last_step + 1
        print(f"resumed from {resume} at step {start_step} (checkpoint ended at {last_step})")

    ensure_log_file(log_path, device)

    grad_accum_steps = cfg["grad_accum_steps"]
    print(f"effective batch size: {cfg['batch_size'] * grad_accum_steps}")

    step_times = deque(maxlen=20)
    step_tokens = deque(maxlen=20)
    warmup = cfg["timing_warmup_steps"]

    B, T = cfg["batch_size"], cfg["block_size"]
    tokens = B * T * grad_accum_steps  # tokens per optimizer step, incl. grad accum

    for step in range(start_step, cfg["max_iters"]):
        # lr scheduling
        if cfg["use_lr_scheduler"]:
            lr = get_lr(
                step,
                max_lr=cfg["learning_rate"],
                min_lr=cfg["min_lr"],
                warmup_steps=cfg["warmup_steps"],
                max_iters=cfg["max_iters"],
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = cfg["learning_rate"]

        # Accumulate the loss as a device tensor. Calling .item() inside the
        # micro-step loop forces a host-device sync every micro-step and is a
        # hard graph break for torch.compile, so Inductor cannot fuse across it.
        total_loss = torch.zeros((), device=device)

        synchronize(device)
        step_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        for _micro_step in range(grad_accum_steps):
            xb, yb = get_batch("train", train_data, val_data, cfg, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, loss = model(xb, yb)
                total_loss += loss.detach()
                loss = loss / grad_accum_steps

            loss.backward()

        optimizer.step()

        # Without this the timer stops once the work is queued, not once it is
        # done. It used to be masked by the .item() sync removed above.
        synchronize(device)
        step_time = time.perf_counter() - step_start

        if step - start_step >= warmup:
            step_times.append(step_time)
            step_tokens.append(tokens)

        rolling_tokens_per_sec = (
            sum(step_tokens) / sum(step_times) if step_times else float("nan")
        )

        # Evaluate periodically and log both to stdout and CSV.
        if step % cfg["eval_interval"] == 0 or step == cfg["max_iters"] - 1:
            avg_train_loss = (total_loss / grad_accum_steps).item()
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            val_loss = losses["val"]

            # if this is the best model so far, save a checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    f"{out_dir}/best.pt", model, optimizer, step, best_val_loss, cfg
                )

            print(
                f"step {step}: train loss {avg_train_loss:.4f} | val loss {val_loss:.4f} "
                f"| step time {step_time:.4f} | tokens/sec {rolling_tokens_per_sec:.4f}"
            )

            log_row = [
                step,
                f"{avg_train_loss:.4f}",
                f"{val_loss:.4f}",
                # 4 decimal places rounded any schedule below 1e-4 to 0.0000,
                # which flattened the whole learning-rate column.
                f"{lr:.6g}",
                f"{step_time:.4f}",
                f"{rolling_tokens_per_sec:.4f}",
            ]
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9  # in GB
                reserved = torch.cuda.memory_reserved() / 1e9  # in GB
                max_allocated = torch.cuda.max_memory_allocated() / 1e9  # peak so far
                log_row.extend(
                    [f"{allocated:.4f}", f"{reserved:.4f}", f"{max_allocated:.4f}"]
                )

            append_log(log_path, log_row)

            save_checkpoint(
                f"{out_dir}/latest.pt", model, optimizer, step, best_val_loss, cfg
            )

    return best_val_loss


def resolve_config_files(config_arg):
    """Accept a config name, a path to a yaml, or a directory of yamls."""
    config_path_candidate = f"configs/{config_arg}"
    config_file_candidate = f"{config_path_candidate}.yaml"

    if os.path.isdir(config_path_candidate):
        return [
            os.path.join(config_path_candidate, fn)
            for fn in sorted(os.listdir(config_path_candidate))
            if fn.endswith((".yaml", ".yml"))
        ]
    if os.path.isfile(config_file_candidate):
        return [config_file_candidate]
    if os.path.isdir(config_arg):
        return [
            os.path.join(config_arg, fn)
            for fn in sorted(os.listdir(config_arg))
            if fn.endswith((".yaml", ".yml"))
        ]
    if os.path.isfile(config_arg):
        return [config_arg]

    raise FileNotFoundError(
        f"Could not find config file or directory for '{config_arg}'"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="override the device from the config (default: config value, else auto)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    for cfg_file in resolve_config_files(args.config):
        print(f"Running config: {cfg_file}")
        cfg, out_dir, log_path = prepare_run_from_file(cfg_file, args.device)
        train(cfg, out_dir, log_path, resume=args.resume)


if __name__ == "__main__":
    main()
