from collections import deque
import time
import torch
import csv
import os
import argparse
import shutil
import math
from datetime import datetime
from model import Transformer
from utils import load_config, parse_config

device = "cuda" if torch.cuda.is_available() else "cpu"

# parse command line arguments for run_name, config file and optional checkpoint to resume from
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

# globals that will be set per-config run
cfg = None
OUT_DIR = None
LOG_PATH = None
use_amp = False
amp_dtype = torch.bfloat16

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def prepare_run_from_file(config_path):
    """Load and prepare globals for a single config file path."""
    global cfg, OUT_DIR, LOG_PATH, use_amp, amp_dtype

    raw = load_config(config_path)
    cfg = parse_config(raw)
    print(cfg)

    # create run id and output directory for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # include config filename to distinguish runs from the same run_name
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    OUT_DIR = f"outputs/{cfg['run_name']}_{cfg_name}_{run_id}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # save config used
    shutil.copy(config_path, f"{OUT_DIR}/config.yaml")

    LOG_PATH = f"{OUT_DIR}/train_log.csv"

    # mixed precision
    use_amp = cfg.get("mixed_precision", False) and torch.cuda.is_available()

    amp_dtype = dtype_map.get(cfg.get("dtype", "bf16"), torch.bfloat16)

def load_data(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data, len(chars)

def get_batch(split, train_data, val_data):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - cfg["block_size"], (cfg["batch_size"],))
    x = torch.stack([source[i : i + cfg["block_size"]] for i in ix])
    y = torch.stack([source[i + 1 : i + cfg["block_size"] + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            xb, yb = get_batch(split, train_data, val_data)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# logging
def ensure_log_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            headers = [
                "step",
                "train_loss",
                "val_loss",
                "learning_rate",
                "step_time",
                "tokens_per_sec"
            ]
            if torch.cuda.is_available():
                headers.extend([
                    "allocated_gb",
                    "reserved_gb",
                    "max_allocated_gb"
                ])
            writer.writerow(headers)

def append_log(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# model checkpointing
def save_checkpoint(path, model, optimizer, step, best_val_loss, cfg):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, map_location=device):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

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

def train():
    torch.manual_seed(1337)
    
    # load data
    train_data, val_data, vocab_size = load_data()

    # init model
    model = Transformer(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        use_flash_attention=cfg["use_flash_attention"],
    ).to(device)
    
    # torch compile
    if cfg.get("torch_compile", False):
        print("compiling model with torch.compile...")
        model = torch.compile(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    # best loss tracking
    best_val_loss = float("inf")

    # resume from checkpoint if specified
    start_step = 0
    if args.resume is not None:
        start_step, best_val_loss, _ = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at step {start_step}")
    
    # ensure log file exists
    ensure_log_file(LOG_PATH)

    # grad accum 
    grad_accum_steps = cfg["grad_accum_steps"]
    
    # rolling average of step time and tokens processed for logging tokens/sec
    step_times = deque(maxlen=20)
    step_tokens = deque(maxlen=20)

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

        total_loss = 0.0
        step_start = time.time()

        B, T  = cfg["batch_size"], cfg["block_size"]
        tokens = B * T * grad_accum_steps # total tokens processed in this step (accounting for grad accum)

        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(grad_accum_steps): # accum gradient over steps
            xb, yb = get_batch("train", train_data, val_data)
            
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits, loss = model(xb, yb)
                total_loss += loss.item()
                loss = loss / grad_accum_steps
            
            loss.backward()

        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)
        step_tokens.append(tokens)
        rolling_tokens_per_sec = sum(step_tokens) / sum(step_times) # rolling tokens/sec over last 20 steps
        
        # Compute average training loss from accumulated gradients
        avg_train_loss = total_loss / grad_accum_steps

        # Evaluate periodically and log both to stdout and CSV.
        if step % cfg["eval_interval"] == 0 or step == cfg["max_iters"] - 1:
            losses = estimate_loss(model, train_data, val_data)
            val_loss = losses["val"]

            # if this is the best model so far, save a checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    f"{OUT_DIR}/best.pt",
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    cfg,
                )

            print(
                f"step {step}: train loss {avg_train_loss:.4f} | val loss {val_loss:.4f} | step time {step_time:.4f} | tokens/sec {rolling_tokens_per_sec:.4f}"
            )

            log_row = [
                step,
                f"{avg_train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{lr:.4f}",
                f"{step_time:.4f}", # step time
                f"{rolling_tokens_per_sec:.4f}", # tokens/sec
            ]
            # memory usage (only for CUDA)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9  # in GB
                reserved = torch.cuda.memory_reserved() / 1e9    # in GB
                max_allocated = torch.cuda.max_memory_allocated() / 1e9  # peak so far
                log_row.extend([
                    f"{allocated:.4f}",  # allocated memory
                    f"{reserved:.4f}",   # reserved memory
                    f"{max_allocated:.4f}",  # max allocated memory
                ])

            append_log(LOG_PATH, log_row)

            save_checkpoint(
                f"{OUT_DIR}/latest.pt",
                model,
                optimizer,
                step,
                best_val_loss,
                cfg,
            )

if __name__ == "__main__":
    # allow --config to be either a single config name (without .yaml),
    # a path to a single yaml, or a directory under configs/ containing many yaml files.
    config_arg = args.config
    config_path_candidate = f"configs/{config_arg}"
    config_file_candidate = f"{config_path_candidate}.yaml"

    config_files = []
    if os.path.isdir(config_path_candidate):
        # collect all .yaml files in the directory
        for fn in sorted(os.listdir(config_path_candidate)):
            if fn.endswith(".yaml") or fn.endswith(".yml"):
                config_files.append(os.path.join(config_path_candidate, fn))
    elif os.path.isfile(config_file_candidate):
        config_files = [config_file_candidate]
    elif os.path.isfile(config_arg):
        # user supplied a direct path
        config_files = [config_arg]
    else:
        raise FileNotFoundError(f"Could not find config file or directory for '{config_arg}'")

    for cfg_file in config_files:
        print(f"Running config: {cfg_file}")
        prepare_run_from_file(cfg_file)
        train()
