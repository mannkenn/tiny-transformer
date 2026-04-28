from collections import deque
import time
import torch
import csv
import os
import yaml
import argparse
import shutil
from datetime import datetime
from model import Transformer

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "grad_accum_steps": int(cfg.get("grad_accum_steps", 1)), # optional grad accum
    }

# parse command line arguments for run_name, config file and optional checkpoint to resume from
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--resume", type=str, default=None)
args = parser.parse_args()

cfg = load_config(f"configs/{args.config}.yaml")
cfg = parse_config(cfg)
print(cfg)

# create run id and output directory for this run
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"outputs/{cfg['run_name']}_{run_id}"
os.makedirs(OUT_DIR, exist_ok=True)

# save config used
shutil.copy(
    f"configs/{args.config}.yaml",
    f"{OUT_DIR}/config.yaml"
)

LOG_PATH = f"{OUT_DIR}/train_log.csv"

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
                "lr",
                "elapsed_time",
                "tokens_per_sec"
            ]
            if torch.cuda.is_available():
                headers.extend([
                    "allocated_memory_gb",
                    "reserved_memory_gb",
                    "max_allocated_memory_gb"
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
    ).to(device)

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
        total_loss = 0.0
        
        step_start = time.time()

        B, T  = cfg["batch_size"], cfg["block_size"]
        tokens = B * T

        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(grad_accum_steps):
            xb, yb = get_batch("train", train_data, val_data)
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

            # memory usage (only for CUDA)
            log_row = [
                step,
                f"{avg_train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{cfg['learning_rate']:.4f}",
                f"{step_time:.4f}", # step time
                f"{rolling_tokens_per_sec:.4f}", # tokens/sec
            ]
            
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
    train()
