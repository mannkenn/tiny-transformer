from collections import deque
import time
import torch
import csv
import os
import yaml
import argparse
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
    }

# parse command line arguments for run_name and config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

cfg = load_config(f"configs/{args.config}.yaml")
cfg = parse_config(cfg)
print(cfg)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = f"logs/train_log_{cfg['run_name']}.csv"

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
            writer.writerow([
                "step",
                "train_loss",
                "val_loss",
                "lr",
                "elapsed_time",
                "tokens_per_sec"
            ])

def append_log(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

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

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    ensure_log_file(LOG_PATH)

    step_times = deque(maxlen=20)
    step_tokens = deque(maxlen=20)
    for step in range(cfg["max_iters"]):
        step_start = time.time()

        B, T  = cfg["batch_size"], cfg["block_size"]
        tokens = B * T

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)
        step_tokens.append(tokens)
        rolling_tokens_per_sec = sum(step_tokens) / sum(step_times) # rolling tokens/sec over last 20 steps

        # Evaluate periodically and log both to stdout and CSV.
        if step % cfg["eval_interval"] == 0 or step == cfg["max_iters"] - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | step time {step_time:.4f} | tokens/sec {rolling_tokens_per_sec:.4f}"
            )

            append_log(LOG_PATH, [
                step,
                f"{losses['train']:.4f}",
                f"{losses['val']:.4f}",
                f"{cfg['learning_rate']:.4f}",
                f"{step_time:.4f}", # step time
                f"{rolling_tokens_per_sec:.4f}" # tokens/sec
            ])

if __name__ == "__main__":
    train()
