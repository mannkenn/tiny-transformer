import torch
import csv
import os
from model import Transformer

LOG_PATH = "logs/train_log.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
block_size = 64
max_iters = 3000
eval_interval = 200
eval_iters = 100
learning_rate = 3e-4


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
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
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
                "lr"
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
        block_size=block_size,
        n_embd=512,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
    ).to(device)

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    ensure_log_file(LOG_PATH)

    for step in range(max_iters):
        # Evaluate periodically and log both to stdout and CSV.
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            append_log(LOG_PATH, [
                step,
                losses['train'],
                losses['val'],
                learning_rate
            ])

        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train()
