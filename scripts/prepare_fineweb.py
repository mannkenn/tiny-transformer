"""Tokenize FineWeb-Edu into uint16 shards (adapted from Karpathy's build-nanogpt).

STATUS: not wired into training. `train.py` trains a character-level model on
`input.txt`; this produces GPT-2 BPE token shards. Using them would mean a new
data loader, a vocab of 50257 instead of 65, and a different model scale, which
would invalidate every result currently in `experiment_summary.md`. It is kept
here as the data-prep half of that future work, not as something the training
loop calls.

Requires extra dependencies that are deliberately not in `requirements.txt`:

    pip install datasets tiktoken tqdm

Downloads roughly 10B tokens (tens of GB) with the default `sample-10BT`.

    python scripts/prepare_fineweb.py --shard-size 100000000
"""

import argparse
import multiprocessing as mp
import os

import numpy as np


def build_tokenizer():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]  # end of text token

    def tokenize(doc):
        # tokenizes a single document and returns a numpy array of uint16 tokens
        tokens = [eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
            "token dictionary too large for uint16"
        )
        return tokens_np.astype(np.uint16)

    return tokenize


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default="edu_fineweb10B")
    parser.add_argument("--remote-name", default="sample-10BT")
    parser.add_argument(
        "--shard-size", type=int, default=int(1e8), help="tokens per shard"
    )
    parser.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    from datasets import load_dataset
    from tqdm import tqdm

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.local_dir)
    os.makedirs(cache_dir, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=args.remote_name, split="train")
    tokenize = build_tokenizer()
    shard_size = args.shard_size

    with mp.Pool(args.nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    cache_dir, f"edufineweb_{split}_{shard_index:06d}"
                )
                # split the document into whatever fits; the remainder goes next
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(cache_dir, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
