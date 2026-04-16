import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """General attention head for self-attention and cross-attention."""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, is_causal: bool = False):
        # query: (B, Tq, C), key/value: (B, Tk, C)
        q = self.query(query)  # (B, Tq, hs)
        k = self.key(key)      # (B, Tk, hs)
        v = self.value(value)  # (B, Tk, hs)

        # (B, Tq, hs) @ (B, hs, Tk) -> (B, Tq, Tk)
        att = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)

        if is_causal:
            Tq, Tk = att.shape[-2], att.shape[-1]
            causal_mask = torch.tril(
                torch.ones(Tq, Tk, device=att.device, dtype=torch.bool)
            )
            att = att.masked_fill(~causal_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # (B, Tq, Tk) @ (B, Tk, hs) -> (B, Tq, hs)
        out = att @ v
        return out


class MultiHeadedSelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()
        assert n_embd % n_heads == 0

        self.head_size = n_embd // n_heads
        self.heads = nn.ModuleList(
            [Head(n_embd, self.head_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_causal=False):
        out = torch.cat([h(x, x, x, is_causal) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout):
        super().__init__()

        self.mhsa = MultiHeadedSelfAttention(n_embd, n_heads, dropout)
        self.ff = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.mhsa(x, is_causal=True))
        x = self.ln2(x + self.ff(x))
        return x
    
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=64,
        n_embd=512,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_embd = n_embd

        self.src_tok_emb = nn.Embedding(vocab_size, n_embd)
        self.tgt_tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_heads, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        for block in self.decoder_blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))  # (B, T, V)

        loss = None
        if targets is not None:
            Bt, Tt, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(Bt * Tt, C),
                targets.reshape(Bt * Tt),
                ignore_index=self.pad_id,
            )

        return logits, loss