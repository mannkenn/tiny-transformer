import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """General attention head for self-attention and cross-attention."""

    def __init__(self, n_embd, head_size, dropout, use_flash_attention=True):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_flash_attention = use_flash_attention

    def forward(self, query, key, value, is_causal: bool = False):
        # query: (B, Tq, C), key/value: (B, Tk, C)
        q = self.query(query)  # (B, Tq, hs)
        k = self.key(key)      # (B, Tk, hs)
        v = self.value(value)  # (B, Tk, hs)

        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )  # (B, Tq, hs)
        else:
            scale = q.size(-1) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            if is_causal:
                q_len, k_len = attn.size(-2), attn.size(-1)
                causal_mask = torch.triu(
                    torch.ones(q_len, k_len, device=attn.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn = attn.masked_fill(causal_mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout.p, training=self.training)
            out = torch.matmul(attn, v)  # (B, Tq, hs)
        return out


class MultiHeadedSelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, n_embd, n_heads, dropout, use_flash_attention=True):
        super().__init__()
        assert n_embd % n_heads == 0

        self.head_size = n_embd // n_heads
        self.heads = nn.ModuleList(
            [
                Head(n_embd, self.head_size, dropout, use_flash_attention)
                for _ in range(n_heads)
            ]
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
    def __init__(self, n_embd, n_heads, dropout, use_flash_attention=True):
        super().__init__()

        self.mhsa = MultiHeadedSelfAttention(
            n_embd,
            n_heads,
            dropout,
            use_flash_attention,
        )
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
        use_flash_attention=True,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_embd = n_embd

        self.src_tok_emb = nn.Embedding(vocab_size, n_embd)
        self.tgt_tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(n_embd, n_heads, dropout, use_flash_attention)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.src_tok_emb(idx) # (B,T,C)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device)) # (T,C)
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
            )

        return logits, loss