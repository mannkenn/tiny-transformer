import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadedSelfAttention(nn.Module):
    """Multi-head self-attention with a single fused QKV projection.

    All heads are computed in one batched attention call. The earlier version of
    this module held an ``nn.ModuleList`` of per-head submodules and looped over
    them in Python, which issued ``3 * n_heads`` small GEMMs and ``n_heads``
    attention calls per layer. The math is unchanged -- see
    ``tests/test_attention_equivalence.py`` for the proof against the original
    implementation, which is preserved in ``tests/reference_attention.py``.
    """

    def __init__(self, n_embd, n_heads, dropout, use_flash_attention=True):
        super().__init__()
        assert n_embd % n_heads == 0

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads
        self.use_flash_attention = use_flash_attention

        # One projection producing Q, K and V for every head at once. Output is
        # ordered [Q | K | V], and within each block, heads are laid out
        # contiguously, so row block [i * head_size : (i + 1) * head_size] of Q
        # is exactly what head i's own query projection used to produce.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        # Dropout applied to the attention weights (inside attention)...
        self.attn_dropout_p = dropout
        # ...and dropout applied to the block output (after the projection).
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, t, B, T):
        # (B, T, n_embd) -> (B, n_heads, T, head_size)
        return t.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

    def forward(self, x, is_causal=False):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = self._split_heads(q, B, T)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        dropout_p = self.attn_dropout_p if self.training else 0.0

        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # (B, n_heads, T, head_size)
        else:
            # Explicit attention, kept as the readable baseline the fused path
            # is measured against.
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
            attn = F.dropout(attn, p=dropout_p, training=self.training)
            out = torch.matmul(attn, v)  # (B, n_heads, T, head_size)

        # Concatenate heads back into the embedding dimension. This reproduces
        # the old torch.cat([...], dim=-1) over per-head outputs.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


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
    def __init__(self, n_embd, n_heads, dropout, use_flash_attention=True, norm_first=False):
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
        self.norm_first = norm_first

    def forward(self, x):
        if self.norm_first:
            # Pre-norm, as used by GPT-2 / nanoGPT: the residual stream stays
            # un-normalised end to end, which is more stable at depth.
            x = x + self.mhsa(self.ln1(x), is_causal=True)
            x = x + self.ff(self.ln2(x))
        else:
            # Post-norm, as in the original "Attention Is All You Need". This is
            # the default so that previously recorded results stay comparable.
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
        norm_first=False,
    ):
        super().__init__()

        self.block_size = block_size
        self.n_embd = n_embd

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(n_embd, n_heads, dropout, use_flash_attention, norm_first)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.tok_emb(idx)  # (B,T,C)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

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
