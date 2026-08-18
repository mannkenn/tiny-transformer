"""Frozen copy of the pre-refactor looped attention implementation.

This is the original ``Head`` / ``MultiHeadedSelfAttention`` from ``model.py`` at
commit ddbbf2c, reproduced verbatim apart from the class rename. It is not used
by training -- it exists so that ``test_attention_equivalence.py`` can prove the
fused implementation in ``model.py`` computes exactly the same function.

Do not "fix" or modernise this file. Its value is that it is unchanged.
"""

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


class LoopedMultiHeadedSelfAttention(nn.Module):
    """Multi-head self-attention (original Python-loop implementation)."""

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


def copy_weights_to_fused(looped, fused):
    """Copy looped-attention weights into the fused module.

    Head ``i`` owned three ``(head_size, n_embd)`` matrices. The fused module has
    a single ``(3 * n_embd, n_embd)`` matrix whose rows are ordered
    ``[Q for all heads | K for all heads | V for all heads]``. Head ``i``'s query
    rows therefore land at ``[i * head_size : (i + 1) * head_size]`` within the
    first ``n_embd`` rows, and likewise for K and V in the next two blocks.

    This is exactly the layout that makes
    ``qkv(x).split(n_embd, dim=2)`` followed by
    ``view(B, T, n_heads, head_size).transpose(1, 2)`` reproduce the per-head
    projections and the ``torch.cat(..., dim=-1)`` of the original code.
    """
    n_heads = len(looped.heads)
    head_size = looped.head_size
    n_embd = fused.n_embd

    with torch.no_grad():
        for i, head in enumerate(looped.heads):
            rows = slice(i * head_size, (i + 1) * head_size)
            fused.qkv.weight[rows] = head.query.weight
            fused.qkv.weight[n_embd + rows.start : n_embd + rows.stop] = head.key.weight
            fused.qkv.weight[2 * n_embd + rows.start : 2 * n_embd + rows.stop] = (
                head.value.weight
            )
        fused.proj.weight.copy_(looped.proj.weight)

    assert n_heads * head_size == n_embd


def copy_weights_to_looped(fused, looped):
    """Inverse of :func:`copy_weights_to_fused`, for end-to-end parity checks."""
    head_size = looped.head_size
    n_embd = fused.n_embd

    with torch.no_grad():
        for i, head in enumerate(looped.heads):
            start, stop = i * head_size, (i + 1) * head_size
            head.query.weight.copy_(fused.qkv.weight[start:stop])
            head.key.weight.copy_(fused.qkv.weight[n_embd + start : n_embd + stop])
            head.value.weight.copy_(
                fused.qkv.weight[2 * n_embd + start : 2 * n_embd + stop]
            )
        looped.proj.weight.copy_(fused.proj.weight)
