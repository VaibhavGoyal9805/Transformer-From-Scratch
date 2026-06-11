"""
Multi-Head Attention
====================

From "Attention Is All You Need" (Vaswani et al., 2017).

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i      = Attention(Q W_i^Q,  K W_i^K,  V W_i^V)

Instead of performing a single attention function with d_model-dimensional
keys, values, and queries, it is beneficial to linearly project Q, K, V
*h* times with different, learned linear projections to d_k, d_k, and d_v
dimensions respectively.  On each of these projected versions we perform
the attention function in parallel, yielding d_v-dimensional output values.
These are concatenated and once again projected, resulting in the final
output.

Multi-head attention allows the model to jointly attend to information
from different representation subspaces at different positions.

Workflow
--------
1. Linear projection:  Q = W_q(query),  K = W_k(key),  V = W_v(value)
2. Split heads:        reshape (batch, seq, d_model) → (batch, n_heads, seq, d_k)
3. Scaled dot-product attention on each head (in parallel via batched matmul)
4. Concatenate heads:  reshape (batch, n_heads, seq, d_k) → (batch, seq, d_model)
5. Output projection:  output = W_o(concatenated)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Splits the model dimension into multiple heads, applies scaled
    dot-product attention independently on each head, concatenates
    the results, and applies a final linear projection.

    Parameters
    ----------
    d_model : int
        Total model dimension (must be divisible by n_heads).
    n_heads : int
        Number of parallel attention heads.
    dropout : float
        Dropout probability applied to attention weights (default 0.0).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimension per head (same for keys, queries, values)
        self.d_v = d_model // n_heads

        # ── Linear projections ────────────────────────────────────────────
        # Each projects from d_model → d_model (i.e. n_heads * d_k)
        self.W_q = nn.Linear(d_model, d_model)  # query projection
        self.W_k = nn.Linear(d_model, d_model)  # key projection
        self.W_v = nn.Linear(d_model, d_model)  # value projection
        self.W_o = nn.Linear(d_model, d_model)  # output projection

        # ── Attention mechanism ───────────────────────────────────────────
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : Tensor  (batch, seq_len_q, d_model)
        key   : Tensor  (batch, seq_len_k, d_model)
        value : Tensor  (batch, seq_len_k, d_model)
        mask  : Tensor, optional
                Shape (batch, 1, 1, seq_len_k)  for padding masks, or
                      (batch, 1, seq_len_q, seq_len_k) for causal masks.
                Boolean — True means IGNORE (same convention as
                ScaledDotProductAttention).

        Returns
        -------
        output      : Tensor  (batch, seq_len_q, d_model)
        attn_weights : Tensor  (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # ── Step 1: Linear projections ────────────────────────────────────
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # ── Step 2: Split into heads ──────────────────────────────────────
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k)
        #                           → (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # ── Step 3: Scaled dot-product attention ──────────────────────────
        # Q, K, V are now (batch, n_heads, seq_len, d_k)
        # mask broadcasts over the n_heads dimension
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)
        # attn_output:  (batch, n_heads, seq_len_q, d_v)
        # attn_weights: (batch, n_heads, seq_len_q, seq_len_k)

        # ── Step 4: Concatenate heads ─────────────────────────────────────
        # (batch, n_heads, seq_len_q, d_v) → (batch, seq_len_q, n_heads, d_v)
        #                                  → (batch, seq_len_q, d_model)
        attn_output = (
            attn_output
            .transpose(1, 2)                              # (batch, seq_len_q, n_heads, d_v)
            .contiguous()                                  # ensure memory layout is contiguous
            .view(batch_size, -1, self.d_model)            # (batch, seq_len_q, d_model)
        )

        # ── Step 5: Output projection ─────────────────────────────────────
        output = self.W_o(attn_output)  # (batch, seq_len_q, d_model)

        return output, attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/multihead_attention.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Multi-Head Attention")
    print("=" * 60)

    torch.manual_seed(42)

    # ── Test 1: Basic shape test ──────────────────────────────────────────
    print("\n1. Basic shape test (batch=2, seq=10, d_model=32, n_heads=4)...")
    batch, seq, d_model, n_heads = 2, 10, 32, 4
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

    x = torch.randn(batch, seq, d_model)
    output, weights = mha(x, x, x)

    print(f"   Input shape   : {tuple(x.shape)}")
    print(f"   Output shape  : {tuple(output.shape)}")
    print(f"   Weights shape : {tuple(weights.shape)}")

    assert output.shape == (batch, seq, d_model), (
        f"Expected output shape {(batch, seq, d_model)}, got {tuple(output.shape)}"
    )
    assert weights.shape == (batch, n_heads, seq, seq), (
        f"Expected weights shape {(batch, n_heads, seq, seq)}, got {tuple(weights.shape)}"
    )
    print("   OK: Shape checks passed.")

    # ── Test 2: Self-attention (Q = K = V) ────────────────────────────────
    print("\n2. Self-attention test (query = key = value)...")
    x_self = torch.randn(batch, seq, d_model)
    out_self, w_self = mha(x_self, x_self, x_self)

    assert out_self.shape == (batch, seq, d_model)
    # Attention weights should sum to 1 across the key dimension
    row_sums = w_self.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Attention weights don't sum to 1: {row_sums}"
    )
    print(f"   Output shape  : {tuple(out_self.shape)}")
    print(f"   Weight row sum (sample): {row_sums[0, 0, :3].tolist()}")
    print("   OK: Self-attention works correctly.")

    # ── Test 3: Cross-attention (different key/value seq length) ──────────
    print("\n3. Cross-attention test (seq_q=10, seq_kv=20)...")
    seq_q, seq_kv = 10, 20
    query = torch.randn(batch, seq_q, d_model)
    key_val = torch.randn(batch, seq_kv, d_model)

    out_cross, w_cross = mha(query, key_val, key_val)

    print(f"   Query shape   : {tuple(query.shape)}")
    print(f"   Key/Val shape : {tuple(key_val.shape)}")
    print(f"   Output shape  : {tuple(out_cross.shape)}")
    print(f"   Weights shape : {tuple(w_cross.shape)}")

    assert out_cross.shape == (batch, seq_q, d_model), (
        f"Expected {(batch, seq_q, d_model)}, got {tuple(out_cross.shape)}"
    )
    assert w_cross.shape == (batch, n_heads, seq_q, seq_kv), (
        f"Expected {(batch, n_heads, seq_q, seq_kv)}, got {tuple(w_cross.shape)}"
    )
    print("   OK: Cross-attention shape checks passed.")

    # ── Test 4: Causal mask ───────────────────────────────────────────────
    print("\n4. Causal (look-ahead) mask test...")
    seq_c = 8
    x_causal = torch.randn(batch, seq_c, d_model)

    # Upper-triangular mask: True above diagonal → can't attend to future
    causal_mask = torch.triu(
        torch.ones(seq_c, seq_c, dtype=torch.bool), diagonal=1
    )
    # Expand to (batch, 1, seq_c, seq_c) for broadcasting over heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)

    out_causal, w_causal = mha(x_causal, x_causal, x_causal, mask=causal_mask)

    print(f"   Causal mask shape : {tuple(causal_mask.shape)}")
    print(f"   Output shape      : {tuple(out_causal.shape)}")

    assert out_causal.shape == (batch, seq_c, d_model)

    # Check that upper triangle of attention weights is zero (no future attention)
    for h in range(n_heads):
        upper = torch.triu(w_causal[0, h], diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), (
            f"Head {h}: future positions have non-zero attention weight!"
        )
    print(f"   Weights (head 0, sample):\n{w_causal[0, 0, :4, :4]}")
    print("   OK: No attention to future positions in any head.")

    # ── Test 5: Parameter count ───────────────────────────────────────────
    print("\n5. Parameter count test...")
    total_params = sum(p.numel() for p in mha.parameters())

    # Expected: 4 linear layers, each (d_model x d_model) weight + d_model bias
    # = 4 * (d_model * d_model + d_model)
    expected_params = 4 * (d_model * d_model + d_model)

    print(f"   Total parameters    : {total_params}")
    print(f"   Expected parameters : {expected_params}")
    print(f"   Breakdown:")
    print(f"     W_q: {d_model}×{d_model} + {d_model} = {d_model * d_model + d_model}")
    print(f"     W_k: {d_model}×{d_model} + {d_model} = {d_model * d_model + d_model}")
    print(f"     W_v: {d_model}×{d_model} + {d_model} = {d_model * d_model + d_model}")
    print(f"     W_o: {d_model}×{d_model} + {d_model} = {d_model * d_model + d_model}")

    assert total_params == expected_params, (
        f"Parameter count mismatch: got {total_params}, expected {expected_params}"
    )
    print("   OK: Parameter count matches.")

    print("\n" + "=" * 60)
    print("All Multi-Head Attention tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
