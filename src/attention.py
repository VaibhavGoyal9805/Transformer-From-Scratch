"""
Scaled Dot-Product Attention
=============================

From "Attention Is All You Need" (Vaswani et al., 2017).

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

The three inputs:
  • Query (Q) — what each position is "looking for"
  • Key   (K) — what each position "advertises"
  • Value (V) — the actual content each position carries

Workflow
--------
1. Compute raw attention scores:   scores = Q K^T / sqrt(d_k)
2. (Optional) Apply mask:          scores[mask] = -inf
3. Normalise to probabilities:     weights = softmax(scores)
4. (Optional) Apply dropout:       weights = dropout(weights)
5. Weighted sum of values:         output  = weights V

Why scale by sqrt(d_k)?
-----------------------
Without scaling, when d_k is large the dot products grow large in magnitude,
pushing softmax into regions where its gradients are extremely small.
Dividing by sqrt(d_k) keeps the variance of the dot products at ~1,
regardless of the dimension, which stabilises training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Supports arbitrary leading dimensions so that the same module works
    for both:
      • 3-D tensors: (batch, seq_len, d_model)       — single-head
      • 4-D tensors: (batch, n_heads, seq_len, d_k)   — multi-head

    Parameters
    ----------
    dropout : float
        Dropout probability applied to the attention weights (default 0.0).
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        query : Tensor  (..., seq_len_q, d_k)
        key   : Tensor  (..., seq_len_k, d_k)
        value : Tensor  (..., seq_len_k, d_v)
        mask  : Tensor, optional  (..., seq_len_q, seq_len_k)
                Boolean mask — **True means IGNORE** (consistent with
                PyTorch's `masked_fill` convention).

        Returns
        -------
        output  : Tensor  (..., seq_len_q, d_v)
        weights : Tensor  (..., seq_len_q, seq_len_k)
        """
        # d_k is the last dimension of query (and key)
        d_k = query.size(-1)

        # ── Step 1: Raw attention scores ──────────────────────────────────
        # Q @ K^T  →  (..., seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # ── Step 2: Apply mask (if provided) ──────────────────────────────
        if mask is not None:
            # mask should be boolean; True → position to mask out
            scores = scores.masked_fill(mask, float('-inf'))

        # ── Step 3: Softmax over the last dimension (key positions) ───────
        weights = F.softmax(scores, dim=-1)

        # Handle the case where an entire row was masked (all -inf → NaN)
        # Replace NaN with 0.0 so downstream computation stays clean
        weights = weights.masked_fill(torch.isnan(weights), 0.0)

        # ── Step 4: Dropout on attention weights ──────────────────────────
        weights = self.dropout(weights)

        # ── Step 5: Weighted sum of values ────────────────────────────────
        # weights @ V  →  (..., seq_len_q, d_v)
        output = torch.matmul(weights, value)

        return output, weights


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/attention.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Scaled Dot-Product Attention")
    print("=" * 60)

    torch.manual_seed(42)

    # ── Test 1: Basic shapes (3-D — single head) ─────────────────────────
    print("\n1. Basic shape test (3-D single-head)...")
    batch, seq_q, seq_k, d_k, d_v = 2, 5, 7, 16, 16
    Q = torch.randn(batch, seq_q, d_k)
    K = torch.randn(batch, seq_k, d_k)
    V = torch.randn(batch, seq_k, d_v)

    attn = ScaledDotProductAttention(dropout=0.0)
    out, weights = attn(Q, K, V)

    print(f"   Q shape       : {tuple(Q.shape)}")
    print(f"   K shape       : {tuple(K.shape)}")
    print(f"   V shape       : {tuple(V.shape)}")
    print(f"   Output shape  : {tuple(out.shape)}")
    print(f"   Weights shape : {tuple(weights.shape)}")

    assert out.shape == (batch, seq_q, d_v), f"Expected {(batch, seq_q, d_v)}, got {tuple(out.shape)}"
    assert weights.shape == (batch, seq_q, seq_k), f"Expected {(batch, seq_q, seq_k)}, got {tuple(weights.shape)}"
    print("   OK: Shape checks passed.")

    # ── Test 2: Shapes (4-D — multi-head) ────────────────────────────────
    print("\n2. Shape test (4-D multi-head)...")
    n_heads = 4
    Q4 = torch.randn(batch, n_heads, seq_q, d_k)
    K4 = torch.randn(batch, n_heads, seq_k, d_k)
    V4 = torch.randn(batch, n_heads, seq_k, d_v)

    out4, weights4 = attn(Q4, K4, V4)
    print(f"   Q shape       : {tuple(Q4.shape)}")
    print(f"   Output shape  : {tuple(out4.shape)}")
    print(f"   Weights shape : {tuple(weights4.shape)}")

    assert out4.shape == (batch, n_heads, seq_q, d_v)
    assert weights4.shape == (batch, n_heads, seq_q, seq_k)
    print("   OK: 4-D shape checks passed.")

    # ── Test 3: Attention weights sum to 1 ───────────────────────────────
    print("\n3. Attention weights sum-to-1 test...")
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"Row sums not 1: {row_sums}"
    print(f"   Row sums (sample): {row_sums[0]}")
    print("   OK: Each query's weights sum to 1.")

    # ── Test 4: Manual math verification ─────────────────────────────────
    print("\n4. Manual math verification (tiny example)...")
    # 1 batch, 2 queries, 3 keys, d_k=2, d_v=2
    Q_t = torch.tensor([[[1.0, 0.0],
                          [0.0, 1.0]]])       # (1, 2, 2)
    K_t = torch.tensor([[[1.0, 0.0],
                          [0.0, 1.0],
                          [1.0, 1.0]]])       # (1, 3, 2)
    V_t = torch.tensor([[[10.0, 0.0],
                          [0.0, 10.0],
                          [5.0,  5.0]]])      # (1, 3, 2)

    out_t, w_t = attn(Q_t, K_t, V_t)

    # Compute expected scores manually: Q @ K^T / sqrt(2)
    sqrt_dk = math.sqrt(2.0)
    expected_scores = torch.tensor([[[1.0/sqrt_dk, 0.0/sqrt_dk, 1.0/sqrt_dk],
                                      [0.0/sqrt_dk, 1.0/sqrt_dk, 1.0/sqrt_dk]]])
    expected_weights = F.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_weights, V_t)

    assert torch.allclose(w_t, expected_weights, atol=1e-5), \
        f"Weights mismatch!\nGot:      {w_t}\nExpected: {expected_weights}"
    assert torch.allclose(out_t, expected_output, atol=1e-5), \
        f"Output mismatch!\nGot:      {out_t}\nExpected: {expected_output}"
    print(f"   Weights:\n   {w_t.squeeze()}")
    print(f"   Output:\n   {out_t.squeeze()}")
    print("   OK: Manual computation matches.")

    # ── Test 5: Padding mask ─────────────────────────────────────────────
    print("\n5. Padding mask test...")
    # Mask out the last key position for all queries
    mask_pad = torch.zeros(1, 2, 3, dtype=torch.bool)
    mask_pad[:, :, 2] = True  # mask position 2 (third key)

    out_masked, w_masked = attn(Q_t, K_t, V_t, mask=mask_pad)
    print(f"   Mask:\n   {mask_pad.squeeze()}")
    print(f"   Weights (masked):\n   {w_masked.squeeze()}")

    # Position 2 should have zero weight
    assert torch.allclose(w_masked[:, :, 2], torch.zeros(1, 2), atol=1e-6), \
        "Masked position should have zero weight!"
    # Remaining weights should still sum to 1
    row_sums_m = w_masked.sum(dim=-1)
    assert torch.allclose(row_sums_m, torch.ones_like(row_sums_m), atol=1e-5)
    print("   OK: Masked positions have zero weight; rows still sum to 1.")

    # ── Test 6: Causal (look-ahead) mask ─────────────────────────────────
    print("\n6. Causal (look-ahead) mask test...")
    seq = 4
    Q_c = torch.randn(1, seq, d_k)
    K_c = torch.randn(1, seq, d_k)
    V_c = torch.randn(1, seq, d_k)

    # Upper-triangular mask: True above diagonal → can't attend to future
    causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0)  # (1, seq, seq)

    _, w_causal = attn(Q_c, K_c, V_c, mask=causal_mask)
    print(f"   Causal mask:\n{causal_mask.squeeze()}")
    print(f"   Causal weights:\n{w_causal.squeeze()}")

    # Check upper triangle of weights is zero
    upper = torch.triu(w_causal.squeeze(), diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), \
        "Future positions should have zero weight!"
    print("   OK: No attention to future positions.")

    # ── Test 7: Dropout on attention weights ─────────────────────────────
    print("\n7. Dropout test...")
    attn_drop = ScaledDotProductAttention(dropout=0.5)
    attn_drop.train()
    _, w_drop = attn_drop(Q, K, V)
    num_zeros = (w_drop == 0.0).sum().item()
    total = w_drop.numel()
    print(f"   Zeros in train mode: {num_zeros} / {total}")
    assert num_zeros > 0, "Dropout should zero out some weights in training mode!"

    attn_drop.eval()
    _, w_nodrop = attn_drop(Q, K, V)
    num_zeros_eval = (w_nodrop == 0.0).sum().item()
    print(f"   Zeros in eval mode : {num_zeros_eval} / {total}")
    print("   OK: Dropout behaves correctly in train/eval modes.")

    print("\n" + "=" * 60)
    print("All Scaled Dot-Product Attention tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
