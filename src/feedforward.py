"""
Position-wise Feed-Forward Network
====================================

From "Attention Is All You Need" (Vaswani et al., 2017), Section 3.3.

    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

Each position in the sequence is transformed independently and identically
by the same two-layer fully-connected network:

    Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model) → Dropout

The inner dimension d_ff is typically 4× the model dimension (e.g. 2048
when d_model = 512), giving the network extra capacity to learn non-linear
transformations before projecting back down.

Why position-wise?
------------------
The same weight matrices W₁, W₂ are shared across every position in the
sequence, but each position is processed independently — there is no
interaction between positions inside the FFN.  Cross-position mixing
happens solely in the attention layers.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applies two linear transformations with a ReLU activation in between,
    followed by dropout after each transformation stage.

    Parameters
    ----------
    d_model : int
        Input and output dimensionality (default 512 in the paper).
    d_ff : int
        Inner (hidden) dimensionality (default 2048 in the paper, i.e. 4× d_model).
    dropout : float
        Dropout probability applied after ReLU and after the second linear
        layer (default 0.1).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # ── Layer 1: Expand d_model → d_ff ────────────────────────────────
        self.linear1 = nn.Linear(d_model, d_ff)

        # ── ReLU activation ───────────────────────────────────────────────
        self.relu = nn.ReLU()

        # ── Dropout after activation ──────────────────────────────────────
        self.dropout1 = nn.Dropout(p=dropout)

        # ── Layer 2: Project d_ff → d_model ───────────────────────────────
        self.linear2 = nn.Linear(d_ff, d_model)

        # ── Dropout after second linear layer ─────────────────────────────
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise feed-forward network.

        Parameters
        ----------
        x : Tensor  (..., d_model)
            Input tensor — typically (batch, seq_len, d_model), but any
            number of leading dimensions is supported.

        Returns
        -------
        Tensor  (..., d_model)
            Output with the same shape as the input.
        """
        # ── Step 1: First linear + ReLU ───────────────────────────────────
        # x @ W₁ᵀ + b₁  →  (..., d_ff),  then ReLU clips negatives to 0
        out = self.relu(self.linear1(x))

        # ── Step 2: Dropout after activation ──────────────────────────────
        out = self.dropout1(out)

        # ── Step 3: Second linear projection back to d_model ──────────────
        # out @ W₂ᵀ + b₂  →  (..., d_model)
        out = self.linear2(out)

        # ── Step 4: Final dropout ─────────────────────────────────────────
        out = self.dropout2(out)

        return out


# Convenience alias used elsewhere in the project
FFN = FeedForwardNetwork


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/feedforward.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Position-wise Feed-Forward Network")
    print("=" * 60)

    torch.manual_seed(42)

    d_model = 32
    d_ff = 128  # 4× expansion
    batch = 2
    seq_len = 10

    # ── Test 1: Shape preservation ────────────────────────────────────────
    print("\n1. Shape preservation test...")
    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=0.0)
    x = torch.randn(batch, seq_len, d_model)
    out = ffn(x)

    print(f"   Input shape  : {tuple(x.shape)}")
    print(f"   Output shape : {tuple(out.shape)}")

    assert out.shape == (batch, seq_len, d_model), \
        f"Expected {(batch, seq_len, d_model)}, got {tuple(out.shape)}"
    print("   OK: Shape preserved (batch=2, seq=10, d_model=32).")

    # ── Test 2: d_ff = 128 (4× expansion) ────────────────────────────────
    print("\n2. Inner dimension test (d_ff=128, 4× d_model)...")
    assert ffn.linear1.in_features == d_model
    assert ffn.linear1.out_features == d_ff
    assert ffn.linear2.in_features == d_ff
    assert ffn.linear2.out_features == d_model
    print(f"   linear1: {d_model} → {d_ff}")
    print(f"   linear2: {d_ff} → {d_model}")
    print("   OK: Expansion factor is correct.")

    # ── Test 3: Parameter count ───────────────────────────────────────────
    print("\n3. Parameter count test...")
    # W1: (d_ff, d_model), b1: (d_ff), W2: (d_model, d_ff), b2: (d_model)
    expected_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
    actual_params = sum(p.numel() for p in ffn.parameters())

    print(f"   Expected : {expected_params}")
    print(f"   Actual   : {actual_params}")

    assert actual_params == expected_params, \
        f"Parameter count mismatch: expected {expected_params}, got {actual_params}"
    print("   OK: Parameter count matches formula.")

    # ── Test 4: Dropout behaviour (train vs eval) ─────────────────────────
    print("\n4. Dropout test (train vs eval)...")
    ffn_drop = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=0.5)
    x_drop = torch.randn(batch, seq_len, d_model)

    # Train mode — dropout active, outputs should vary between runs
    ffn_drop.train()
    out_train_1 = ffn_drop(x_drop)
    out_train_2 = ffn_drop(x_drop)
    differ = not torch.allclose(out_train_1, out_train_2, atol=1e-6)
    print(f"   Train mode: two forward passes differ? {differ}")
    assert differ, "Dropout should cause different outputs in train mode!"

    # Eval mode — dropout disabled, outputs should be deterministic
    ffn_drop.eval()
    out_eval_1 = ffn_drop(x_drop)
    out_eval_2 = ffn_drop(x_drop)
    same = torch.allclose(out_eval_1, out_eval_2, atol=1e-6)
    print(f"   Eval mode : two forward passes same?   {same}")
    assert same, "Outputs should be identical in eval mode (no dropout)!"
    print("   OK: Dropout behaves correctly in train/eval modes.")

    # ── Test 5: ReLU clipping (negative inputs) ──────────────────────────
    print("\n5. ReLU activation test...")
    ffn_relu = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=0.0)

    # Force all-negative input to linear1 by setting weights/bias so
    # linear1 output is negative, then verify ReLU clips to zero.
    with torch.no_grad():
        # Set W₁ to identity-like (first d_model rows) and large negative bias
        ffn_relu.linear1.weight.zero_()
        ffn_relu.linear1.bias.fill_(-100.0)  # ensure output is very negative
        ffn_relu.linear2.weight.fill_(1.0)
        ffn_relu.linear2.bias.zero_()

    x_relu = torch.randn(batch, seq_len, d_model)
    out_relu = ffn_relu(x_relu)

    # After ReLU all intermediates are 0 → output = 0·W₂ + b₂ = 0
    assert torch.allclose(out_relu, torch.zeros_like(out_relu), atol=1e-5), \
        f"ReLU should clip all negatives to 0, making output zero. Got max={out_relu.abs().max().item()}"
    print(f"   Output max abs value: {out_relu.abs().max().item():.6f}")
    print("   OK: ReLU correctly clips negative intermediate values.")

    # ── Test 6: Alias check ──────────────────────────────────────────────
    print("\n6. Alias test...")
    assert FFN is FeedForwardNetwork, "FFN should be an alias for FeedForwardNetwork"
    ffn_alias = FFN(d_model=16, d_ff=64)
    assert isinstance(ffn_alias, FeedForwardNetwork)
    print("   OK: FFN is a valid alias for FeedForwardNetwork.")

    print("\n" + "=" * 60)
    print("All Feed-Forward Network tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
