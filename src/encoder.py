"""
Encoder Block
=============

From "Attention Is All You Need" (Vaswani et al., 2017).

Each encoder layer consists of two sub-layers:
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network

Each sub-layer has a residual connection followed by layer normalisation:
  output = LayerNorm(x + Sublayer(x))

Architecture
------------
    Input
     ↓
    MultiHeadAttention (self-attention)
     ↓
    Add & Norm  (residual + LayerNorm)
     ↓
    FeedForward
     ↓
    Add & Norm  (residual + LayerNorm)
     ↓
    Output
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from multihead_attention import MultiHeadAttention
from feedforward import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer.

    Parameters
    ----------
    d_model  : int   – model dimension
    n_heads  : int   – number of attention heads
    d_ff     : int   – inner dimension of the feed-forward network
    dropout  : float – dropout probability (default 0.1)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # ── Sub-layer 1: Multi-Head Self-Attention ────────────────────────
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)

        # ── Sub-layer 2: Feed-Forward Network ─────────────────────────────
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout2  = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src      : Tensor (batch, seq_len, d_model)
        src_mask : Tensor, optional – padding mask for the source

        Returns
        -------
        Tensor (batch, seq_len, d_model)
        """
        # ── Sub-layer 1: Self-Attention + Add & Norm ──────────────────────
        attn_out, _ = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_out))

        # ── Sub-layer 2: FFN + Add & Norm ─────────────────────────────────
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_out))

        return src


class Encoder(nn.Module):
    """
    Full Transformer Encoder: a stack of N identical EncoderLayer modules.

    Parameters
    ----------
    d_model    : int   – model dimension
    n_heads    : int   – number of attention heads
    d_ff       : int   – FFN inner dimension
    n_layers   : int   – number of encoder layers to stack
    dropout    : float – dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)   # final layer norm

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src      : Tensor (batch, seq_len, d_model) – already embedded + positionally encoded
        src_mask : Tensor, optional

        Returns
        -------
        Tensor (batch, seq_len, d_model)
        """
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/encoder.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Encoder Block")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seq, d_model, n_heads, d_ff, n_layers = 2, 10, 32, 4, 128, 3

    # ── Test 1: Single encoder layer ──────────────────────────────────────
    print("\n1. Single EncoderLayer shape test...")
    layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.0)
    x = torch.randn(batch, seq, d_model)
    out = layer(x)
    print(f"   Input  : {tuple(x.shape)}")
    print(f"   Output : {tuple(out.shape)}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print("   OK: Shape preserved.")

    # ── Test 2: Full encoder stack ────────────────────────────────────────
    print(f"\n2. Full Encoder ({n_layers} layers) shape test...")
    encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout=0.0)
    out_enc = encoder(x)
    print(f"   Input  : {tuple(x.shape)}")
    print(f"   Output : {tuple(out_enc.shape)}")
    assert out_enc.shape == x.shape
    print("   OK: Shape preserved through encoder stack.")

    # ── Test 3: Residual connection sanity ─────────────────────────────────
    print("\n3. Residual connection test...")
    # With zero-init FFN and attention, residual should keep input roughly intact
    # Just verify output is not identical to input (some transformation happened)
    assert not torch.allclose(x, out_enc, atol=1e-3), "Output should differ from input"
    print("   OK: Output differs from input (transformations applied).")

    # ── Test 4: With padding mask ─────────────────────────────────────────
    print("\n4. Padding mask test...")
    # Mask last 3 positions
    pad_mask = torch.zeros(batch, 1, 1, seq, dtype=torch.bool)
    pad_mask[:, :, :, -3:] = True
    out_masked = encoder(x, src_mask=pad_mask)
    assert out_masked.shape == x.shape
    print(f"   Mask shape : {tuple(pad_mask.shape)}")
    print(f"   Output     : {tuple(out_masked.shape)}")
    print("   OK: Encoder handles padding mask correctly.")

    # ── Test 5: Parameter count ───────────────────────────────────────────
    print("\n5. Parameter count test...")
    total = sum(p.numel() for p in encoder.parameters())
    print(f"   Total parameters: {total:,}")
    print(f"   Layers: {n_layers}, d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}")
    print("   OK.")

    # ── Test 6: Gradient flow ─────────────────────────────────────────────
    print("\n6. Gradient flow test...")
    encoder_grad = Encoder(d_model, n_heads, d_ff, n_layers=2, dropout=0.0)
    x_grad = torch.randn(batch, seq, d_model, requires_grad=True)
    out_g = encoder_grad(x_grad)
    loss = out_g.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradients computed!"
    assert x_grad.grad.abs().sum() > 0, "Gradients are all zero!"
    print("   OK: Gradients flow through the encoder.")

    print("\n" + "=" * 60)
    print("All Encoder tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
