"""
Decoder Block
=============

From "Attention Is All You Need" (Vaswani et al., 2017).

Each decoder layer has three sub-layers:
  1. Masked Multi-Head Self-Attention  (prevents attending to future tokens)
  2. Multi-Head Cross-Attention         (attends to encoder output)
  3. Position-wise Feed-Forward Network

Each sub-layer has a residual connection followed by layer normalisation:
  output = LayerNorm(x + Sublayer(x))

Architecture
------------
    Target Input
     ↓
    Masked Self-Attention
     ↓
    Add & Norm
     ↓
    Cross-Attention (Q=decoder, K=V=encoder output)
     ↓
    Add & Norm
     ↓
    FeedForward
     ↓
    Add & Norm
     ↓
    Output
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from multihead_attention import MultiHeadAttention
from feedforward import FeedForwardNetwork


class DecoderLayer(nn.Module):
    """
    A single Transformer decoder layer.

    Parameters
    ----------
    d_model  : int   – model dimension
    n_heads  : int   – number of attention heads
    d_ff     : int   – FFN inner dimension
    dropout  : float – dropout probability (default 0.1)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # ── Sub-layer 1: Masked Multi-Head Self-Attention ─────────────────
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)

        # ── Sub-layer 2: Multi-Head Cross-Attention ───────────────────────
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout2   = nn.Dropout(dropout)

        # ── Sub-layer 3: Feed-Forward Network ─────────────────────────────
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm3     = nn.LayerNorm(d_model)
        self.dropout3  = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt         : Tensor (batch, tgt_len, d_model) – decoder input
        memory      : Tensor (batch, src_len, d_model) – encoder output
        tgt_mask    : Tensor, optional – causal + padding mask for target
        memory_mask : Tensor, optional – padding mask for encoder output

        Returns
        -------
        Tensor (batch, tgt_len, d_model)
        """
        # ── Sub-layer 1: Masked Self-Attention + Add & Norm ───────────────
        attn_out, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        # ── Sub-layer 2: Cross-Attention + Add & Norm ─────────────────────
        # Query = decoder state, Key & Value = encoder output
        cross_out, _ = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_out))

        # ── Sub-layer 3: FFN + Add & Norm ─────────────────────────────────
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))

        return tgt


class Decoder(nn.Module):
    """
    Full Transformer Decoder: a stack of N identical DecoderLayer modules.

    Parameters
    ----------
    d_model    : int   – model dimension
    n_heads    : int   – number of attention heads
    d_ff       : int   – FFN inner dimension
    n_layers   : int   – number of decoder layers to stack
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
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)   # final layer norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt         : Tensor (batch, tgt_len, d_model)
        memory      : Tensor (batch, src_len, d_model)
        tgt_mask    : Tensor, optional
        memory_mask : Tensor, optional

        Returns
        -------
        Tensor (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/decoder.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Decoder Block")
    print("=" * 60)

    torch.manual_seed(42)
    batch, tgt_len, src_len = 2, 8, 12
    d_model, n_heads, d_ff, n_layers = 32, 4, 128, 3

    # Simulated encoder output and decoder input
    memory = torch.randn(batch, src_len, d_model)
    tgt    = torch.randn(batch, tgt_len, d_model)

    # ── Test 1: Single decoder layer ──────────────────────────────────────
    print("\n1. Single DecoderLayer shape test...")
    layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0.0)
    out = layer(tgt, memory)
    print(f"   Target input : {tuple(tgt.shape)}")
    print(f"   Memory       : {tuple(memory.shape)}")
    print(f"   Output       : {tuple(out.shape)}")
    assert out.shape == tgt.shape
    print("   OK: Shape preserved.")

    # ── Test 2: Full decoder stack ────────────────────────────────────────
    print(f"\n2. Full Decoder ({n_layers} layers) shape test...")
    decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout=0.0)
    out_dec = decoder(tgt, memory)
    print(f"   Output : {tuple(out_dec.shape)}")
    assert out_dec.shape == tgt.shape
    print("   OK: Shape preserved through decoder stack.")

    # ── Test 3: With causal mask ──────────────────────────────────────────
    print("\n3. Causal mask test...")
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)
    out_masked = decoder(tgt, memory, tgt_mask=causal_mask)
    assert out_masked.shape == tgt.shape
    print(f"   Causal mask shape : {tuple(causal_mask.shape)}")
    print(f"   Output            : {tuple(out_masked.shape)}")
    print("   OK: Decoder handles causal mask.")

    # ── Test 4: With memory padding mask ──────────────────────────────────
    print("\n4. Memory padding mask test...")
    mem_mask = torch.zeros(batch, 1, 1, src_len, dtype=torch.bool)
    mem_mask[:, :, :, -2:] = True  # mask last 2 encoder positions
    out_mem = decoder(tgt, memory, memory_mask=mem_mask)
    assert out_mem.shape == tgt.shape
    print(f"   Memory mask shape : {tuple(mem_mask.shape)}")
    print(f"   Output            : {tuple(out_mem.shape)}")
    print("   OK: Decoder handles memory mask.")

    # ── Test 5: Gradient flow ─────────────────────────────────────────────
    print("\n5. Gradient flow test...")
    decoder_g = Decoder(d_model, n_heads, d_ff, n_layers=2, dropout=0.0)
    tgt_g = torch.randn(batch, tgt_len, d_model, requires_grad=True)
    mem_g = torch.randn(batch, src_len, d_model, requires_grad=True)
    out_g = decoder_g(tgt_g, mem_g)
    loss = out_g.sum()
    loss.backward()
    assert tgt_g.grad is not None and tgt_g.grad.abs().sum() > 0
    assert mem_g.grad is not None and mem_g.grad.abs().sum() > 0
    print("   OK: Gradients flow through decoder (both tgt and memory).")

    # ── Test 6: Parameter count ───────────────────────────────────────────
    print("\n6. Parameter count...")
    total = sum(p.numel() for p in decoder.parameters())
    print(f"   Total parameters: {total:,}")
    print("   OK.")

    print("\n" + "=" * 60)
    print("All Decoder tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
