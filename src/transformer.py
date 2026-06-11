"""
Full Transformer
=================

From "Attention Is All You Need" (Vaswani et al., 2017).

This module provides two models:

1. **Transformer** – full encoder-decoder for sequence-to-sequence tasks.
2. **TransformerLM** – decoder-only language model for autoregressive generation
   (used for training on Tiny Shakespeare / WikiText-2).

Architecture (Full Transformer)
-------------------------------
    Source → Embedding → Positional Encoding → Encoder Stack → memory
    Target → Embedding → Positional Encoding → Decoder Stack(memory) → Linear → logits

Architecture (Decoder-Only LM)
------------------------------
    Tokens → Embedding → Positional Encoding → Decoder Stack(self-attn only) → Linear → logits
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_causal_mask(sz: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate an upper-triangular boolean causal mask.

    Returns shape (1, 1, sz, sz) — True means IGNORE.
    Broadcasts over (batch, n_heads, ...).
    """
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, sz, sz)


def generate_padding_mask(
    x: torch.Tensor, pad_id: int = 0
) -> torch.Tensor:
    """
    Generate a padding mask from token ids.

    Parameters
    ----------
    x      : Tensor (batch, seq_len)  – token ids
    pad_id : int – the padding token id

    Returns
    -------
    Tensor (batch, 1, 1, seq_len) – True where padded
    """
    return (x == pad_id).unsqueeze(1).unsqueeze(2)   # (batch, 1, 1, seq_len)


# ──────────────────────────────────────────────────────────────────────────────
# Full Encoder-Decoder Transformer
# ──────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer for sequence-to-sequence tasks.

    Parameters
    ----------
    src_vocab_size : int
    tgt_vocab_size : int
    d_model        : int   (default 512)
    n_heads        : int   (default 8)
    d_ff           : int   (default 2048)
    n_layers       : int   (default 6)
    dropout        : float (default 0.1)
    max_len        : int   (default 5000) – max sequence length for PE
    pad_id         : int   (default 0)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id  = pad_id

        # ── Embeddings ────────────────────────────────────────────────────
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding  = PositionalEncoding(d_model, max_len, dropout, batch_first=True)

        # ── Encoder & Decoder stacks ──────────────────────────────────────
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout)

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : Tensor (batch, src_len) – source token ids
        tgt : Tensor (batch, tgt_len) – target token ids
        src_mask, tgt_mask, memory_mask : optional masks

        Returns
        -------
        logits : Tensor (batch, tgt_len, tgt_vocab_size)
        """
        # ── Encode source ─────────────────────────────────────────────────
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        memory  = self.encoder(src_emb, src_mask)

        # ── Decode target ─────────────────────────────────────────────────
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

        # ── Project to vocabulary ─────────────────────────────────────────
        logits = self.output_proj(dec_out)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Decoder-Only Transformer Language Model
# ──────────────────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer for autoregressive language modelling.

    Used for training on Tiny Shakespeare and WikiText-2.
    Each decoder layer uses ONLY self-attention (no cross-attention).

    Parameters
    ----------
    vocab_size : int
    d_model    : int   (default 256)
    n_heads    : int   (default 4)
    d_ff       : int   (default 1024)
    n_layers   : int   (default 4)
    dropout    : float (default 0.1)
    max_len    : int   (default 5000)
    pad_id     : int   (default 0)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.pad_id     = pad_id
        self.vocab_size = vocab_size

        # ── Token embedding + positional encoding ─────────────────────────
        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout, batch_first=True)

        # ── Decoder stack ─────────────────────────────────────────────────
        # We reuse the Decoder class.  For decoder-only LM, the "memory"
        # input to cross-attention will simply be a dummy zero tensor.
        # However, a cleaner approach: use the Encoder stack (which has
        # only self-attention) with causal masking.
        self.layers = nn.ModuleList([
            _DecoderOnlyLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Optionally tie embedding and output weights
        # self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor (batch, seq_len) – token ids
        mask : Tensor, optional – if None, auto-generates causal mask

        Returns
        -------
        logits : Tensor (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)

        # ── Auto-generate causal mask ─────────────────────────────────────
        if mask is None:
            mask = generate_causal_mask(seq_len, device=x.device)

        # ── Embed + positional encode ─────────────────────────────────────
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)

        # ── Decoder layers (self-attention only) ──────────────────────────
        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)

        # ── Project to vocabulary ─────────────────────────────────────────
        logits = self.output_proj(h)
        return logits


class _DecoderOnlyLayer(nn.Module):
    """
    Single decoder-only layer: masked self-attention + FFN.
    (No cross-attention — this is the GPT-style architecture.)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        from multihead_attention import MultiHeadAttention
        from feedforward import FeedForwardNetwork

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)

        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout2  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/transformer.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Full Transformer & TransformerLM")
    print("=" * 60)

    torch.manual_seed(42)
    batch = 2

    # ── Test 1: Full Encoder-Decoder Transformer ──────────────────────────
    print("\n1. Full Transformer (encoder-decoder)...")
    src_vocab, tgt_vocab = 100, 120
    d_model, n_heads, d_ff, n_layers = 32, 4, 128, 2

    model = Transformer(
        src_vocab, tgt_vocab, d_model, n_heads, d_ff, n_layers,
        dropout=0.0, max_len=100, pad_id=0,
    )

    src = torch.randint(1, src_vocab, (batch, 10))
    tgt = torch.randint(1, tgt_vocab, (batch, 8))

    # Auto masks
    src_mask = generate_padding_mask(src, pad_id=0)
    tgt_mask = generate_causal_mask(tgt.size(1), device=tgt.device)

    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print(f"   src shape    : {tuple(src.shape)}")
    print(f"   tgt shape    : {tuple(tgt.shape)}")
    print(f"   logits shape : {tuple(logits.shape)}")
    assert logits.shape == (batch, 8, tgt_vocab)
    print("   OK: Encoder-decoder shape correct.")

    # ── Test 2: TransformerLM (decoder-only) ──────────────────────────────
    print("\n2. TransformerLM (decoder-only LM)...")
    vocab_size = 200
    lm = TransformerLM(
        vocab_size, d_model=32, n_heads=4, d_ff=128, n_layers=2,
        dropout=0.0, max_len=100, pad_id=0,
    )
    x = torch.randint(1, vocab_size, (batch, 15))
    logits_lm = lm(x)
    print(f"   Input shape  : {tuple(x.shape)}")
    print(f"   Logits shape : {tuple(logits_lm.shape)}")
    assert logits_lm.shape == (batch, 15, vocab_size)
    print("   OK: Decoder-only LM shape correct.")

    # ── Test 3: Gradient flow ─────────────────────────────────────────────
    print("\n3. Gradient flow test (TransformerLM)...")
    x_g = torch.randint(1, vocab_size, (batch, 10))
    logits_g = lm(x_g)
    loss = logits_g.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in lm.parameters() if p.requires_grad
    )
    assert grad_ok, "Some parameters have no gradients!"
    print("   OK: All parameters receive gradients.")

    # ── Test 4: Causal mask helper ────────────────────────────────────────
    print("\n4. Causal mask helper...")
    cm = generate_causal_mask(5)
    print(f"   Shape: {tuple(cm.shape)}")
    print(f"   Mask:\n{cm.squeeze()}")
    assert cm.shape == (1, 1, 5, 5)
    print("   OK.")

    # ── Test 5: Padding mask helper ───────────────────────────────────────
    print("\n5. Padding mask helper...")
    tok_ids = torch.tensor([[5, 3, 0, 0], [7, 2, 4, 0]])
    pm = generate_padding_mask(tok_ids, pad_id=0)
    print(f"   Token ids: {tok_ids.tolist()}")
    print(f"   Pad mask : {pm.squeeze().tolist()}")
    assert pm.shape == (2, 1, 1, 4)
    print("   OK.")

    # ── Test 6: Parameter count summary ───────────────────────────────────
    print("\n6. Parameter count summary...")
    total_tf = sum(p.numel() for p in model.parameters())
    total_lm = sum(p.numel() for p in lm.parameters())
    print(f"   Transformer (enc-dec) : {total_tf:,} parameters")
    print(f"   TransformerLM (dec)   : {total_lm:,} parameters")
    print("   OK.")

    print("\n" + "=" * 60)
    print("All Transformer tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
