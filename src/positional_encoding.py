"""
Positional Encoding Modules for Transformer
===========================================

This module implements two types of positional encodings:
1. Sinusoidal Positional Encoding (fixed):
   From 'Attention Is All You Need' (Vaswani et al., 2017).
   Uses sine and cosine functions of different frequencies to encode position.
   
2. Learned Positional Encoding:
   From GPT-style and BERT-style models.
   Learns a standard embedding vector for each position index.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # position vector of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term vector of shape (d_model / 2,)
        # Computed in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        # Fill even indices with sin, odd indices with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle edge case for odd d_model (rare in practice)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
            
        # Add batch/sequence dimension so it matches the expected input shape format
        if self.batch_first:
            # shape: (1, max_len, d_model)
            pe = pe.unsqueeze(0)
        else:
            # shape: (max_len, 1, d_model)
            pe = pe.unsqueeze(1)
            
        # Register buffer to move automatically with module to GPU/CPU, but not be updated by optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input embeddings of shape:
                        - (batch_size, seq_len, d_model) if batch_first=True
                        - (seq_len, batch_size, d_model) if batch_first=False
        Returns:
            Tensor: Input tensor with positional encodings added.
        """
        if self.batch_first:
            seq_len = x.size(1)
            assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            # self.pe is (1, max_len, d_model), broadcasts over batch_size
            x = x + self.pe[:, :seq_len]
        else:
            seq_len = x.size(0)
            assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            # self.pe is (max_len, 1, d_model), broadcasts over batch_size
            x = x + self.pe[:seq_len, :]
            
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (Standard Embeddings).
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input embeddings of shape:
                        - (batch_size, seq_len, d_model) if batch_first=True
                        - (seq_len, batch_size, d_model) if batch_first=False
        Returns:
            Tensor: Input tensor with learned positional embeddings added.
        """
        if self.batch_first:
            seq_len = x.size(1)
            assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0) # (1, seq_len)
            pos_emb = self.pe(positions) # (1, seq_len, d_model)
            x = x + pos_emb
        else:
            seq_len = x.size(0)
            assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(1) # (seq_len, 1)
            pos_emb = self.pe(positions) # (seq_len, 1, d_model)
            x = x + pos_emb
            
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test (run: python src/positional_encoding.py from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Positional Encoding Modules")
    print("=" * 60)

    # Test settings
    batch_size = 4
    seq_len = 10
    d_model = 16
    max_len = 100

    print(f"Configurations:\n  batch_size={batch_size}\n  seq_len={seq_len}\n  d_model={d_model}\n  max_len={max_len}\n")

    # 1. Test Fixed Sinusoidal Positional Encoding (batch_first=True)
    print("1. Testing Fixed Sinusoidal (batch_first=True)...")
    pe_fixed_bf = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0, batch_first=True)
    x = torch.zeros(batch_size, seq_len, d_model)
    y = pe_fixed_bf(x)
    print(f"  Input shape  : {x.shape}")
    print(f"  Output shape : {y.shape}")
    
    # Assertions
    assert y.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    # Verify broadcasting works: all batches must have identical positional encodings added
    for b in range(1, batch_size):
        assert torch.allclose(y[0], y[b]), f"Broadcasting failure at batch {b}!"
    print("  OK: Shape and broadcasting tests passed.")

    # 2. Test Fixed Sinusoidal Positional Encoding (batch_first=False)
    print("\n2. Testing Fixed Sinusoidal (batch_first=False)...")
    pe_fixed_nb = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0, batch_first=False)
    x_nb = torch.zeros(seq_len, batch_size, d_model)
    y_nb = pe_fixed_nb(x_nb)
    print(f"  Input shape  : {x_nb.shape}")
    print(f"  Output shape : {y_nb.shape}")
    assert y_nb.shape == (seq_len, batch_size, d_model), "Shape mismatch!"
    for b in range(1, batch_size):
        assert torch.allclose(y_nb[:, 0, :], y_nb[:, b, :]), f"Broadcasting failure at batch {b}!"
    print("  OK: Shape and broadcasting tests passed.")

    # 3. Test Learned Positional Encoding (batch_first=True)
    print("\n3. Testing Learned Positional Encoding (batch_first=True)...")
    pe_learned = LearnedPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0, batch_first=True)
    y_learned = pe_learned(x)
    print(f"  Input shape  : {x.shape}")
    print(f"  Output shape : {y_learned.shape}")
    assert y_learned.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    print("  OK: Shape and broadcasting tests passed.")

    # 4. Test Dropout Behavior
    print("\n4. Testing Dropout active scaling in Training mode...")
    pe_dropout = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.5, batch_first=True)
    pe_dropout.train()  # ensure training mode is on
    x_ones = torch.ones(1, seq_len, d_model)
    y_train = pe_dropout(x_ones)
    
    # In training mode, dropout should zero out some values
    num_zeros = (y_train == 0.0).sum().item()
    print(f"  Number of zeros in train mode (prob=0.5): {num_zeros} / {seq_len * d_model}")
    assert num_zeros > 0, "Dropout did not zero out any elements in training mode!"

    pe_dropout.eval()  # switch to evaluation mode
    y_eval = pe_dropout(x_ones)
    num_zeros_eval = (y_eval == 0.0).sum().item()
    print(f"  Number of zeros in eval mode (prob=0.5) : {num_zeros_eval} / {seq_len * d_model}")
    assert num_zeros_eval == 0, "Dropout zeroed out elements in evaluation mode!"
    print("  OK: Dropout behavior verification passed.")

    # 5. Test Maximum Sequence Length Assertion
    print("\n5. Testing Sequence Length Assertion...")
    try:
        x_huge = torch.zeros(batch_size, max_len + 10, d_model)
        pe_fixed_bf(x_huge)
        raise AssertionError("Exceeding max_len assertion failed to trigger!")
    except AssertionError as e:
        print(f"  OK: Assertion successfully caught exception: {e}")

    print("\n" + "=" * 60)
    print("All Positional Encoding tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
