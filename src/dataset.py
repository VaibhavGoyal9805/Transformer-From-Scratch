"""
Dataset & DataLoaders for Autoregressive Language Modelling
============================================================

Provides a PyTorch ``Dataset`` that yields (input, target) pairs for
next-token prediction on the Tiny Shakespeare corpus.

Each sample is a contiguous window of ``seq_len`` token ids (input)
paired with the same window shifted one position to the right (target).

    x = tokens[i   : i + seq_len]
    y = tokens[i+1 : i + seq_len + 1]

The helper ``get_dataloaders`` handles the full pipeline:
    1. Build / load a ``WordTokenizer``
    2. Encode the entire corpus to a 1-D tensor
    3. Split into train / validation sets
    4. Wrap each split in a ``TextDataset`` + ``DataLoader``
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from tokenizer import WordTokenizer, DATA_PATH, TOKENIZER_SAVE


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Map-style dataset for autoregressive language modelling.

    Every index ``i`` returns an ``(x, y)`` pair where ``x`` is a window of
    ``seq_len`` contiguous token ids and ``y`` is the same window shifted
    right by one position (the prediction target).

    Parameters
    ----------
    data : torch.Tensor
        1-D tensor of token ids representing the entire (sub-)corpus.
    seq_len : int
        Context window length (number of tokens per input sample).
    """

    def __init__(self, data: torch.Tensor, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        # We need seq_len tokens for x and one more for the last target token
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]           # input
        y = self.data[idx + 1 : idx + self.seq_len + 1]   # target (shifted by 1)
        return x, y


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    file_path: str = DATA_PATH,
    seq_len: int = 64,
    batch_size: int = 32,
    split_ratio: float = 0.9,
    min_freq: int = 1,
) -> Tuple[WordTokenizer, DataLoader, DataLoader]:
    """
    Build a tokenizer, encode the corpus, split into train / val, and
    return ready-to-use ``DataLoader`` objects.

    Parameters
    ----------
    file_path : str
        Path to the raw text file (default: ``data/tiny_shakespeare.txt``).
    seq_len : int
        Context window length for each sample (default 64).
    batch_size : int
        Mini-batch size for both loaders (default 32).
    split_ratio : float
        Fraction of the data used for training (default 0.9).
    min_freq : int
        Minimum word frequency to include in vocabulary (default 1).

    Returns
    -------
    tokenizer : WordTokenizer
        The fitted tokenizer (needed later for decoding).
    train_loader : DataLoader
        Training data loader (shuffled).
    val_loader : DataLoader
        Validation data loader (sequential).
    """
    # ── 1. Build tokenizer ────────────────────────────────────────────────
    tokenizer = WordTokenizer(file_path=file_path, min_freq=min_freq)

    # ── 2. Encode the full corpus to a 1-D tensor ────────────────────────
    data = tokenizer.encode_corpus_tensor()  # shape: (corpus_length,)

    # ── 3. Train / validation split ──────────────────────────────────────
    split_idx = int(split_ratio * len(data))
    train_data = data[:split_idx]
    val_data   = data[split_idx:]

    # ── 4. Wrap in TextDataset ───────────────────────────────────────────
    train_dataset = TextDataset(train_data, seq_len)
    val_dataset   = TextDataset(val_data,   seq_len)

    # ── 5. Create DataLoaders ────────────────────────────────────────────
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return tokenizer, train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/dataset.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Dataset & DataLoaders")
    print("=" * 60)

    # ── 1. Build dataloaders ──────────────────────────────────────────────
    seq_len    = 64
    batch_size = 32
    tokenizer, train_loader, val_loader = get_dataloaders(
        seq_len=seq_len, batch_size=batch_size,
    )

    # ── 2. Print tokenizer info & dataset sizes ──────────────────────────
    print(f"\n   Tokenizer       : {tokenizer}")
    print(f"   Vocab size      : {tokenizer.vocab_size}")
    print(f"   Train samples   : {len(train_loader.dataset)}")
    print(f"   Val   samples   : {len(val_loader.dataset)}")
    print(f"   Train batches   : {len(train_loader)}")
    print(f"   Val   batches   : {len(val_loader)}")

    # ── 3. Fetch one batch and print shapes ──────────────────────────────
    print("\n3. Fetching one training batch...")
    x_batch, y_batch = next(iter(train_loader))
    print(f"   x_batch shape : {tuple(x_batch.shape)}")
    print(f"   y_batch shape : {tuple(y_batch.shape)}")

    assert x_batch.shape == (batch_size, seq_len), \
        f"Expected x shape {(batch_size, seq_len)}, got {tuple(x_batch.shape)}"
    assert y_batch.shape == (batch_size, seq_len), \
        f"Expected y shape {(batch_size, seq_len)}, got {tuple(y_batch.shape)}"
    print("   OK: Batch shapes are correct.")

    # ── 4. Verify that y == x shifted by 1 position ─────────────────────
    print("\n4. Verifying target is input shifted by 1...")
    # Grab a single sample directly from the dataset (not shuffled)
    dataset = train_loader.dataset
    x_sample, y_sample = dataset[0]

    # y should equal the data slice one position ahead of x
    assert torch.equal(x_sample[1:], y_sample[:-1]), \
        "Overlap check failed: x[1:] should equal y[:-1]"
    print(f"   x[:8] = {x_sample[:8].tolist()}")
    print(f"   y[:8] = {y_sample[:8].tolist()}")
    print("   OK: y is x shifted by one token position.")

    # ── 5. Quick validation loader check ─────────────────────────────────
    print("\n5. Fetching one validation batch...")
    x_val, y_val = next(iter(val_loader))
    print(f"   x_val shape : {tuple(x_val.shape)}")
    print(f"   y_val shape : {tuple(y_val.shape)}")
    print("   OK: Validation loader works.")

    print("\n" + "=" * 60)
    print("All Dataset & DataLoader tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
