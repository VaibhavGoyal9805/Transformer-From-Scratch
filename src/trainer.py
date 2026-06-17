"""
Training Loop
==============

Training pipeline for the TransformerLM language model.

Features
--------
• Cross-entropy loss with label smoothing
• Adam optimiser with Noam learning rate schedule
• Gradient clipping
• Train/validation loop with perplexity tracking
• Checkpoint saving & loading
• Training curves (loss + perplexity) via Matplotlib
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
from typing import Optional

# ── Path helpers ──────────────────────────────────────────────────────────────
_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
CHECKPOINT_DIR = os.path.join(_PROJECT_DIR, "checkpoints")

# Add src to path so we can import siblings
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from transformer import TransformerLM
from dataset import get_dataloaders


# ──────────────────────────────────────────────────────────────────────────────
# Noam Learning Rate Schedule
# ──────────────────────────────────────────────────────────────────────────────

class NoamScheduler:
    """
    Implements the learning rate schedule from the original Transformer paper:

        lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    Parameters
    ----------
    optimiser   : torch.optim.Optimizer
    d_model     : int
    warmup_steps: int (default 4000)
    """

    def __init__(self, optimiser, d_model: int, warmup_steps: int = 4000):
        self.optimiser    = optimiser
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimiser.param_groups:
            pg['lr'] = lr

    def _get_lr(self):
        step = max(self._step, 1)
        return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def get_last_lr(self):
        return self._get_lr()


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimiser, scheduler, device, clip_grad=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward
        logits = model(x_batch)                        # (batch, seq_len, vocab_size)
        loss = criterion(
            logits.view(-1, logits.size(-1)),           # (batch*seq_len, vocab_size)
            y_batch.view(-1),                           # (batch*seq_len,)
        )

        # Backward
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on val set. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y_batch.view(-1),
        )
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int       = 15,
    d_model: int      = 256,
    n_heads: int      = 8,
    d_ff: int         = 1024,
    n_layers: int     = 6,
    dropout: float    = 0.1,
    seq_len: int      = 128,
    batch_size: int   = 32,
    warmup_steps: int = 500,
    clip_grad: float  = 1.0,
    save_every: int   = 5,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    device_str: str   = "auto",
):
    """
    Full training pipeline.

    Returns
    -------
    model, tokenizer, history (dict with train_loss, val_loss, perplexity lists)
    """
    # ── Device ────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"[Trainer] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("[Trainer] Loading data...")
    tokenizer, train_loader, val_loader = get_dataloaders(
        seq_len=seq_len, batch_size=batch_size,
    )
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_id
    print(f"[Trainer] Vocab size: {vocab_size:,}")
    print(f"[Trainer] Train batches: {len(train_loader):,}  Val batches: {len(val_loader):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        max_len=seq_len + 100,
        pad_id=pad_id,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Trainer] Model parameters: {total_params:,}")

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    scheduler = NoamScheduler(optimiser, d_model, warmup_steps)

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "perplexity": []}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n[Trainer] Starting training for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, scheduler, device, clip_grad)
        val_loss   = evaluate(model, val_loader, criterion, device)
        ppl        = math.exp(min(val_loss, 20))   # cap to avoid overflow

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["perplexity"].append(ppl)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()
        print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"Train Loss {train_loss:.4f} │ "
            f"Val Loss {val_loss:.4f} │ "
            f"PPL {ppl:.2f} │ "
            f"LR {lr:.2e} │ "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"transformer_lm_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  [Saved] {ckpt_path}")

    print("-" * 60)
    print("[Trainer] Training complete.")

    return model, tokenizer, history


# ──────────────────────────────────────────────────────────────────────────────
# Plot training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path=None):
    """Plot training/validation loss and perplexity."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Trainer] matplotlib not available, skipping plots.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker='o', markersize=3)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker='s', markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Perplexity
    ax2.plot(epochs, history["perplexity"], label="Perplexity", marker='o', markersize=3, color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Validation Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or os.path.join(CHECKPOINT_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"[Trainer] Curves saved -> {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test: train for 2 epochs with small model
    print("=" * 60)
    print("Trainer Smoke Test (2 epochs, small model)")
    print("=" * 60)

    model, tokenizer, history = train(
        epochs=2,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.1,
        seq_len=32,
        batch_size=32,
        warmup_steps=100,
        save_every=2,
    )

    print(f"\n  Final train loss : {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss   : {history['val_loss'][-1]:.4f}")
    print(f"  Final perplexity : {history['perplexity'][-1]:.2f}")

    # Plot
    plot_training_curves(history)

    print("\n" + "=" * 60)
    print("Trainer smoke test complete!")
    print("=" * 60)
