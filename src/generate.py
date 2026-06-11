"""
Text Generation
================

Multiple decoding strategies for autoregressive text generation
with the TransformerLM model:

1. **Greedy Search**      – always pick the most likely next token
2. **Top-k Sampling**     – sample from the k most likely tokens
3. **Temperature Sampling** – scale logits by temperature before sampling

Usage
-----
    from generate import generate_text
    text = generate_text(model, tokenizer, prompt="To be", max_tokens=50, method="top_k")
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional

# ── Path helpers ──────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from transformer import TransformerLM


# ──────────────────────────────────────────────────────────────────────────────
# Core generation function
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer,
    prompt: str = "",
    max_tokens: int = 100,
    method: str = "greedy",
    temperature: float = 0.8,
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text autoregressively.

    Parameters
    ----------
    model      : TransformerLM – the trained decoder-only model
    tokenizer  : WordTokenizer – for encode/decode
    prompt     : str – starting text
    max_tokens : int – how many new tokens to generate
    method     : str – "greedy", "top_k", or "temperature"
    temperature: float – temperature for sampling (lower = more greedy)
    top_k      : int – number of top tokens to sample from
    device     : torch.device – if None, inferred from model

    Returns
    -------
    str – the generated text (prompt + generated)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    if prompt:
        token_ids = tokenizer.encode(prompt)
    else:
        # Start with a random token or SOS
        token_ids = [tokenizer.sos_id] if tokenizer.sos_id is not None else [1]

    generated = list(token_ids)

    for _ in range(max_tokens):
        # Prepare input — truncate to model's max_len if needed
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        # Get logits for the last position
        logits = model(input_ids)                # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]           # (vocab_size,)

        # ── Decoding strategy ─────────────────────────────────────────────
        if method == "greedy":
            next_token = greedy_decode(next_logits)
        elif method == "top_k":
            next_token = top_k_sample(next_logits, k=top_k, temperature=temperature)
        elif method == "temperature":
            next_token = temperature_sample(next_logits, temperature=temperature)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'greedy', 'top_k', or 'temperature'.")

        generated.append(next_token)

        # Stop at EOS
        if tokenizer.eos_id is not None and next_token == tokenizer.eos_id:
            break

    return tokenizer.decode(generated, skip_special=True)


# ──────────────────────────────────────────────────────────────────────────────
# Decoding strategies
# ──────────────────────────────────────────────────────────────────────────────

def greedy_decode(logits: torch.Tensor) -> int:
    """Pick the token with the highest probability."""
    return logits.argmax(dim=-1).item()


def temperature_sample(logits: torch.Tensor, temperature: float = 0.8) -> int:
    """
    Scale logits by temperature, then sample from the resulting distribution.

    Lower temperature → more deterministic (peakier distribution).
    Higher temperature → more random (flatter distribution).
    """
    if temperature <= 0:
        return greedy_decode(logits)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def top_k_sample(logits: torch.Tensor, k: int = 10, temperature: float = 0.8) -> int:
    """
    Keep only the top-k most probable tokens, zero out the rest,
    apply temperature, then sample.
    """
    if temperature <= 0:
        return greedy_decode(logits)

    # Get top-k values and indices
    top_values, top_indices = torch.topk(logits, k)

    # Apply temperature and convert to probabilities
    probs = F.softmax(top_values / temperature, dim=-1)

    # Sample from the top-k distribution
    sampled_idx = torch.multinomial(probs, num_samples=1).item()
    return top_indices[sampled_idx].item()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run: python src/generate.py  from project root)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Testing Text Generation")
    print("=" * 60)

    torch.manual_seed(42)

    # Build a tiny untrained model for shape/API testing
    from tokenizer import WordTokenizer, DATA_PATH

    print("\n[Setup] Building tokenizer & model...")
    tokenizer = WordTokenizer(file_path=DATA_PATH, min_freq=1)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_layers=2,
        dropout=0.0,
        max_len=200,
        pad_id=tokenizer.pad_id,
    )
    model.eval()

    # ── Test 1: Greedy generation ─────────────────────────────────────────
    print("\n1. Greedy generation (untrained model)...")
    text_greedy = generate_text(model, tokenizer, prompt="To be", max_tokens=20, method="greedy")
    print(f"   Output: {text_greedy[:80]}...")
    assert len(text_greedy) > 0
    print("   OK: Greedy generation works.")

    # ── Test 2: Top-k sampling ────────────────────────────────────────────
    print("\n2. Top-k sampling (k=10)...")
    text_topk = generate_text(model, tokenizer, prompt="To be", max_tokens=20, method="top_k", top_k=10)
    print(f"   Output: {text_topk[:80]}...")
    assert len(text_topk) > 0
    print("   OK: Top-k sampling works.")

    # ── Test 3: Temperature sampling ──────────────────────────────────────
    print("\n3. Temperature sampling (T=0.8)...")
    text_temp = generate_text(model, tokenizer, prompt="To be", max_tokens=20, method="temperature", temperature=0.8)
    print(f"   Output: {text_temp[:80]}...")
    assert len(text_temp) > 0
    print("   OK: Temperature sampling works.")

    # ── Test 4: Low temperature (near greedy) ─────────────────────────────
    print("\n4. Low temperature (T=0.1, should be near-deterministic)...")
    t1 = generate_text(model, tokenizer, prompt="To be", max_tokens=10, method="temperature", temperature=0.1)
    t2 = generate_text(model, tokenizer, prompt="To be", max_tokens=10, method="temperature", temperature=0.1)
    print(f"   Run 1: {t1[:60]}")
    print(f"   Run 2: {t2[:60]}")
    print("   OK: Low temperature produces output.")

    # ── Test 5: Empty prompt ──────────────────────────────────────────────
    print("\n5. Empty prompt generation...")
    text_empty = generate_text(model, tokenizer, prompt="", max_tokens=15, method="greedy")
    print(f"   Output: {text_empty[:60]}...")
    assert len(text_empty) > 0
    print("   OK: Empty prompt works.")

    print("\n" + "=" * 60)
    print("All Text Generation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
