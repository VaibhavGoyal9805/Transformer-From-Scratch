# 🎭 Transformer From Scratch

A **complete Transformer architecture** implemented entirely from scratch in PyTorch — no Hugging Face, no pre-built layers. Every component is hand-coded following the original *"Attention Is All You Need"* paper (Vaswani et al., 2017).

This project culminates in an **11.5-million parameter** Language Model trained on Tiny Shakespeare, complete with a beautiful, interactive Flask Web UI for generating text.

![Web Interface](https://img.shields.io/badge/Web_UI-Included-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red?style=for-the-badge&logo=pytorch)
![Parameters](https://img.shields.io/badge/Parameters-11.5M-brightgreen?style=for-the-badge)

## ✨ Features

- **From-Scratch Implementation:** Everything from Multi-Head Attention to Sinusoidal Positional Encoding is built using raw PyTorch primitives.
- **Optimized Architecture:** Scaled up to 11.5M parameters (`d_model=256`, `n_heads=8`, `n_layers=6`) with AdamW, Weight Decay, and Label Smoothing.
- **Kaggle/Colab Ready:** Includes a Jupyter notebook (`notebooks/colab_training.ipynb`) configured for fast cloud-GPU training.
- **Interactive Web App:** A sleek, modern Flask interface to chat with your model and tweak decoding methods (`Greedy`, `Top-k`, `Temperature`).

## 📁 Architecture Breakdown

| Component | File | Status |
|-----------|------|--------|
| Word-level Tokenizer | `src/tokenizer.py` | ✅ |
| Dataset & DataLoaders | `src/dataset.py` | ✅ |
| Sinusoidal Positional Encoding | `src/positional_encoding.py` | ✅ |
| Scaled Dot-Product Attention | `src/attention.py` | ✅ |
| Multi-Head Attention | `src/multihead_attention.py` | ✅ |
| Feed-Forward Network | `src/feedforward.py` | ✅ |
| Encoder Block + Stack | `src/encoder.py` | ✅ |
| Decoder Block + Stack | `src/decoder.py` | ✅ |
| Full Transformer (Enc-Dec + LM) | `src/transformer.py` | ✅ |
| Training Pipeline + Noam Schedule | `src/trainer.py` | ✅ |
| Text Generation (Greedy/Top-k/Temp) | `src/generate.py` | ✅ |
| Web Interface | `app.py` | ✅ |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/VaibhavGoyal9805/Transformer-From-Scratch.git
cd Transformer-From-Scratch
pip install -r requirements.txt
```

### 2. Launch the Web App
If you already have a trained checkpoint (e.g., `checkpoints/transformer_lm_epoch15.pt`), you can immediately start the web interface:

```bash
python app.py
```
*Navigate to `http://localhost:5001` in your browser.*

### 3. Train the Model Locally

```python
from src.trainer import train

model, tokenizer, history = train(
    epochs=15,
    d_model=256,
    n_heads=8,
    d_ff=1024,
    n_layers=6,
    dropout=0.1,
    seq_len=128,
    batch_size=64, 
    warmup_steps=4000,
    weight_decay=0.01,
    label_smoothing=0.1
)
```

*(Note: Training the 11.5M parameter model locally takes several hours. We highly recommend using the provided Kaggle notebook for free cloud GPU training).*

### 4. Generate Text via Python

```python
import torch
from src.generate import generate_text

# Assuming model and tokenizer are already loaded
text = generate_text(
    model, 
    tokenizer, 
    prompt="this life", 
    max_tokens=50, 
    method="top_k", 
    top_k=40, 
    temperature=0.8
)
print(text)
```

## 🧠 What's Next?
- Try implementing Byte-Pair Encoding (BPE) to replace the word-level tokenizer.
- Scale up to a 50M+ parameter model on a larger dataset (like WikiText-103).
