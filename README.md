# 🤖 Transformer From Scratch

A **complete Transformer architecture** implemented from scratch in PyTorch — no Hugging Face, no pre-built layers. Every component is hand-coded following the original *"Attention Is All You Need"* paper (Vaswani et al., 2017).

## ✨ Features

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

## 📁 Project Structure

```
Transformer-From-Scratch/
├── app.py                    ← Flask web interface
├── requirements.txt
├── README.md
├── data/
│   ├── tiny_shakespeare.txt  ← 1.1 MB Shakespeare corpus
│   └── wikitext2/
├── src/
│   ├── tokenizer.py          ← Phase 1: Word-level tokenizer
│   ├── dataset.py            ← Phase 1: PyTorch Dataset + DataLoaders
│   ├── positional_encoding.py← Phase 2: Sinusoidal + Learned PE
│   ├── attention.py          ← Phase 3: Scaled Dot-Product Attention
│   ├── multihead_attention.py← Phase 4: Multi-Head Attention
│   ├── feedforward.py        ← Phase 5: Feed-Forward Network
│   ├── encoder.py            ← Phase 6: Encoder Block + Stack
│   ├── decoder.py            ← Phase 7: Decoder Block + Stack
│   ├── transformer.py        ← Phase 8: Full Transformer + LM
│   ├── trainer.py            ← Phase 9: Training Pipeline
│   └── generate.py           ← Phase 10: Text Generation
├── checkpoints/              ← Saved model weights & curves
├── static/                   ← Web UI assets (CSS, JS)
└── templates/                ← Web UI HTML
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Individual Modules

```bash
cd Transformer-From-Scratch

python src/positional_encoding.py    # Phase 2
python src/attention.py              # Phase 3
python src/multihead_attention.py    # Phase 4
python src/feedforward.py            # Phase 5
python src/encoder.py                # Phase 6
python src/decoder.py                # Phase 7
python src/transformer.py            # Phase 8
python src/dataset.py                # Dataset
python src/generate.py               # Generation
```

### 3. Train the Model

**Quick smoke test (2 epochs):**
```bash
python src/trainer.py
```

**Full training (from Python):**
```python
from src.trainer import train

model, tokenizer, history = train(
    epochs=10,
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=4,
    seq_len=64,
    batch_size=64,
)
```

### 4. Generate Text

```python
from src.generate import generate_text

text = generate_text(
    model, tokenizer,
    prompt="To be or not",
    max_tokens=100,
    method="top_k",   # "greedy", "top_k", or "temperature"
    top_k=10,
    temperature=0.8,
)
print(text)
```

### 5. Launch Web Interface

```bash
python app.py
```

Open **http://localhost:5000** in your browser. The web UI lets you:
- 🏋️ Train the model with custom hyperparameters
- ✍️ Generate Shakespeare-style text with different strategies
- 📊 View model architecture and parameters

## 🏗️ Architecture Overview

```
Input Tokens
     ↓
Embedding × √d_model
     ↓
+ Positional Encoding (sinusoidal)
     ↓
┌─────────────────────────────┐
│   Encoder Layer × N         │
│   ┌───────────────────────┐ │
│   │ Multi-Head Attention  │ │
│   │ Add & Layer Norm      │ │
│   │ Feed-Forward Network  │ │
│   │ Add & Layer Norm      │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│   Decoder Layer × N         │
│   ┌───────────────────────┐ │
│   │ Masked Self-Attention │ │
│   │ Add & Layer Norm      │ │
│   │ Cross-Attention       │ │
│   │ Add & Layer Norm      │ │
│   │ Feed-Forward Network  │ │
│   │ Add & Layer Norm      │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
     ↓
Linear → Softmax → Output Probabilities
```

## 📊 Expected Results

| Dataset | Expected Perplexity |
|---------|-------------------|
| Tiny Shakespeare | 3–8 |
| WikiText-2 | 40–100 |

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch 2.0+** — model, training, tensors
- **NumPy** — numerical utilities
- **Matplotlib** — training curves
- **Flask** — web interface

## 📖 References

- Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## 📝 License

This project is for educational purposes.
