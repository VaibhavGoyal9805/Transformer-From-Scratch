"""
Transformer From Scratch — Web Interface
=========================================

A Flask web app that provides a browser UI for:
  • Training the TransformerLM on Tiny Shakespeare
  • Generating text with greedy / top-k / temperature sampling
  • Viewing model architecture and training metrics
"""

import os
import sys
import json
import math
import time
import threading

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_APP_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from flask import Flask, render_template, request, jsonify

import torch
torch.set_num_threads(1)
from transformer import TransformerLM
from tokenizer import WordTokenizer, DATA_PATH
from generate import generate_text
from trainer import train as train_model, plot_training_curves, CHECKPOINT_DIR

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "training": False,
    "history": None,
    "config": {},
}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_or_build_model(d_model=256, n_heads=8, d_ff=1024, n_layers=6, dropout=0.1, max_len=228):
    """Build tokenizer and model. Load checkpoint if available."""
    device = _get_device()
    _state["device"] = device

    tokenizer = WordTokenizer(file_path=DATA_PATH, min_freq=1)
    _state["tokenizer"] = tokenizer

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout,
        max_len=max_len,
        pad_id=tokenizer.pad_id,
    ).to(device)

    # Try loading latest checkpoint
    ckpt_files = sorted([
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("transformer_lm_epoch") and f.endswith(".pt")
    ]) if os.path.exists(CHECKPOINT_DIR) else []

    if ckpt_files:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_files[-1])
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Web] Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"[Web] Could not load checkpoint ({e}), using fresh model.")

    model.eval()
    _state["model"] = model
    _state["config"] = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "dropout": dropout,
        "total_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    return model, tokenizer


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info")
def model_info():
    if _state["model"] is None:
        return jsonify({"loaded": False})
    return jsonify({"loaded": True, **_state["config"]})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if _state["model"] is None:
        return jsonify({"error": "Model not loaded. Train or load a model first."}), 400

    data = request.json or {}
    prompt      = data.get("prompt", "To be")
    max_tokens  = min(int(data.get("max_tokens", 100)), 500)
    method      = data.get("method", "top_k")
    temperature = float(data.get("temperature", 0.8))
    top_k       = int(data.get("top_k", 10))

    try:
        t0 = time.time()
        text = generate_text(
            _state["model"],
            _state["tokenizer"],
            prompt=prompt,
            max_tokens=max_tokens,
            method=method,
            temperature=temperature,
            top_k=top_k,
            device=_state["device"],
        )
        elapsed = time.time() - t0
        return jsonify({
            "text": text,
            "time": round(elapsed, 2),
            "method": method,
            "tokens_generated": max_tokens,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    if _state["training"]:
        return jsonify({"error": "Training already in progress."}), 409

    data = request.json or {}
    epochs  = min(int(data.get("epochs", 3)), 20)
    d_model = int(data.get("d_model", 256))
    n_heads = int(data.get("n_heads", 8))
    d_ff    = int(data.get("d_ff", 1024))
    n_layers = int(data.get("n_layers", 6))

    def _train_thread():
        _state["training"] = True
        try:
            model, tokenizer, history = train_model(
                epochs=epochs,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_layers=n_layers,
                dropout=0.1,
                seq_len=128,
                batch_size=64,
                warmup_steps=500,
            )
            model.eval()
            _state["model"] = model
            _state["tokenizer"] = tokenizer
            _state["history"] = history
            _state["config"] = {
                "vocab_size": tokenizer.vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "n_layers": n_layers,
                "dropout": 0.1,
                "total_params": sum(p.numel() for p in model.parameters()),
                "device": str(_state["device"]),
            }
            plot_training_curves(history)
        except Exception as e:
            print(f"[Web] Training error: {e}")
        finally:
            _state["training"] = False

    thread = threading.Thread(target=_train_thread, daemon=True)
    thread.start()

    return jsonify({"status": "Training started", "epochs": epochs})


@app.route("/api/training-status")
def training_status():
    result = {"training": _state["training"]}
    if _state["history"]:
        result["history"] = _state["history"]
    return jsonify(result)


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Transformer From Scratch — Web Interface")
print("=" * 60)

# ── Reassemble Model Checkpoint ────────────────────────────────────────────────
ckpt_path = os.path.join(_APP_DIR, "checkpoints", "transformer_lm_epoch15.pt")
if not os.path.exists(ckpt_path):
    ckpt_dir = os.path.join(_APP_DIR, "checkpoints")
    chunks = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("chunk_transformer_lm_epoch15.pt_")])
    if chunks:
        print(f"[Web] Reassembling model from {len(chunks)} chunks...")
        with open(ckpt_path, 'wb') as outfile:
            for chunk in chunks:
                with open(os.path.join(ckpt_dir, chunk), 'rb') as infile:
                    outfile.write(infile.read())
        print("[Web] Reassembly complete.")

_load_or_build_model(d_model=256, n_heads=8, d_ff=1024, n_layers=6)
print(f"[Web] Model loaded: {_state['config']['total_params']:,} parameters")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[Web] Starting server at http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(debug=False, port=port, host="0.0.0.0")
