"""
Word-level Tokenizer for the Tiny Shakespeare Dataset
======================================================

Directory layout expected
─────────────────────────
Transformer-From-Scratch/
├── data/
│   └── tiny_shakespeare.txt   <- corpus lives here
├── src/
│   └── tokenizer.py           <- THIS file lives here
└── checkpoints/               <- saved vocab JSON goes here
"""

import os
import re
import json
import torch
from typing import List, Optional
from collections import Counter


# ── Path helpers ──────────────────────────────────────────────────────────────

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

DATA_PATH      = os.path.join(_PROJECT_DIR, "data", "tiny_shakespeare.txt")
CHECKPOINT_DIR = os.path.join(_PROJECT_DIR, "checkpoints")
TOKENIZER_SAVE = os.path.join(CHECKPOINT_DIR, "word_tokenizer.json")


class WordTokenizer:
    DEFAULT_SPECIAL = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    def __init__(self, text=None, file_path=None, min_freq=1, special_tokens=None):
        if file_path is not None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset not found at: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        if text is None:
            raise ValueError("Provide either text or file_path.")

        self.text           = text
        self.min_freq       = min_freq
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL

        words       = self._split(text)
        freq        = Counter(words)
        vocab_words = sorted(w for w, c in freq.items() if c >= min_freq)
        all_tokens  = self.special_tokens + vocab_words

        self.stoi = {w: i for i, w in enumerate(all_tokens)}
        self.itos = {i: w for w, i in self.stoi.items()}

        self.vocab_size = len(self.stoi)
        self.n_special  = len(self.special_tokens)

        self.pad_id = self.stoi.get("<PAD>")
        self.unk_id = self.stoi.get("<UNK>")
        self.sos_id = self.stoi.get("<SOS>")
        self.eos_id = self.stoi.get("<EOS>")

    @staticmethod
    def _split(text):
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text, add_sos=False, add_eos=False):
        words = self._split(text)
        ids   = [self.stoi.get(w, self.unk_id) for w in words]
        if add_sos and self.sos_id is not None:
            ids = [self.sos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=False):
        words = [self.itos.get(i, "<UNK>") for i in ids]
        if skip_special:
            words = [w for w in words if w not in self.special_tokens]
        result = ""
        for i, w in enumerate(words):
            if i == 0:
                result += w
            elif re.match(r"[^\w\s]", w):
                result += w
            else:
                result += " " + w
        return result

    def encode_to_tensor(self, text, add_sos=False, add_eos=False, dtype=torch.long):
        return torch.tensor(self.encode(text, add_sos=add_sos, add_eos=add_eos), dtype=dtype)

    def decode_from_tensor(self, t, skip_special=False):
        return self.decode(t.tolist(), skip_special=skip_special)

    def pad_sequence(self, sequences, max_len=None):
        max_len = max_len or max(len(s) for s in sequences)
        padded  = [s + [self.pad_id] * (max_len - len(s)) for s in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def get_vocab(self):
        return dict(self.stoi)

    def token_to_id(self, token):
        return self.stoi.get(token, self.unk_id)

    def id_to_token(self, idx):
        return self.itos.get(idx, "<UNK>")

    def encode_corpus(self):
        return self.encode(self.text)

    def encode_corpus_tensor(self):
        return self.encode_to_tensor(self.text)

    def get_freq_table(self):
        return Counter(self.encode(self.text))

    def save(self, path=TOKENIZER_SAVE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"special_tokens": self.special_tokens, "min_freq": self.min_freq, "stoi": self.stoi}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved  -> {path}")

    @classmethod
    def load(cls, path=TOKENIZER_SAVE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved tokenizer at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj                = cls.__new__(cls)
        obj.special_tokens = data["special_tokens"]
        obj.min_freq       = data["min_freq"]
        obj.stoi           = data["stoi"]
        obj.itos           = {int(i): w for w, i in obj.stoi.items()}
        obj.vocab_size     = len(obj.stoi)
        obj.n_special      = len(obj.special_tokens)
        obj.text           = ""
        obj.pad_id = obj.stoi.get("<PAD>")
        obj.unk_id = obj.stoi.get("<UNK>")
        obj.sos_id = obj.stoi.get("<SOS>")
        obj.eos_id = obj.stoi.get("<EOS>")
        print(f"[Tokenizer] Loaded <- {path}")
        return obj

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (f"WordTokenizer(vocab_size={self.vocab_size}, "
                f"min_freq={self.min_freq}, "
                f"special_tokens={self.special_tokens})")
