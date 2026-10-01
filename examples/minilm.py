from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256
LAYERS = 6
HEADS = 12
HIDDEN = 384


class MiniLM:
    def __init__(self, weights, tokenizer: Tokenizer):
        self.weights = weights
        self.tokenizer = tokenizer

    def encode(self, texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False):
        rows = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            rows.append(self._encode_batch(batch, normalize_embeddings))
            if show_progress_bar:
                done = min(start + batch_size, total)
                print(f"  encoded {done}/{total}", flush=True)
        if not rows:
            return np.zeros((0, HIDDEN), dtype=np.float32)
        return np.concatenate(rows, axis=0)

    def _encode_batch(self, texts, normalize: bool):
        encoded = self.tokenizer.encode_batch([text or "" for text in texts])
        token_ids = torch.tensor([item.ids for item in encoded], dtype=torch.long)
        mask = torch.tensor([item.attention_mask for item in encoded], dtype=torch.long)
        hidden = _bert(self.weights, token_ids, mask)
        pooled = _mean_pool(hidden, mask)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.numpy().astype(np.float32)


def from_cache(model_id: str = MODEL_ID) -> MiniLM:
    folder = _snapshot(model_id)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0, pad_token="[PAD]")
    weights = {}
    with safe_open(str(folder / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            weights[key] = handle.get_tensor(key).float()
    return MiniLM(weights, tokenizer)


def _snapshot(model_id: str) -> Path:
    folder_name = "models--" + model_id.replace("/", "--")
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    ref = hub / folder_name / "refs" / "main"
    if not ref.exists():
        raise SystemExit(
            f"{model_id} is not in the Hugging Face cache at {hub}. "
            "On Bouchet, run this after the model has been downloaded into HF_HOME."
        )
    folder = hub / folder_name / "snapshots" / ref.read_text(encoding="utf-8").strip()
    if not (folder / "model.safetensors").exists():
        raise SystemExit(f"Missing weights in {folder}.")
    return folder


def _bert(weights, token_ids, mask):
    batch, length = token_ids.shape
    positions = torch.arange(length).unsqueeze(0).expand(batch, length)
    types = torch.zeros_like(token_ids)
    hidden = (
        weights["embeddings.word_embeddings.weight"][token_ids]
        + weights["embeddings.position_embeddings.weight"][positions]
        + weights["embeddings.token_type_embeddings.weight"][types]
    )
    hidden = _layer_norm(hidden, weights, "embeddings.LayerNorm")
    for layer in range(LAYERS):
        hidden = _layer(hidden, mask, weights, layer)
    return hidden


def _layer(hidden, mask, weights, layer: int):
    prefix = f"encoder.layer.{layer}."
    heads = HEADS
    head_dim = HIDDEN // heads
    batch, length, _ = hidden.shape
    query = _linear(hidden, weights, prefix + "attention.self.query")
    key = _linear(hidden, weights, prefix + "attention.self.key")
    value = _linear(hidden, weights, prefix + "attention.self.value")

    def split(tensor):
        return tensor.view(batch, length, heads, head_dim).transpose(1, 2)

    query, key, value = split(query), split(key), split(value)
    scores = (query @ key.transpose(-1, -2)) / (head_dim ** 0.5)
    scores = scores.masked_fill(mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
    context = torch.softmax(scores, dim=-1) @ value
    context = context.transpose(1, 2).contiguous().view(batch, length, HIDDEN)
    attended = _linear(context, weights, prefix + "attention.output.dense")
    hidden = _layer_norm(hidden + attended, weights, prefix + "attention.output.LayerNorm")
    intermediate = F.gelu(_linear(hidden, weights, prefix + "intermediate.dense"))
    projected = _linear(intermediate, weights, prefix + "output.dense")
    return _layer_norm(hidden + projected, weights, prefix + "output.LayerNorm")


def _linear(hidden, weights, name: str):
    return hidden @ weights[name + ".weight"].T + weights[name + ".bias"]


def _layer_norm(hidden, weights, name: str):
    return F.layer_norm(hidden, (HIDDEN,), weights[name + ".weight"], weights[name + ".bias"], 1e-12)


def _mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).float()
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return summed / counts
