"""Local embedding backends.

* ``sentence-transformers`` (default): a real local model, downloaded once from Hugging Face.
* ``hashing``: dependency-free feature hashing. Weak semantically, but deterministic and offline;
  used by the tests and as an explicit opt-in (``HR_EMBEDDER=hashing``) when no model is available.

There is deliberately no silent fallback: mixing vector spaces or quietly degrading retrieval
quality is worse than a clear error.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#]*")


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [_stem(t) for t in _TOKEN.findall(text.lower())]
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers`, "
                "or set HR_EMBEDDER=hashing for a lightweight offline fallback."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(kind: str | None = None) -> Embedder:
    kind = (kind or os.getenv("HR_EMBEDDER", "sentence-transformers")).lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(os.getenv("HR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    raise ValueError(f"Unknown HR_EMBEDDER {kind!r} (use 'sentence-transformers' or 'hashing')")
