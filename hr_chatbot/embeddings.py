"""Embedding backends, chosen with ``HR_EMBEDDER``.

* ``azure``: Azure OpenAI embeddings (default ``text-embedding-3-large``). Selected automatically when
  ``AZURE_OPENAI_ENDPOINT`` is set, the same switch ``llm.py`` uses for the chat model.
* ``sentence-transformers``: a local model, downloaded once from Hugging Face (default when Azure is not configured).
* ``hashing``: dependency-free feature hashing. Weak semantically, but deterministic and offline;
  used by the tests and as an explicit opt-in (``HR_EMBEDDER=hashing``) when no model is available.

There is deliberately no silent fallback: mixing vector spaces or quietly degrading retrieval
quality is worse than a clear error. The index records which embedder built it and refuses to open with another.
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


class AzureOpenAIEmbedder:
    """Azure OpenAI embeddings. Configuration (environment variables):

    AZURE_OPENAI_ENDPOINT                 required, e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_API_KEY                  required
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT     deployment name (default: text-embedding-3-large)
    AZURE_OPENAI_EMBEDDING_DIMENSIONS     optional: shorten the vectors (text-embedding-3-* supports this)
    AZURE_OPENAI_EMBEDDING_API_VERSION    optional (falls back to OPENAI_API_VERSION, then 2024-10-21)

    The vector size is read from the first response when not configured, so the index always matches the model.
    """

    def __init__(
        self,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError(
                "Azure embeddings need AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY. "
                "Set them (see .env.example), or choose HR_EMBEDDER=sentence-transformers / hashing."
            )
        if dimensions is None and os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS"):
            dimensions = int(os.environ["AZURE_OPENAI_EMBEDDING_DIMENSIONS"])
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

        from langchain_openai import AzureOpenAIEmbeddings

        kwargs: dict = dict(
            azure_endpoint=endpoint,
            azure_deployment=self.deployment,
            api_key=api_key,
            api_version=api_version
            or os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
            or os.getenv("OPENAI_API_VERSION", "2024-10-21"),
            # Chunks are ~900 characters, far below the token limit. Leaving the length check on would make the
            # library download a tokenizer file at runtime, which fails on networks without general internet access.
            check_embedding_ctx_length=False,
            max_retries=3,
            timeout=60,
        )
        if dimensions:
            kwargs["dimensions"] = int(dimensions)
        self._client = AzureOpenAIEmbeddings(**kwargs)

        probe = self._embed_raw(["dimension probe"])
        self.dim = int(probe.shape[1])
        if dimensions and self.dim != int(dimensions):
            raise RuntimeError(f"Azure returned {self.dim}-dimensional vectors but {dimensions} were requested.")
        self.name = f"azure:{self.deployment}"

    def _embed_raw(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._client.embed_documents(list(texts)), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise RuntimeError(f"Unexpected embeddings response shape {vectors.shape} for {len(texts)} inputs.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed_raw(texts)


def default_embedder_kind() -> str:
    explicit = os.getenv("HR_EMBEDDER")
    if explicit:
        return explicit.lower()
    return "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "sentence-transformers"


def get_embedder(kind: str | None = None) -> Embedder:
    kind = (kind or default_embedder_kind()).lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("azure", "azure-openai"):
        return AzureOpenAIEmbedder()
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(os.getenv("HR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    raise ValueError(f"Unknown HR_EMBEDDER {kind!r} (use 'azure', 'sentence-transformers' or 'hashing')")
