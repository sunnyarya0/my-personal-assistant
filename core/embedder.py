"""
core/embedder.py
================
Local text embedding pipeline using sentence-transformers.

Runs 100% locally — no API key or network calls required.
Default model: all-MiniLM-L6-v2  (22MB, 384-dim, fast and accurate)

Usage:
    from core.embedder import get_embedder

    embedder = get_embedder()
    vec = embedder.embed("My cat is fluffy")       # → list[float] len=384
    vecs = embedder.embed_batch(["text1", "text2"]) # → list[list[float]]
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default model — tiny (22 MB), fast, good quality for semantic search
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """
    Wraps sentence-transformers to produce fixed-size embedding vectors
    from text strings. The model is lazy-loaded on first use.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load(self):
        """Load the sentence-transformers model (first call only)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                logger.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded (dim=%d)", self.dimension)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (depends on model)."""
        model = self._load()
        return model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a float vector.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        model = self._load()
        vector = model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in one batched call (more efficient).

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors (same order as input).
        """
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Cosine similarity score in range [-1, 1] (higher = more similar).
        """
        import math
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


@lru_cache(maxsize=1)
def get_embedder(model_name: str = DEFAULT_MODEL) -> LocalEmbedder:
    """
    Return the singleton LocalEmbedder instance (cached after first call).

    Args:
        model_name: sentence-transformers model identifier.

    Returns:
        Configured LocalEmbedder ready to use.
    """
    return LocalEmbedder(model_name)


__all__ = ["LocalEmbedder", "get_embedder", "DEFAULT_MODEL"]
