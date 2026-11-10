"""
tests/test_embedder.py
======================
Unit tests for core/embedder.py.
All tests use a mocked sentence-transformers to avoid downloading model files during unit tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from core.embedder import DEFAULT_MODEL, LocalEmbedder, get_embedder


class TestLocalEmbedder:
    @pytest.fixture(autouse=True)
    def mock_sentence_transformers(self) -> Iterator[None]:
        """Mock the sentence_transformers library to avoid network/local downloads."""
        self.mock_model_class = MagicMock()
        self.mock_model_instance = MagicMock()

        # Mock the embedding dimension
        self.mock_model_instance.get_sentence_embedding_dimension.return_value = 384

        # Mock encoding a single text
        self.mock_vector = MagicMock()
        self.mock_vector.tolist.return_value = [0.1] * 384

        # Mock encoding batch texts
        self.mock_vectors = [self.mock_vector, self.mock_vector]

        def encode_mock(
            text_or_texts: str | list[str], **kwargs: object
        ) -> MagicMock | list[MagicMock]:
            if isinstance(text_or_texts, str):
                return self.mock_vector
            else:
                return self.mock_vectors

        self.mock_model_instance.encode.side_effect = encode_mock
        self.mock_model_class.return_value = self.mock_model_instance

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = self.mock_model_class

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            yield

    def test_init_and_lazy_load(self) -> None:
        embedder = LocalEmbedder(model_name="test-model")
        assert embedder.model_name == "test-model"
        assert embedder._model is None

        # Verify lazy-loading on dimension check
        dim = embedder.dimension
        assert dim == 384
        assert embedder._model is self.mock_model_instance
        self.mock_model_class.assert_called_once_with("test-model")

    def test_embed_success(self) -> None:
        embedder = LocalEmbedder()
        vec = embedder.embed("test sentence")

        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)
        self.mock_model_instance.encode.assert_called_once_with(
            "test sentence", convert_to_numpy=True
        )

    def test_embed_empty_or_whitespace_raises(self) -> None:
        embedder = LocalEmbedder()
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed("")

        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedder.embed("    ")

    def test_embed_batch_success(self) -> None:
        embedder = LocalEmbedder()
        texts = ["sentence one", "sentence two"]
        vecs = embedder.embed_batch(texts)

        assert isinstance(vecs, list)
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)
        self.mock_model_instance.encode.assert_called_once_with(
            texts, convert_to_numpy=True, batch_size=32
        )

    def test_embed_batch_empty(self) -> None:
        embedder = LocalEmbedder()
        assert embedder.embed_batch([]) == []

    def test_similarity_calculation(self) -> None:
        embedder = LocalEmbedder()
        # Identical vectors (perfect cosine similarity = 1.0)
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert pytest.approx(embedder.similarity(v1, v2)) == 1.0

        # Orthogonal vectors (similarity = 0.0)
        v3 = [0.0, 1.0, 0.0]
        assert pytest.approx(embedder.similarity(v1, v3)) == 0.0

        # Opposite vectors (similarity = -1.0)
        v4 = [-1.0, 0.0, 0.0]
        assert pytest.approx(embedder.similarity(v1, v4)) == -1.0

        # Zero vector handling (should return 0.0 to avoid ZeroDivisionError)
        v_zero = [0.0, 0.0, 0.0]
        assert embedder.similarity(v1, v_zero) == 0.0

    def test_singleton_get_embedder(self) -> None:
        # Clear lru_cache for a clean test
        get_embedder.cache_clear()

        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2
        assert embedder1.model_name == DEFAULT_MODEL

    def test_import_error_raises_friendly_message(self) -> None:
        embedder = LocalEmbedder()
        # Simulate import error
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers is not installed"):
                embedder._load()
