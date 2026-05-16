"""
tests/test_embeddings.py — Tests for the embedding layer and Vertex AI mock.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.embeddings import (
    LocalEmbeddingEngine,
    MockVertexAITextEmbeddingModel,
    _MockTextEmbeddingResponse,
    embed_query,
    embed_texts,
    get_embedding_model,
)


# ---------------------------------------------------------------------------
# LocalEmbeddingEngine
# ---------------------------------------------------------------------------

class TestLocalEmbeddingEngine:
    def setup_method(self):
        self.engine = LocalEmbeddingEngine()

    def test_encode_returns_correct_shape(self):
        texts = ["Hello world", "Test sentence"]
        vecs = self.engine.encode(texts)
        assert vecs.shape == (2, self.engine.dimension)

    def test_encode_single_returns_1d(self):
        vec = self.engine.encode_single("Hello")
        assert vec.ndim == 1
        assert vec.shape[0] == self.engine.dimension

    def test_vectors_are_l2_normalised(self):
        """L2 norm of each embedding should be ~1.0 (for in-vocabulary words)."""
        vecs = self.engine.encode(["load balancing server scaling", "caching redis latency"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dtype_is_float32(self):
        vecs = self.engine.encode(["dtype check"])
        assert vecs.dtype == np.float32

    def test_similar_texts_have_high_cosine_similarity(self):
        engine = self.engine
        v1 = engine.encode_single("load balancing distributes traffic")
        v2 = engine.encode_single("traffic distribution via load balancer")
        v_unrelated = engine.encode_single("quantum chromodynamics")
        sim_related = float(v1 @ v2)
        sim_unrelated = float(v1 @ v_unrelated)
        assert sim_related > sim_unrelated, (
            f"Related pair should score higher than unrelated: {sim_related:.3f} vs {sim_unrelated:.3f}"
        )


# ---------------------------------------------------------------------------
# MockVertexAITextEmbeddingModel
# ---------------------------------------------------------------------------

class TestMockVertexAITextEmbeddingModel:
    def setup_method(self):
        self.model = MockVertexAITextEmbeddingModel.from_pretrained(
            "textembedding-gecko@003"
        )

    def test_from_pretrained_returns_instance(self):
        assert isinstance(self.model, MockVertexAITextEmbeddingModel)

    def test_get_embeddings_returns_list_of_responses(self):
        texts = ["first", "second", "third"]
        responses = self.model.get_embeddings(texts)
        assert len(responses) == 3
        for r in responses:
            assert isinstance(r, _MockTextEmbeddingResponse)
            assert len(r.values) == self.model.dimension

    def test_get_embeddings_values_normalised(self):
        responses = self.model.get_embeddings(["unit vector test"])
        vec = np.array(responses[0].values)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_dimension_matches_engine(self):
        assert self.model.dimension == self.model._engine.dimension

    def test_model_name_stored(self):
        assert self.model.model_name == "textembedding-gecko@003"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

class TestEmbedHelpers:
    def setup_method(self):
        self.model = get_embedding_model()

    def test_embed_texts_shape(self):
        texts = ["a", "b", "c"]
        vecs = embed_texts(self.model, texts)
        assert vecs.shape == (3, self.model.dimension)

    def test_embed_query_is_1d(self):
        vec = embed_query(self.model, "single query")
        assert vec.ndim == 1

    def test_get_embedding_model_returns_mock(self):
        model = get_embedding_model()
        assert isinstance(model, MockVertexAITextEmbeddingModel)

    def test_patch_vertex_ai_sdk(self):
        """
        Demonstrates how to patch the real Vertex AI SDK in unit tests.
        The mock is imported as if it were the real vertexai module.
        """
        fake_response = MagicMock()
        fake_response.values = [0.1] * 384

        with patch.object(
            MockVertexAITextEmbeddingModel,
            "get_embeddings",
            return_value=[fake_response],
        ) as mock_fn:
            model = get_embedding_model()
            result = embed_texts(model, ["patched input"])
            mock_fn.assert_called_once_with(["patched input"])
            assert result.shape[1] == 384