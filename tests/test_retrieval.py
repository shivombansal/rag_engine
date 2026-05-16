"""
tests/test_retrieval.py — Tests for retrieval strategies and query expansion mock.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.embeddings import get_embedding_model, embed_texts
from src.query_expansion import (
    MockVertexAIGenerativeModel,
    expand_query,
    get_generative_model,
)
from src.retrieval import RetrievalComparison, StrategyA, StrategyB, compare_strategies
from src.storage import Document, NumpyVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    {"id": "d1", "title": "Load Balancing", "text": "Load balancing distributes traffic across servers during peak load."},
    {"id": "d2", "title": "Caching", "text": "Redis caches frequently accessed data to reduce latency."},
    {"id": "d3", "title": "Rate Limiting", "text": "Token-bucket rate limiting prevents backend overload from API clients."},
    {"id": "d4", "title": "Sharding", "text": "Database sharding partitions data for horizontal scalability."},
    {"id": "d5", "title": "Replication", "text": "Synchronous replication ensures no data loss during failover."},
]


@pytest.fixture(scope="module")
def populated_store():
    model = get_embedding_model()
    texts = [d["text"] for d in CORPUS]
    docs = [Document(id=d["id"], title=d["title"], text=d["text"]) for d in CORPUS]
    embeddings = embed_texts(model, texts)
    store = NumpyVectorStore(dimension=model.dimension)
    store.add(docs, embeddings)
    return store, model


@pytest.fixture(scope="module")
def strategy_a(populated_store):
    store, model = populated_store
    return StrategyA(embedding_model=model, vector_store=store)


@pytest.fixture(scope="module")
def strategy_b(populated_store):
    store, model = populated_store
    gen_model = get_generative_model()
    return StrategyB(embedding_model=model, generative_model=gen_model, vector_store=store)


# ---------------------------------------------------------------------------
# Query expansion mock
# ---------------------------------------------------------------------------

class TestQueryExpansionMock:
    def test_returns_string(self):
        model = MockVertexAIGenerativeModel()
        result = expand_query(model, "How does peak load handling work?")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_peak_load_expansion_contains_relevant_terms(self):
        model = MockVertexAIGenerativeModel()
        expanded = expand_query(model, "How does the system handle peak load?")
        lower = expanded.lower()
        relevant = {"load balancing", "auto-scaling", "circuit breaker", "rate limiting", "backpressure"}
        matched = [term for term in relevant if term in lower]
        assert len(matched) >= 2, f"Expected ≥2 relevant terms, got: {matched}"

    def test_different_queries_produce_different_expansions(self):
        model = MockVertexAIGenerativeModel()
        e1 = expand_query(model, "How does caching work?")
        e2 = expand_query(model, "How is data replicated?")
        assert e1 != e2

    def test_mock_generative_model_generate_content(self):
        model = MockVertexAIGenerativeModel()
        response = model.generate_content('Query: "peak load handling"')
        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    def test_get_generative_model_returns_mock(self):
        model = get_generative_model()
        assert isinstance(model, MockVertexAIGenerativeModel)

    def test_patch_generative_model(self):
        """Verify patching the GCP GenerativeModel interface."""
        fake_model = MagicMock()
        fake_model.generate_content.return_value = MagicMock(
            text="expanded: auto-scaling load balancer circuit breaker"
        )
        result = expand_query(fake_model, "peak load")
        fake_model.generate_content.assert_called_once()
        assert "auto-scaling" in result


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------

class TestStrategyA:
    def test_returns_list(self, strategy_a):
        results = strategy_a.retrieve("peak load handling", k=3)
        assert isinstance(results, list)

    def test_returns_k_results(self, strategy_a):
        results = strategy_a.retrieve("caching strategy", k=2)
        assert len(results) == 2

    def test_load_query_returns_load_balancing_doc(self, strategy_a):
        results = strategy_a.retrieve("traffic spike load balancing", k=1)
        assert results[0].document.id == "d1"

    def test_scores_descending(self, strategy_a):
        results = strategy_a.retrieve("database partitioning", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_sequential(self, strategy_a):
        results = strategy_a.retrieve("test", k=3)
        assert [r.rank for r in results] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------

class TestStrategyB:
    def test_returns_expanded_query_and_results(self, strategy_b):
        expanded, results = strategy_b.retrieve("peak load", k=3)
        assert isinstance(expanded, str)
        assert len(expanded) > len("peak load")
        assert isinstance(results, list)
        assert len(results) == 3

    def test_expanded_query_differs_from_original(self, strategy_b):
        original = "rate limiting"
        expanded, _ = strategy_b.retrieve(original, k=3)
        assert expanded != original

    def test_expansion_influences_retrieval(self, strategy_a, strategy_b):
        """
        After expansion, the top result MAY differ from raw search.
        We just confirm strategy B runs end-to-end without error.
        """
        query = "prevent clients from overwhelming backend"
        a_results = strategy_a.retrieve(query, k=3)
        _, b_results = strategy_b.retrieve(query, k=3)
        # Both return 3 results
        assert len(a_results) == 3
        assert len(b_results) == 3


# ---------------------------------------------------------------------------
# compare_strategies
# ---------------------------------------------------------------------------

class TestCompareStrategies:
    def test_returns_comparison_object(self, strategy_a, strategy_b):
        comparison = compare_strategies(strategy_a, strategy_b, "peak load", k=3)
        assert isinstance(comparison, RetrievalComparison)

    def test_comparison_fields(self, strategy_a, strategy_b):
        comparison = compare_strategies(strategy_a, strategy_b, "test query", k=3)
        assert comparison.original_query == "test query"
        assert isinstance(comparison.expanded_query, str)
        assert len(comparison.strategy_a_results) == 3
        assert len(comparison.strategy_b_results) == 3