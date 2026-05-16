"""
tests/test_storage.py — Tests for VectorStore implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.storage import (
    Document,
    FAISSVectorStore,
    NumpyVectorStore,
    SearchResult,
    create_vector_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for fast tests


def make_docs(n: int) -> list[Document]:
    return [Document(id=f"doc_{i}", title=f"Title {i}", text=f"Content {i}") for i in range(n)]


def make_random_vecs(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # L2-normalise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


# ---------------------------------------------------------------------------
# NumpyVectorStore
# ---------------------------------------------------------------------------

class TestNumpyVectorStore:
    def _store(self, n=5):
        store = NumpyVectorStore(dimension=DIM)
        docs = make_docs(n)
        vecs = make_random_vecs(n)
        store.add(docs, vecs)
        return store, docs, vecs

    def test_add_increments_length(self):
        store = NumpyVectorStore(dimension=DIM)
        assert len(store) == 0
        docs = make_docs(3)
        vecs = make_random_vecs(3)
        store.add(docs, vecs)
        assert len(store) == 3

    def test_search_returns_k_results(self):
        store, docs, vecs = self._store(n=5)
        query = make_random_vecs(1)[0]
        results = store.search(query, k=3)
        assert len(results) == 3

    def test_search_results_are_sorted_descending(self):
        store, _, _ = self._store(n=5)
        query = make_random_vecs(1)[0]
        results = store.search(query, k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_nearest_doc_is_query_itself(self):
        """If we embed doc 0 and use it as the query, rank 1 should be doc 0."""
        store, docs, vecs = self._store(n=5)
        query = vecs[0]
        results = store.search(query, k=1)
        assert results[0].document.id == docs[0].id

    def test_ranks_are_sequential(self):
        store, _, _ = self._store(n=5)
        query = make_random_vecs(1)[0]
        results = store.search(query, k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_empty_store_returns_empty_list(self):
        store = NumpyVectorStore(dimension=DIM)
        query = make_random_vecs(1)[0]
        assert store.search(query, k=3) == []

    def test_mismatch_raises(self):
        store = NumpyVectorStore(dimension=DIM)
        with pytest.raises(AssertionError):
            store.add(make_docs(3), make_random_vecs(2))  # 3 docs, 2 vecs

    def test_search_result_type(self):
        store, _, _ = self._store(n=3)
        results = store.search(make_random_vecs(1)[0], k=1)
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].document, Document)


# ---------------------------------------------------------------------------
# FAISSVectorStore (if available)
# ---------------------------------------------------------------------------

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False


@pytest.mark.skipif(not _FAISS, reason="faiss not installed")
class TestFAISSVectorStore:
    def _store(self, n=5):
        store = FAISSVectorStore(dimension=DIM)
        docs = make_docs(n)
        vecs = make_random_vecs(n)
        store.add(docs, vecs)
        return store, docs, vecs

    def test_add_increments_length(self):
        store = FAISSVectorStore(dimension=DIM)
        store.add(make_docs(3), make_random_vecs(3))
        assert len(store) == 3

    def test_search_returns_k_results(self):
        store, _, _ = self._store(5)
        results = store.search(make_random_vecs(1)[0], k=3)
        assert len(results) == 3

    def test_nearest_is_self(self):
        store, docs, vecs = self._store(5)
        results = store.search(vecs[2], k=1)
        assert results[0].document.id == docs[2].id

    def test_sorted_descending(self):
        store, _, _ = self._store(5)
        results = store.search(make_random_vecs(1)[0], k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# create_vector_store factory
# ---------------------------------------------------------------------------

class TestCreateVectorStore:
    def test_returns_a_store_with_correct_dimension(self):
        store = create_vector_store(DIM)
        # should not raise
        store.add(make_docs(2), make_random_vecs(2))
        assert len(store) == 2