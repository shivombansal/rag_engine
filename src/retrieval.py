"""
retrieval.py — Two retrieval strategies.

Strategy A: Raw Vector Search
    Embed the user query directly and search the vector store.

Strategy B: AI-Enhanced Retrieval
    Expand the query via MockVertexAIGenerativeModel, then embed and search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.embeddings import MockVertexAITextEmbeddingModel, embed_query
from src.query_expansion import MockVertexAIGenerativeModel, expand_query
from src.storage import SearchResult, create_vector_store


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalComparison:
    original_query: str
    expanded_query: str
    strategy_a_results: List[SearchResult]
    strategy_b_results: List[SearchResult]


# ---------------------------------------------------------------------------
# Strategy A — Raw vector search
# ---------------------------------------------------------------------------

class StrategyA:
    """
    Direct embedding similarity search.

    Query → embed → FAISS/NumPy inner-product search → top-k results.
    """

    name = "Strategy A: Raw Vector Search"

    def __init__(
        self,
        embedding_model: MockVertexAITextEmbeddingModel,
        vector_store,
    ):
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> List[SearchResult]:
        """Embed query and return top-k most similar documents."""
        query_vec = embed_query(self._embedding_model, query)
        return self._vector_store.search(query_vec, k=k)


# ---------------------------------------------------------------------------
# Strategy B — AI-Enhanced Retrieval with query expansion
# ---------------------------------------------------------------------------

class StrategyB:
    """
    Query expansion + embedding similarity search.

    Query → GenerativeModel rewrite → embed expanded text → search → top-k.
    """

    name = "Strategy B: AI-Enhanced Retrieval (Query Expansion)"

    def __init__(
        self,
        embedding_model: MockVertexAITextEmbeddingModel,
        generative_model: MockVertexAIGenerativeModel,
        vector_store,
    ):
        self._embedding_model = embedding_model
        self._generative_model = generative_model
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> tuple[str, List[SearchResult]]:
        """
        Expand the query, embed the expansion, and return top-k results.

        Returns
        -------
        (expanded_query, results)
        """
        expanded = expand_query(self._generative_model, query)
        query_vec = embed_query(self._embedding_model, expanded)
        results = self._vector_store.search(query_vec, k=k)
        return expanded, results


# ---------------------------------------------------------------------------
# Head-to-head comparison helper
# ---------------------------------------------------------------------------

def compare_strategies(
    strategy_a: StrategyA,
    strategy_b: StrategyB,
    query: str,
    k: int = 3,
) -> RetrievalComparison:
    """
    Run both strategies on the same query and return a comparison object.
    """
    a_results = strategy_a.retrieve(query, k=k)
    expanded_query, b_results = strategy_b.retrieve(query, k=k)
    return RetrievalComparison(
        original_query=query,
        expanded_query=expanded_query,
        strategy_a_results=a_results,
        strategy_b_results=b_results,
    )