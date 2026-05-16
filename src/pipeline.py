"""
pipeline.py — Orchestration class that wires ingestion, embedding, storage,
and both retrieval strategies into a single cohesive interface.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from src.embeddings import (
    MockVertexAITextEmbeddingModel,
    embed_texts,
    get_embedding_model,
)
from src.query_expansion import MockVertexAIGenerativeModel, get_generative_model
from src.retrieval import RetrievalComparison, StrategyA, StrategyB, compare_strategies
from src.storage import Document, SearchResult, create_vector_store


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Manages:
    - Ingestion of raw text documents
    - Embedding generation (via mocked Vertex AI TextEmbeddingModel)
    - Storage in a local FAISS / NumPy vector store
    - Retrieval via Strategy A (raw) and Strategy B (query-expanded)
    """

    def __init__(
        self,
        embedding_model: Optional[MockVertexAITextEmbeddingModel] = None,
        generative_model: Optional[MockVertexAIGenerativeModel] = None,
    ):
        self._embedding_model = embedding_model or get_embedding_model()
        self._generative_model = generative_model or get_generative_model()

        dimension = self._embedding_model.dimension
        self._vector_store = create_vector_store(dimension)

        self._strategy_a = StrategyA(self._embedding_model, self._vector_store)
        self._strategy_b = StrategyB(
            self._embedding_model, self._generative_model, self._vector_store
        )

        self._ingested = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, raw_documents: List[Dict]) -> None:
        """
        Ingest a list of document dicts with keys: id, title, text.

        Generates embeddings in a single batch and adds them to the store.

        Parameters
        ----------
        raw_documents : list of dicts — each must have 'id', 'title', 'text'
        """
        documents = [
            Document(id=d["id"], title=d["title"], text=d["text"])
            for d in raw_documents
        ]
        texts = [d.text for d in documents]

        # Re-fit offline engine on the actual corpus for better vocabulary
        self._embedding_model.refit_if_offline(texts)

        # Recreate store with (possibly updated) dimension
        dimension = self._embedding_model.dimension
        self._vector_store = create_vector_store(dimension)
        self._strategy_a = StrategyA(self._embedding_model, self._vector_store)
        self._strategy_b = StrategyB(
            self._embedding_model, self._generative_model, self._vector_store
        )

        print(f"[Pipeline] Embedding {len(texts)} documents (dim={dimension}) …")
        embeddings = embed_texts(self._embedding_model, texts)

        self._vector_store.add(documents, embeddings)
        self._ingested = True
        print(f"[Pipeline] Ingestion complete. Store size: {len(self._vector_store)} vectors.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_strategy_a(self, query: str, k: int = 3) -> List[SearchResult]:
        """Raw vector search."""
        self._assert_ingested()
        return self._strategy_a.retrieve(query, k=k)

    def query_strategy_b(self, query: str, k: int = 3) -> tuple[str, List[SearchResult]]:
        """Query-expanded search. Returns (expanded_query, results)."""
        self._assert_ingested()
        return self._strategy_b.retrieve(query, k=k)

    def compare(self, query: str, k: int = 3) -> RetrievalComparison:
        """Run both strategies and return a side-by-side comparison."""
        self._assert_ingested()
        return compare_strategies(self._strategy_a, self._strategy_b, query, k=k)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def benchmark(
        self, queries: List[str], k: int = 3
    ) -> List[Dict]:
        """
        Run both strategies on every query and return a structured report.

        Returns
        -------
        List of dicts suitable for JSON serialisation or table rendering.
        """
        self._assert_ingested()
        report = []
        for query in queries:
            comparison = self.compare(query, k=k)
            entry = {
                "original_query": comparison.original_query,
                "expanded_query": comparison.expanded_query,
                "strategy_a": [
                    {
                        "rank": r.rank,
                        "doc_id": r.document.id,
                        "title": r.document.title,
                        "score": round(r.score, 4),
                        "snippet": textwrap.shorten(r.document.text, width=120),
                    }
                    for r in comparison.strategy_a_results
                ],
                "strategy_b": [
                    {
                        "rank": r.rank,
                        "doc_id": r.document.id,
                        "title": r.document.title,
                        "score": round(r.score, 4),
                        "snippet": textwrap.shorten(r.document.text, width=120),
                    }
                    for r in comparison.strategy_b_results
                ],
                "overlap": _compute_overlap(
                    comparison.strategy_a_results, comparison.strategy_b_results
                ),
            }
            report.append(entry)
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_ingested(self) -> None:
        if not self._ingested:
            raise RuntimeError(
                "No documents ingested. Call pipeline.ingest(documents) first."
            )

    @property
    def store_size(self) -> int:
        return len(self._vector_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_overlap(
    results_a: List[SearchResult], results_b: List[SearchResult]
) -> Dict:
    """Compute overlap statistics between two result sets."""
    ids_a = {r.document.id for r in results_a}
    ids_b = {r.document.id for r in results_b}
    common = ids_a & ids_b
    return {
        "shared_docs": sorted(common),
        "overlap_count": len(common),
        "jaccard": round(len(common) / len(ids_a | ids_b), 4) if (ids_a | ids_b) else 0.0,
    }


def print_report(report: List[Dict]) -> None:
    """Pretty-print a benchmark report to stdout."""
    sep = "=" * 80
    for i, entry in enumerate(report, start=1):
        print(f"\n{sep}")
        print(f"QUERY {i}: {entry['original_query']}")
        print(f"EXPANDED: {entry['expanded_query'][:100]}…")
        print(sep)

        print("\n  ── Strategy A: Raw Vector Search ──")
        for r in entry["strategy_a"]:
            print(f"  [{r['rank']}] {r['title']} (score={r['score']})")
            print(f"      {r['snippet']}")

        print("\n  ── Strategy B: AI-Enhanced Retrieval ──")
        for r in entry["strategy_b"]:
            print(f"  [{r['rank']}] {r['title']} (score={r['score']})")
            print(f"      {r['snippet']}")

        ov = entry["overlap"]
        print(
            f"\n  Overlap: {ov['overlap_count']}/3 docs shared "
            f"(Jaccard={ov['jaccard']}) — shared: {ov['shared_docs']}"
        )
    print(f"\n{sep}\n")