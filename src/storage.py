"""
storage.py — Lightweight vector store backed by FAISS (with a NumPy fallback).

Design choice: Cosine similarity (via inner product on L2-normalised vectors).

Why cosine over Euclidean?
- Embeddings from sentence-transformers and textembedding-gecko are
  unit-normalised by default, making dot-product == cosine similarity.
- Cosine is invariant to vector magnitude, so short and long documents
  are compared fairly.
- Euclidean distance on high-dimensional unit vectors is a monotone
  transformation of cosine distance (d_euc² = 2 - 2·cos), so rankings
  are identical — but cosine scores are more interpretable (range [-1, 1]).
- Vertex AI Matching Engine also defaults to DOT_PRODUCT_DISTANCE on
  normalised vectors, making the migration trivial.

Production migration to Vertex AI Vector Search (Matching Engine):
1. Create an Index with `distanceMeasureType=DOT_PRODUCT_DISTANCE` and
   `featureNormType=UNIT_L2_NORM`.
2. Batch-upsert your embeddings via `IndexDatapoints.upsert_datapoints()`.
3. Deploy the index to an `IndexEndpoint`.
4. Replace `VectorStore.search()` with a call to
   `index_endpoint.match(deployed_index_id=..., queries=[vec], num_neighbors=k)`.
The rest of the pipeline stays unchanged.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    title: str
    text: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float          # cosine similarity in [-1, 1]; higher = more similar
    rank: int


# ---------------------------------------------------------------------------
# FAISS-backed store
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    Stores document embeddings in a FAISS flat inner-product index.
    Because vectors are L2-normalised, IP == cosine similarity.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)   # exact, no approximation
        self._documents: List[Document] = []

    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Parameters
        ----------
        documents  : list of Document objects (length N)
        embeddings : float32 array, shape (N, D), assumed L2-normalised
        """
        assert len(documents) == embeddings.shape[0], "doc/embedding count mismatch"
        self._index.add(embeddings)
        self._documents.extend(documents)

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[SearchResult]:
        """
        Parameters
        ----------
        query_vec : shape (D,), L2-normalised float32
        k         : number of results to return

        Returns
        -------
        List of SearchResult sorted by descending cosine similarity.
        """
        q = query_vec.reshape(1, -1)
        scores, indices = self._index.search(q, k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:        # FAISS returns -1 when fewer than k docs exist
                continue
            results.append(SearchResult(
                document=self._documents[idx],
                score=float(score),
                rank=rank,
            ))
        return results

    def __len__(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Pure-NumPy fallback (no FAISS dependency)
# ---------------------------------------------------------------------------

class NumpyVectorStore:
    """
    Cosine-similarity store implemented with NumPy matrix multiplication.
    Suitable for small corpora and environments where FAISS is unavailable.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._embeddings: Optional[np.ndarray] = None   # shape (N, D)
        self._documents: List[Document] = []

    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        assert len(documents) == embeddings.shape[0]
        if self._embeddings is None:
            self._embeddings = embeddings.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        self._documents.extend(documents)

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[SearchResult]:
        if self._embeddings is None or len(self._documents) == 0:
            return []
        # dot product on normalised vectors == cosine similarity
        scores = self._embeddings @ query_vec          # shape (N,)
        top_k_idx = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_k_idx, start=1):
            results.append(SearchResult(
                document=self._documents[idx],
                score=float(scores[idx]),
                rank=rank,
            ))
        return results

    def __len__(self) -> int:
        return len(self._documents)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vector_store(dimension: int) -> "FAISSVectorStore | NumpyVectorStore":
    """Return FAISS store if available, else pure-NumPy fallback."""
    if _FAISS_AVAILABLE:
        return FAISSVectorStore(dimension)
    return NumpyVectorStore(dimension)