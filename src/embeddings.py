"""
embeddings.py — Embedding logic with a mock Vertex AI TextEmbeddingModel.

Real production code would call:
    vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

Here we mock that interface and back it with sentence-transformers locally,
with a TF-IDF + LSA offline fallback for environments without internet access.
"""

from __future__ import annotations

import numpy as np
from typing import List
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Optional: sentence-transformers (preferred when available + internet)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Offline fallback: TF-IDF + LSA via scikit-learn
# ---------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import Normalizer

# Seed corpus for fitting the vectorizer (broad domain vocabulary)
_SEED_CORPUS = [
    "load balancing auto scaling peak traffic server distribution circuit breaker failover",
    "horizontal scaling stateless application database replica connection pool throughput",
    "caching CDN redis eviction TTL invalidation write through consistency latency",
    "database sharding partitioning consistent hashing read replica write primary",
    "kafka message queue consumer producer dead letter idempotency backpressure worker",
    "monitoring metrics logs traces prometheus grafana alerting SLO latency percentile",
    "rate limiting token bucket quota throttle burst allowance sliding window gateway",
    "microservices gRPC REST service mesh istio canary deployment service discovery consul",
    "replication WAL failover patroni etcd RPO RTO durability recovery snapshot backup",
    "security zero trust mutual TLS JWT RBAC vault secrets authentication certificate",
    "fault tolerance resilience high availability redundancy disaster recovery",
    "query expansion semantic search retrieval augmented generation vector embedding cosine",
]

# n_components must be < min(n_samples, n_features); 10 is safe for 12 seed docs
_SVD_COMPONENTS = 10


class OfflineTFIDFEngine:
    """
    TF-IDF + Truncated SVD (Latent Semantic Analysis) embedding engine.

    Works entirely offline — no model downloads required.
    Semantically captures keyword overlap and latent topics in technical text.
    """

    def __init__(self, n_components: int = _SVD_COMPONENTS):
        self._pipeline = SKPipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("svd",   TruncatedSVD(n_components=n_components, random_state=42)),
            ("norm",  Normalizer(norm="l2")),
        ])
        self._pipeline.fit(_SEED_CORPUS)
        # Get actual output dimension (may be < n_components if vocab is small)
        sample = self._pipeline.transform([_SEED_CORPUS[0]])
        self.dimension: int = sample.shape[1]

    def refit(self, extra_texts: List[str]) -> None:
        """Re-fit on seed + document corpus for better vocabulary coverage."""
        self._pipeline.fit(_SEED_CORPUS + extra_texts)
        sample = self._pipeline.transform([_SEED_CORPUS[0]])
        self.dimension = sample.shape[1]

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return shape (N, D) float32 L2-normalised embeddings."""
        return self._pipeline.transform(texts).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class LocalEmbeddingEngine:
    """
    Tries sentence-transformers first; falls back to OfflineTFIDFEngine.

    Returns L2-normalised vectors so dot-product == cosine similarity,
    matching Vertex AI textembedding-gecko behaviour.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        self._use_st = False
        if _ST_AVAILABLE:
            try:
                self._backend = SentenceTransformer(model_name)
                self._use_st = True
                self.dimension: int = self._backend.get_sentence_embedding_dimension()
                return
            except Exception:
                pass
        # Offline fallback
        self._backend = OfflineTFIDFEngine()
        self.dimension = self._backend.dimension

    def refit_if_offline(self, texts: List[str]) -> None:
        """Re-fit offline engine with document corpus for better embeddings."""
        if not self._use_st:
            self._backend.refit(texts)
            self.dimension = self._backend.dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._use_st:
            return self._backend.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
        return self._backend.encode(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Vertex AI mock — mirrors the real SDK surface so tests can patch it
# ---------------------------------------------------------------------------

class _MockTextEmbeddingResponse:
    """Simulates vertexai TextEmbeddingModel response object."""

    def __init__(self, values: List[float]):
        self.values = values


class MockVertexAITextEmbeddingModel:
    """
    Drop-in mock for:
        vertexai.language_models.TextEmbeddingModel.from_pretrained(...)

    It delegates to LocalEmbeddingEngine, so results are semantically real
    while no GCP credentials are required.
    """

    def __init__(self, model_name: str = "textembedding-gecko@003"):
        self._engine = LocalEmbeddingEngine()
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockVertexAITextEmbeddingModel":
        """Mirrors the real SDK class method."""
        return cls(model_name=model_name)

    def get_embeddings(self, texts: List[str]) -> List[_MockTextEmbeddingResponse]:
        """Return a list of response objects, one per input text."""
        vecs = self._engine.encode(texts)
        return [_MockTextEmbeddingResponse(vec.tolist()) for vec in vecs]

    @property
    def dimension(self) -> int:
        return self._engine.dimension

    def refit_if_offline(self, texts: List[str]) -> None:
        self._engine.refit_if_offline(texts)


# ---------------------------------------------------------------------------
# Public helpers used by the pipeline
# ---------------------------------------------------------------------------

def get_embedding_model() -> MockVertexAITextEmbeddingModel:
    """Factory: returns the (mocked) Vertex AI embedding model."""
    return MockVertexAITextEmbeddingModel.from_pretrained("textembedding-gecko@003")


def embed_texts(model: MockVertexAITextEmbeddingModel, texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings via the model.

    Returns
    -------
    np.ndarray  shape (N, D), float32, L2-normalised
    """
    responses = model.get_embeddings(texts)
    vecs = np.array([r.values for r in responses], dtype=np.float32)
    return vecs


def embed_query(model: MockVertexAITextEmbeddingModel, query: str) -> np.ndarray:
    """Embed a single query string → shape (D,)."""
    return embed_texts(model, [query])[0]