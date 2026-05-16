"""
query_expansion.py — Mock Vertex AI GenerativeModel for query rewriting.

In production this would call:
    vertexai.generative_models.GenerativeModel("gemini-1.5-flash")

The mock returns deterministic, semantically richer expansions so the
benchmark is reproducible without GCP credentials or API spend.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Mock response object (mirrors the real GenerativeModel response surface)
# ---------------------------------------------------------------------------

class _MockGenerationResponse:
    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# Mock GenerativeModel
# ---------------------------------------------------------------------------

# Keyword → expansion rules.  Keys are lowercase substrings of the query.
_EXPANSION_RULES: list[tuple[str, str]] = [
    (
        "peak load",
        (
            "How does the system handle peak load, traffic spikes, high throughput, "
            "and sudden demand surges? Include auto-scaling, load balancing, circuit "
            "breakers, rate limiting, and backpressure mechanisms."
        ),
    ),
    (
        "scal",
        (
            "What horizontal and vertical scaling strategies are used? "
            "Include auto-scaling groups, stateless services, database read replicas, "
            "sharding, partitioning, and connection pooling."
        ),
    ),
    (
        "cach",
        (
            "How is caching implemented across CDN, application, and database layers? "
            "Explain cache invalidation strategies, TTL policies, write-through vs "
            "eventual consistency, and Redis eviction policies."
        ),
    ),
    (
        "databas",
        (
            "How is the database layer designed for scale and resilience? "
            "Include sharding, partitioning, replication, failover, connection pooling, "
            "and point-in-time recovery strategies."
        ),
    ),
    (
        "secur",
        (
            "What security mechanisms are in place? Include zero-trust architecture, "
            "mutual TLS, JWT authentication, RBAC, secrets management, and certificate rotation."
        ),
    ),
    (
        "monitor",
        (
            "How is the system monitored and observed? Include metrics collection, "
            "distributed tracing, structured logging, alerting, and SLO burn-rate tracking."
        ),
    ),
    (
        "messag",
        (
            "How does the system use asynchronous messaging and event-driven patterns? "
            "Include Kafka topics, consumer groups, dead-letter queues, idempotency, "
            "and worker auto-scaling."
        ),
    ),
    (
        "microservice",
        (
            "How do microservices communicate? Include synchronous REST, gRPC, service mesh, "
            "service discovery, mutual TLS, canary deployments, and traffic policies."
        ),
    ),
    (
        "fault",
        (
            "How does the system achieve fault tolerance and high availability? "
            "Include replication, automated failover, circuit breakers, WAL archiving, "
            "and RPO/RTO objectives."
        ),
    ),
    (
        "rate limit",
        (
            "How does the system enforce rate limiting and throttling? "
            "Include token-bucket algorithms, per-client quotas, burst allowances, "
            "HTTP 429 responses, and distributed Redis counters."
        ),
    ),
]

_FALLBACK_TEMPLATE = (
    "Expand the following technical query to be more specific and embedding-friendly. "
    "Include related technical concepts, synonyms, and implementation details: {query}"
)


class MockVertexAIGenerativeModel:
    """
    Drop-in mock for vertexai.generative_models.GenerativeModel.

    Expansion rules are deterministic so benchmarks are reproducible.
    In production, replace generate_content() with the real SDK call.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name

    def generate_content(self, prompt: str) -> _MockGenerationResponse:
        """
        Simulate query expansion.

        The prompt is expected to contain the original user query.
        We match keywords and return a richer version.
        """
        # Extract the raw query from the prompt (last line or full prompt)
        raw_query = self._extract_query(prompt)
        expanded = self._expand(raw_query)
        return _MockGenerationResponse(text=expanded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_query(prompt: str) -> str:
        """Pull the user query from a formatted prompt string."""
        # Look for lines after common prompt prefixes
        for pattern in [r"Query:\s*(.+)", r"query:\s*(.+)", r'"""(.+?)"""']:
            m = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip().strip('"').strip()
        # Fallback: use full prompt
        return prompt.strip()

    @staticmethod
    def _expand(query: str) -> str:
        """Return a semantically expanded version of the query."""
        q_lower = query.lower()
        for keyword, expansion in _EXPANSION_RULES:
            if keyword in q_lower:
                return expansion
        # Generic expansion
        return (
            f"{query} — include related concepts such as scalability, "
            "fault tolerance, performance optimisation, distributed systems patterns, "
            "and infrastructure automation."
        )


# ---------------------------------------------------------------------------
# Prompt builder + public helper
# ---------------------------------------------------------------------------

EXPANSION_PROMPT_TEMPLATE = """\
You are a search query optimisation assistant. Rewrite the user query below
into an embedding-friendly, semantically rich version that captures all
relevant technical concepts, synonyms, and related topics.

Original Query: "{query}"

Output only the rewritten query — no preamble, no explanation.
"""


def expand_query(
    model: MockVertexAIGenerativeModel,
    query: str,
) -> str:
    """
    Use the (mocked) generative model to rewrite a user query.

    Parameters
    ----------
    model : MockVertexAIGenerativeModel (or real GenerativeModel in production)
    query : raw user question

    Returns
    -------
    Expanded / rewritten query string ready for embedding.
    """
    prompt = EXPANSION_PROMPT_TEMPLATE.format(query=query)
    response = model.generate_content(prompt)
    return response.text.strip()


def get_generative_model(model_name: str = "gemini-1.5-flash") -> MockVertexAIGenerativeModel:
    """Factory: returns the (mocked) Vertex AI GenerativeModel."""
    return MockVertexAIGenerativeModel(model_name=model_name)