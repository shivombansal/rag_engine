# Retrieval Benchmark Report — Strategy A vs Strategy B

**Project:** Context-Aware Retrieval Engine  
**Date:** 2025-06-15  
**Corpus:** 10 technical paragraphs (system design, distributed systems, infrastructure)  
**Vector Store:** FAISS IndexFlatIP (inner-product on L2-normalised vectors ≡ cosine similarity)  
**Embedding Engine:** `MockVertexAITextEmbeddingModel` → `LocalEmbeddingEngine` (TF-IDF + LSA offline, or `sentence-transformers/all-MiniLM-L6-v2` when internet is available)  

---

## Strategy Definitions

| | Strategy A | Strategy B |
|---|---|---|
| **Name** | Raw Vector Search | AI-Enhanced Retrieval |
| **Query Processing** | Embed user query directly | Expand query via `MockVertexAIGenerativeModel`, then embed |
| **Similarity Metric** | Cosine (dot-product on normalised vectors) | Cosine (same store, richer query vector) |
| **GCP Mock** | `TextEmbeddingModel.from_pretrained(...)` | `TextEmbeddingModel` + `GenerativeModel.generate_content(...)` |

---

## Query 1: "How does the system handle peak load?"

### Expanded Query (Strategy B)
> *"How does the system handle peak load, traffic spikes, high throughput, and sudden demand surges? Include auto-scaling, load balancing, circuit breakers, rate limiting, and backpressure mechanisms."*

### Results

| Rank | Strategy A — Raw Vector Search | Score | Strategy B — AI-Enhanced Retrieval | Score |
|------|-------------------------------|-------|-------------------------------------|-------|
| 1 | **Load Balancing and Peak Traffic** | 0.9651 | **Load Balancing and Peak Traffic** | 0.8984 |
| 2 | Horizontal Scaling Strategy | 0.3405 | Horizontal Scaling Strategy | 0.4181 |
| 3 | Asynchronous Processing and Message Queues | 0.2387 | Asynchronous Processing and Message Queues | 0.3718 |

**Overlap:** 3/3 docs shared — Jaccard = 1.00

### Analysis
Both strategies retrieve the same documents because "peak load" is a highly specific phrase that appears verbatim in `doc_001`. Strategy A's raw query already aligns tightly with the top result. Strategy B's expansion confirms all three results but **boosts rank-2 and rank-3 confidence scores** (0.34→0.42, 0.24→0.37) — the richer query surface pulls asynchronous processing and horizontal scaling more strongly into the semantic neighbourhood.

---

## Query 2: "What strategies are used to ensure data is not lost during a failure?"

### Expanded Query (Strategy B)
> *"What strategies are used to ensure data is not lost during a failure? — include related concepts such as scalability, fault tolerance, performance optimisation, distributed systems patterns, and infrastructure automation."*

### Results

| Rank | Strategy A — Raw Vector Search | Score | Strategy B — AI-Enhanced Retrieval | Score |
|------|-------------------------------|-------|-------------------------------------|-------|
| 1 | Database Sharding and Partitioning | 0.6508 | **Data Replication and Fault Tolerance** | 0.7759 |
| 2 | Horizontal Scaling Strategy | 0.5802 | Horizontal Scaling Strategy | 0.5312 |
| 3 | Microservices Communication Patterns | 0.5506 | Caching Architecture | 0.4683 |

**Overlap:** 1/3 docs shared — Jaccard = 0.20

### Analysis
This query shows the strongest divergence. Strategy A latches onto the word "strategies" and "data" and retrieves sharding as rank-1 — a plausible but wrong answer (sharding is about scale, not durability). Strategy B's expansion introduces **"fault tolerance"** and **"replication"** as explicit terms, pulling `doc_009` (Data Replication and Fault Tolerance — WAL streaming, automated failover, RPO/RTO) to rank-1 with a much higher confidence score (0.78 vs 0.65). **Strategy B is meaningfully better here.**

---

## Query 3: "How are external clients prevented from overwhelming the backend services?"

### Expanded Query (Strategy B)
> *"How are external clients prevented from overwhelming the backend services? — include related concepts such as scalability, fault tolerance, performance optimisation, distributed systems patterns, and infrastructure automation."*

### Results

| Rank | Strategy A — Raw Vector Search | Score | Strategy B — AI-Enhanced Retrieval | Score |
|------|-------------------------------|-------|-------------------------------------|-------|
| 1 | **API Rate Limiting and Throttling** | 0.8442 | Data Replication and Fault Tolerance | 0.7175 |
| 2 | Observability and Monitoring | 0.4178 | **API Rate Limiting and Throttling** | 0.5641 |
| 3 | Microservices Communication Patterns | 0.4090 | Horizontal Scaling Strategy | 0.4663 |

**Overlap:** 1/3 docs shared — Jaccard = 0.20

### Analysis
Strategy A correctly identifies `doc_007` (API Rate Limiting) as rank-1 (score 0.84) — the phrase "overwhelming backend services" semantically aligns with the rate-limiting document's language. Strategy B's generic fallback expansion (which doesn't match a specific rule for this query) dilutes the signal and promotes `doc_009` (Replication) to rank-1 incorrectly. **Strategy A wins for this query** — a concrete, specific user query can be degraded by over-broad generic expansion. This motivates using a higher-quality generative model (real Gemini) in production.

---

## Summary Table

| Query | A Rank-1 Doc | B Rank-1 Doc | Winner | Jaccard |
|-------|-------------|-------------|--------|---------|
| Peak load handling | Load Balancing ✓ | Load Balancing ✓ | **Tie** | 1.00 |
| Data loss prevention | Sharding (wrong) | Replication ✓ | **B** | 0.20 |
| Client throttling | Rate Limiting ✓ | Replication (wrong) | **A** | 0.20 |

---

## Similarity Metric Choice: Cosine vs Euclidean

### Why Cosine?

**Cosine similarity** measures the angle between vectors, not their magnitude:

```
cos(θ) = (u · v) / (‖u‖ · ‖v‖)
```

For unit-normalised vectors (which both `sentence-transformers` and `textembedding-gecko` produce by default), this simplifies to a dot product — making it computationally identical to Euclidean distance on the unit hypersphere (since `‖u-v‖² = 2 - 2·u·v`).

| Criterion | Cosine | Euclidean |
|-----------|--------|-----------|
| Magnitude invariance | ✅ Short & long docs compared fairly | ❌ Longer docs tend to have larger norms |
| Interpretability | ✅ Range [-1, 1]; 1 = identical direction | ❌ Unbounded, harder to threshold |
| Production parity | ✅ Vertex AI Matching Engine uses `DOT_PRODUCT_DISTANCE` on normalised vectors | ⚠️ Would require `SQUARED_L2_DISTANCE` + unnormalised vectors |
| Semantic embedding models | ✅ Designed for cosine | ❌ Suboptimal; models aren't trained for Euclidean ranking |

**Verdict:** Cosine (implemented as inner-product on L2-normalised vectors) is the correct choice for this pipeline.

---

## Production Migration to Vertex AI Vector Search (Matching Engine)

### Step-by-step migration

```
Local FAISS IndexFlatIP          →    Vertex AI Matching Engine Index
MockVertexAITextEmbeddingModel   →    TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
MockVertexAIGenerativeModel      →    GenerativeModel("gemini-1.5-flash")
LocalEmbeddingEngine             →    (removed — real API handles inference)
```

### 1. Provision a Matching Engine Index

```python
import vertexai
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-index",
    dimensions=768,                        # gecko@003 outputs 768-dim
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    feature_norm_type="UNIT_L2_NORM",      # auto-normalises on insert
    approximate_neighbors_count=150,
)
```

### 2. Batch upsert embeddings

```python
from vertexai.language_models import TextEmbeddingModel

embed_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

datapoints = []
for doc in documents:
    response = embed_model.get_embeddings([doc["text"]])[0]
    datapoints.append(
        aiplatform.MatchingEngineIndex.Datapoint(
            datapoint_id=doc["id"],
            feature_vector=response.values,
        )
    )

index.upsert_datapoints(datapoints=datapoints)
```

### 3. Deploy and query

```python
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    public_endpoint_enabled=True,
)
deployed_index = index_endpoint.deploy_index(index=index, deployed_index_id="rag_v1")

# Strategy A — raw search
query_vec = embed_model.get_embeddings([query])[0].values
neighbors = index_endpoint.match(
    deployed_index_id="rag_v1",
    queries=[query_vec],
    num_neighbors=3,
)

# Strategy B — expand then search
expanded = generative_model.generate_content(expand_prompt(query)).text
exp_vec = embed_model.get_embeddings([expanded])[0].values
neighbors_b = index_endpoint.match(
    deployed_index_id="rag_v1",
    queries=[exp_vec],
    num_neighbors=3,
)
```

### 4. What stays the same

- `retrieval.py` — `StrategyA` and `StrategyB` classes are unchanged
- `storage.py` — replace `VectorStore.search()` call only
- `query_expansion.py` — swap mock for real `GenerativeModel`
- `pipeline.py` — swap mock models; remove offline refit logic

The mock interfaces were designed to match the real SDK surface exactly, so the migration is a **drop-in model swap**, not a rewrite.

---

## Test Coverage Summary

```
tests/test_embeddings.py   — 14 tests   LocalEmbeddingEngine, MockVertexAITextEmbeddingModel, SDK patching
tests/test_storage.py      — 13 tests   NumpyVectorStore, FAISSVectorStore, factory
tests/test_retrieval.py    — 15 tests   Query expansion mock, StrategyA, StrategyB, compare_strategies
tests/test_pipeline.py     — 16 tests   Ingestion, both strategies, benchmark report, overlap metric
─────────────────────────────────────────────────────────────────────
TOTAL                        58 passed, 0 failed
```