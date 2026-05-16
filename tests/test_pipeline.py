"""
tests/test_pipeline.py — Integration tests for the RAGPipeline orchestrator.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from src.pipeline import RAGPipeline, _compute_overlap, print_report
from src.storage import SearchResult, Document


# ---------------------------------------------------------------------------
# Sample corpus (small, fast)
# ---------------------------------------------------------------------------

MINI_CORPUS = [
    {"id": "p1", "title": "Load Balancing", "text": "Load balancers distribute peak traffic across server pools."},
    {"id": "p2", "title": "Caching",         "text": "CDN and Redis caching reduce latency and backend load."},
    {"id": "p3", "title": "Replication",     "text": "WAL streaming replication ensures fault tolerance and durability."},
]


@pytest.fixture(scope="module")
def pipeline():
    p = RAGPipeline()
    p.ingest(MINI_CORPUS)
    return p


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_store_size_after_ingest(self, pipeline):
        assert pipeline.store_size == len(MINI_CORPUS)

    def test_query_without_ingest_raises(self):
        p = RAGPipeline()
        with pytest.raises(RuntimeError, match="No documents ingested"):
            p.query_strategy_a("test")

    def test_double_ingest_replaces_store(self):
        """
        Each call to ingest() re-fits the offline engine and recreates the store.
        The second ingest replaces the first (not accumulates).
        """
        p = RAGPipeline()
        p.ingest(MINI_CORPUS)
        assert p.store_size == len(MINI_CORPUS)
        p.ingest(MINI_CORPUS)
        # Store is reset on each ingest (by design — prevents dimension mismatch)
        assert p.store_size == len(MINI_CORPUS)


# ---------------------------------------------------------------------------
# Strategy A via pipeline
# ---------------------------------------------------------------------------

class TestPipelineStrategyA:
    def test_returns_list(self, pipeline):
        results = pipeline.query_strategy_a("traffic and load balancing", k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_top_result_is_load_doc(self, pipeline):
        results = pipeline.query_strategy_a("peak traffic load balancing", k=1)
        assert results[0].document.id == "p1"


# ---------------------------------------------------------------------------
# Strategy B via pipeline
# ---------------------------------------------------------------------------

class TestPipelineStrategyB:
    def test_returns_tuple(self, pipeline):
        expanded, results = pipeline.query_strategy_b("peak load", k=2)
        assert isinstance(expanded, str)
        assert len(results) == 2

    def test_expanded_query_longer_than_original(self, pipeline):
        original = "cache"
        expanded, _ = pipeline.query_strategy_b(original, k=1)
        assert len(expanded) > len(original)


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

class TestBenchmark:
    QUERIES = [
        "How does the system handle peak load?",
        "What prevents data loss during failure?",
    ]

    def test_benchmark_returns_list(self, pipeline):
        report = pipeline.benchmark(self.QUERIES, k=3)
        assert isinstance(report, list)
        assert len(report) == len(self.QUERIES)

    def test_report_entry_structure(self, pipeline):
        report = pipeline.benchmark(self.QUERIES, k=1)
        entry = report[0]
        assert "original_query" in entry
        assert "expanded_query" in entry
        assert "strategy_a" in entry
        assert "strategy_b" in entry
        assert "overlap" in entry

    def test_result_entries_have_required_fields(self, pipeline):
        report = pipeline.benchmark(self.QUERIES, k=2)
        for entry in report:
            for result in entry["strategy_a"] + entry["strategy_b"]:
                assert "rank" in result
                assert "doc_id" in result
                assert "title" in result
                assert "score" in result
                assert "snippet" in result

    def test_overlap_computation(self, pipeline):
        report = pipeline.benchmark(self.QUERIES, k=3)
        for entry in report:
            ov = entry["overlap"]
            assert 0 <= ov["overlap_count"] <= 3
            assert 0.0 <= ov["jaccard"] <= 1.0

    def test_report_is_json_serialisable(self, pipeline):
        report = pipeline.benchmark(self.QUERIES, k=2)
        serialised = json.dumps(report)   # should not raise
        parsed = json.loads(serialised)
        assert len(parsed) == len(self.QUERIES)


# ---------------------------------------------------------------------------
# Overlap helper
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def _make_results(self, ids):
        return [
            SearchResult(document=Document(id=i, title="", text=""), score=0.9, rank=r)
            for r, i in enumerate(ids, start=1)
        ]

    def test_full_overlap(self):
        a = self._make_results(["d1", "d2", "d3"])
        b = self._make_results(["d1", "d2", "d3"])
        ov = _compute_overlap(a, b)
        assert ov["overlap_count"] == 3
        assert ov["jaccard"] == 1.0

    def test_no_overlap(self):
        a = self._make_results(["d1", "d2"])
        b = self._make_results(["d3", "d4"])
        ov = _compute_overlap(a, b)
        assert ov["overlap_count"] == 0
        assert ov["jaccard"] == 0.0

    def test_partial_overlap(self):
        a = self._make_results(["d1", "d2", "d3"])
        b = self._make_results(["d1", "d4", "d5"])
        ov = _compute_overlap(a, b)
        assert ov["overlap_count"] == 1
        # union = {d1,d2,d3,d4,d5} = 5; jaccard = 1/5 = 0.2
        assert abs(ov["jaccard"] - 0.2) < 0.001