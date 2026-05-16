"""
benchmark.py — Entry point for the Strategy A vs Strategy B comparison.

Usage:
    python benchmark.py              # prints table + saves JSON
    python benchmark.py --json-only  # prints raw JSON only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.corpus import DOCUMENTS
from src.pipeline import RAGPipeline, print_report

# ---------------------------------------------------------------------------
# Benchmark queries (required: ≥ 3 complex queries)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What strategies are used to ensure data is not lost during a failure?",
    "How are external clients prevented from overwhelming the backend services?",
]


def run_benchmark(output_path: Path = Path("retrieval_benchmark.json")) -> list:
    pipeline = RAGPipeline()
    pipeline.ingest(DOCUMENTS)

    report = pipeline.benchmark(BENCHMARK_QUERIES, k=3)
    return report


def save_json(report: list, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2))
    print(f"\n[benchmark] JSON report saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="RAG Strategy A vs B Benchmark")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only")
    parser.add_argument(
        "--output", default="retrieval_benchmark.json", help="JSON output file"
    )
    args = parser.parse_args()

    report = run_benchmark()

    if args.json_only:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
        save_json(report, Path(args.output))


if __name__ == "__main__":
    main()