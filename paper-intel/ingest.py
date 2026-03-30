#!/usr/bin/env python3
"""
CLI wrapper for Sanshodhak ingestion.

Example:
  python ingest.py --query "retrieval augmented generation" \
    --max_results 3000 --output data/corpus/rag.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from ingestion.ingestion_engine import IngestionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper ingestion for one query")
    parser.add_argument("--query", required=True, help="Research query")
    parser.add_argument("--max_results", type=int, default=3000, help="Target papers")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--min_year", type=int, default=2015, help="Minimum publication year")
    parser.add_argument("--max_iterations", type=int, default=3, help="Max expansion iterations")
    parser.add_argument("--prefer_open_access", action="store_true", help="Boost OA papers in ranking")
    parser.add_argument("--download_dir", default="paper-intel/papers", help="PDF download directory")
    return parser.parse_args()


async def run_ingestion(args: argparse.Namespace) -> dict:
    engine = IngestionEngine(
        query=args.query,
        required_count=args.max_results,
        output_dir=args.download_dir,
        min_year=args.min_year,
        max_iterations=args.max_iterations,
        prefer_open_access=args.prefer_open_access,
    )
    try:
        return await engine.run()
    finally:
        engine.close()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(run_ingestion(args))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[OK] Ingestion complete")
    print(f"  Query: {args.query}")
    print(f"  Total found: {result.get('total_found', 0)}")
    print(f"  Downloaded: {result.get('total_downloaded', 0)}")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
