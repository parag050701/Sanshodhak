#!/usr/bin/env python3
"""
Smoke test for ingestion system with health reporting.

Usage:
    python smoke_test.py
    python smoke_test.py --query "machine learning"
    python smoke_test.py --only-robust  # Skip unstable sources
"""
import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.discovery.search_engine import SearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Smoke test for ingestion system')
    parser.add_argument('--query', default='retrieval augmented generation', help='Search query')
    parser.add_argument('--limit', type=int, default=10, help='Results per source')
    parser.add_argument('--min-year', type=int, default=2020, help='Minimum publication year')
    parser.add_argument('--only-robust', action='store_true', help='Only use robust sources')
    parser.add_argument('--enable-unstable', action='store_true', help='Enable unstable sources')
    args = parser.parse_args()

    # Set env flags if requested
    if args.enable_unstable:
        os.environ['ENABLE_UNSTABLE_SOURCES'] = 'true'
        logger.info("Enabled unstable sources")

    query = args.query
    logger.info("=" * 80)
    logger.info(f"SMOKE TEST: Query='{query}'")
    logger.info(f"Only robust sources: {args.only_robust}")
    logger.info(f"Unstable sources enabled: {args.enable_unstable}")
    logger.info("=" * 80)

    try:
        # Initialize search engine
        engine = SearchEngine(
            unpaywall_email=os.getenv('CROSSREF_EMAIL', 'user@example.com'),
            enable_core=True,
            only_robust=args.only_robust
        )

        logger.info("\nStarting open-source search...")
        open_papers = engine.search_open_source(
            query=query,
            limit_per_source=args.limit,
            min_year=args.min_year,
            parallel=True
        )

        logger.info(f"\nOpen-source results: {len(open_papers)} papers")

        logger.info("\nStarting closed-access search...")
        closed_papers = engine.search_closed_access(
            query=query,
            limit=args.limit * 2,
            min_year=args.min_year
        )

        logger.info(f"Closed-access results: {len(closed_papers)} papers")

        # Combined results
        from ingestion.discovery.doi_utils import deduplicate_papers
        all_papers = open_papers + closed_papers
        unique_papers = deduplicate_papers(all_papers)

        logger.info("\n" + "=" * 80)
        logger.info(f"TOTAL RESULTS: {len(unique_papers)} unique papers")
        logger.info("=" * 80)

        # Show sample papers
        if unique_papers:
            logger.info("\nSample papers:")
            for i, paper in enumerate(unique_papers[:5], 1):
                logger.info(f"\n{i}. {paper.title[:80]}...")
                logger.info(f"   Authors: {', '.join(paper.authors[:3]) if paper.authors else 'Unknown'}")
                logger.info(f"   Year: {paper.year}, Source: {paper.source}")
                logger.info(f"   Open Access: {paper.is_open_access}")
                if paper.doi:
                    logger.info(f"   DOI: {paper.doi}")

        # Health report is printed by search_unified, but let's also get it
        health_report = engine.health.get_health_report()
        logger.info("\n" + "=" * 80)
        logger.info("SOURCE HEALTH SUMMARY")
        logger.info("=" * 80)

        for name, health in sorted(health_report['sources'].items()):
            status = "✓" if health['is_healthy'] else "✗"
            logger.info(
                f"{status} {name:20s} | "
                f"Success: {health['success_rate']:.1%} | "
                f"Circuit: {health['circuit_state']} | "
                f"Last error: {health['last_failure_reason'] or 'None'}"
            )

        summary = health_report['summary']
        logger.info(f"\nSummary: {summary['healthy_sources']}/{summary['total_sources']} healthy, "
                   f"{summary['open_circuits']} open circuits")

        logger.info("\n" + "=" * 80)
        logger.info("SMOKE TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception("Smoke test failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
