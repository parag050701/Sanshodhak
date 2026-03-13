#!/usr/bin/env python3
"""
Corpus Expansion Script for Sanshodhak
=======================================
Uses the existing IngestionEngine to expand the corpus from ~36 papers
to 500+ papers across 3 domains, then rebuilds the FAISS and BM25 indices.

Target corpus:
  - 200 RAG/IR papers  (queries: RAG, hybrid retrieval, knowledge graph)
  - 200 BioMed papers  (queries: biomedical NLP, clinical NLP, medical imaging AI)
  - 100 Legal/Finance  (queries: legal NLP, financial NLP, document understanding)
  Total: ~500+ papers

Usage:
  conda activate sanshodhak
  python paper-intel/expand_corpus.py

This will:
1. Download papers into research_papers/ (per domain subfolder)
2. Extract text and chunks
3. Rebuild FAISS + BM25 indices in rag_index/
4. Print corpus statistics
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "research_papers"

EXPANSION_PLAN = [
    {
        "domain": "RAG_IR",
        "queries": [
            "retrieval-augmented generation RAG survey",
            "hybrid retrieval dense sparse BM25",
            "knowledge graph RAG retrieval",
            "reciprocal rank fusion information retrieval",
            "FAISS dense retrieval sentence embeddings",
        ],
        "target_papers": 10,
        "subdir": "rag_ir"
    },
    {
        "domain": "BioMed",
        "queries": [
            "biomedical named entity recognition NLP",
            "clinical NLP electronic health records",
            "medical imaging deep learning",
            "PubMed biomedical text mining transformers",
        ],
        "target_papers": 200,
        "subdir": "biomedical"
    },
    {
        "domain": "Legal_Finance",
        "queries": [
            "legal NLP contract analysis document",
            "financial NLP text mining earnings reports",
            "legal document understanding extraction",
        ],
        "target_papers": 10,
        "subdir": "legal_finance"
    }
]


async def run_domain_ingestion(domain_config: dict):
    """Run ingestion for a single domain."""
    try:
        from ingestion.ingestion_engine import IngestionEngine
    except Exception as e:
        logger.error(f"Could not import IngestionEngine: {e}")
        return []

    domain = domain_config["domain"]
    queries = domain_config["queries"]
    target = domain_config["target_papers"]
    subdir = PAPERS_DIR / domain_config["subdir"]
    subdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"INGESTING DOMAIN: {domain} (target: {target} papers)")
    logger.info(f"Output: {subdir}")
    logger.info(f"{'='*60}")

    all_results = []
    per_query_target = max(target // len(queries), 20)

    for q in queries:
        logger.info(f"\nQuery: '{q}' (target: {per_query_target} papers)")
        engine = IngestionEngine(
            query=q,
            required_count=per_query_target,
            output_dir=str(subdir),
            prefer_open_access=True,
            max_iterations=2
        )
        try:
            results = await engine.run()
            logger.info(f"  Got {results['total_downloaded']} papers")
            all_results.append(results)
        except Exception as e:
            logger.error(f"  Ingestion failed for query '{q}': {e}")
        finally:
            engine.close()

    return all_results


def rebuild_index():
    """Rebuild FAISS and BM25 index from all papers in the research_papers directory."""
    logger.info("\n" + "="*60)
    logger.info("REBUILDING FAISS + BM25 INDEX")
    logger.info("="*60)

    try:
        # Show what we have
        pdf_count = 0
        for subdir in PAPERS_DIR.iterdir():
            if subdir.is_dir():
                pdfs = list(subdir.glob("*.pdf"))
                pdf_count += len(pdfs)
                logger.info(f"  {subdir.name}: {len(pdfs)} PDFs")

        logger.info(f"\nTotal PDFs available: {pdf_count}")

        # Try importing the main RAG module to rebuild
        sys.path.insert(0, str(BASE_DIR))
        try:
            from ollama_rag import OllamaRAG
            rag = OllamaRAG()
            rag.build_index(str(PAPERS_DIR))
            logger.info("✅ FAISS index rebuilt successfully via OllamaRAG")
        except Exception as e:
            logger.warning(f"Could not rebuild via OllamaRAG: {e}")
            logger.info("Run the Sanshodhak web UI and click 'Rebuild Index' to rebuild manually.")

    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")


def print_corpus_stats():
    """Print statistics about the expanded corpus."""
    stats = {"domains": {}, "total_pdfs": 0}

    for domain_cfg in EXPANSION_PLAN:
        subdir = PAPERS_DIR / domain_cfg["subdir"]
        if subdir.exists():
            pdfs = list(subdir.glob("*.pdf"))
            stats["domains"][domain_cfg["domain"]] = len(pdfs)
            stats["total_pdfs"] += len(pdfs)

    # Existing research_papers at root
    root_pdfs = list(PAPERS_DIR.glob("*.pdf"))
    stats["domains"]["existing_root"] = len(root_pdfs)
    stats["total_pdfs"] += len(root_pdfs)

    logger.info("\n" + "="*60)
    logger.info("CORPUS EXPANSION STATISTICS")
    logger.info("="*60)
    for domain, count in stats["domains"].items():
        logger.info(f"  {domain}: {count} papers")
    logger.info(f"\n  TOTAL: {stats['total_pdfs']} papers")

    audit_target = "✅ MEETS AUDIT REQUIREMENT (500+)" if stats["total_pdfs"] >= 500 else f"⚠️ {500 - stats['total_pdfs']} more needed to meet target"
    logger.info(f"  {audit_target}")

    with open(BASE_DIR / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\n  Stats saved to corpus_stats.json")
    return stats


async def main():
    logger.info("Sanshodhak Corpus Expansion Script")
    logger.info(f"Working directory: {BASE_DIR}")

    results = []
    for domain_cfg in EXPANSION_PLAN:
        domain_results = await run_domain_ingestion(domain_cfg)
        results.extend(domain_results)

    rebuild_index()
    stats = print_corpus_stats()

    with open(BASE_DIR / "expansion_results.json", "w") as f:
        json.dump({"summary": stats, "detail": results}, f, indent=2, default=str)

    logger.info("\n✅ Corpus expansion complete.")


if __name__ == "__main__":
    asyncio.run(main())
