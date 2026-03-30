#!/usr/bin/env python3
"""
Priority-1 Complete Build Runbook

Automates the full Priority-1 pipeline:
1. Ingestion (10k corpus collection from multi-source APIs)
2. Merge & Deduplicate
3. GPU FAISS Index Building
4. NIM Question Generation (1000+ candidates)
5. Stratified 300-Question Sampling + IAA prep
6. Pipeline validation

Usage (dry-run):
  ./runbook_priority1.sh --dry-run

Usage (full execution):
  ./runbook_priority1.sh --corpus-json data/corpus.json --num-question-candidates 1000

All output stages in: data/priority1_build/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class Runbook:
    def __init__(self, work_dir: str = "data/priority1_build"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.stages = {}
        self.project_root = Path(__file__).resolve().parent
        logger.info(f"Runbook work directory: {self.work_dir}")

    def run_command(self, name: str, cmd: List[str], timeout: int = 3600) -> bool:
        """Run a command and track success/failure."""
        logger.info(f"\n{'='*70}")
        logger.info(f"[{name}]")
        logger.info(f"{'='*70}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=False,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            self.stages[name] = {
                "success": success,
                "timestamp": time.time(),
                "elapsed": time.time() - self.start_time,
            }
            if success:
                logger.info(f"✅ {name} completed")
            else:
                logger.error(f"❌ {name} failed with exit code {result.returncode}")
            return success
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {name} timed out after {timeout}s")
            self.stages[name] = {"success": False, "timeout": True}
            return False
        except Exception as e:
            logger.error(f"❌ {name} error: {e}")
            self.stages[name] = {"success": False, "error": str(e)}
            return False

    def collect_ingestion_sources(self, queries: List[str], max_results: int = 3000) -> bool:
        """Stage 1: Collect papers from multi-source ingestion."""
        logger.info("\n[STAGE 1] Multi-source paper collection")
        
        ingestion_dir = self.work_dir / "ingestion"
        ingestion_dir.mkdir(parents=True, exist_ok=True)

        for i, query in enumerate(queries, 1):
            output_file = ingestion_dir / f"query_{i:02d}_{query.replace(' ', '_')[:30]}.json"
            cmd = [
                "python", "ingest.py",
                "--query", query,
                "--max_results", str(max_results),
                "--output", str(output_file),
                "--min_year", "2020",
            ]
            if not self.run_command(f"Ingestion-Q{i}: {query[:40]}", cmd, timeout=7200):
                return False
        
        return True

    def merge_and_deduplicate(self, ingestion_dir: Optional[str] = None) -> bool:
        """Stage 2: Merge and deduplicate ingestion outputs."""
        logger.info("\n[STAGE 2] Merge & deduplicate")
        
        ing_dir = ingestion_dir or str(self.work_dir / "ingestion")
        merged_output = self.work_dir / "corpus_merged.json"
        stats_output = self.work_dir / "ingestion_stats.json"

        ingestion_files = sorted(Path(ing_dir).glob("*.json"))
        if not ingestion_files:
            logger.error(f"No ingestion JSON files found in {ing_dir}")
            return False

        cmd = [
            "python", "merge_corpus.py",
            "--inputs", *[str(p) for p in ingestion_files],
            "--output", str(merged_output),
            "--stats_output", str(stats_output),
        ]
        return self.run_command("Merge & Deduplicate", cmd)

    def build_gpu_embeddings(self, corpus_file: Optional[str] = None) -> bool:
        """Stage 3: Build GPU FAISS embeddings."""
        logger.info("\n[STAGE 3] GPU FAISS index building")
        
        corpus = corpus_file or str(self.work_dir / "corpus_merged.json")
        index_dir = self.work_dir / "indexes"

        cmd = [
            "python", "build_gpu_embeddings.py",
            "--corpus", corpus,
            "--output", str(index_dir),
            "--batch_size", "256",
            "--device", "cpu",  # Change to "cuda" if GPU available
            "--metric", "cosine",
        ]
        return self.run_command("GPU FAISS Indexing", cmd, timeout=1800)

    def generate_nim_questions(self, corpus_file: Optional[str] = None) -> bool:
        """Stage 4: Generate NIM question candidates."""
        logger.info("\n[STAGE 4] NIM question generation")
        
        corpus = corpus_file or str(self.work_dir / "corpus_merged.json")
        output_file = self.work_dir / "candidate_questions.json"

        cmd = [
            "python", "generate_nim_questions.py",
            "--corpus", corpus,
            "--output", str(output_file),
            "--num_questions", "1000",
            "--batch_size", "8",
        ]
        # Add --use_openrouter if wanting to use OpenRouter; requires API key
        return self.run_command("NIM Question Generation", cmd, timeout=3600)

    def sample_300_questions(self, candidates_file: Optional[str] = None) -> bool:
        """Stage 5: Sample 300 questions for annotation + IAA."""
        logger.info("\n[STAGE 5] Stratified question sampling + IAA prep")
        
        candidates = candidates_file or str(self.work_dir / "candidate_questions.json")
        output_file = self.work_dir / "annotation_set_300.json"
        csv_export = self.work_dir / "iaa_template.csv"

        cmd = [
            "python", "sample_300_questions.py",
            "--candidates", candidates,
            "--output", str(output_file),
            "--iaa_export", str(csv_export),
            "--num_samples", "300",
            "--stratify_by", "source",
        ]
        return self.run_command("Question Sampling + IAA Prep", cmd)

    def validate_outputs(self) -> bool:
        """Stage 6: Validate all outputs exist and have expected structure."""
        logger.info("\n[STAGE 6] Output validation")
        
        expected_files = [
            ("Merged Corpus", self.work_dir / "corpus_merged.json"),
            ("Ingestion Stats", self.work_dir / "ingestion_stats.json"),
            ("FAISS Index", self.work_dir / "indexes" / "index.faiss"),
            ("Index Metadata", self.work_dir / "indexes" / "metadata.json"),
            ("Candidate Questions", self.work_dir / "candidate_questions.json"),
            ("Annotation Set (300Q)", self.work_dir / "annotation_set_300.json"),
            ("IAA CSV Template", self.work_dir / "iaa_template.csv"),
        ]
        
        all_valid = True
        for name, path in expected_files:
            exists = path.exists()
            status = "✅" if exists else "❌"
            logger.info(f"{status} {name}: {path}")
            if not exists:
                all_valid = False

        return all_valid

    def generate_summary(self) -> dict:
        """Generate execution summary."""
        elapsed = time.time() - self.start_time
        stage_results = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "work_directory": str(self.work_dir),
            "stages": self.stages,
            "success": all(s.get("success", False) for s in self.stages.values()),
        }
        return stage_results

    def save_summary(self):
        """Save execution summary to JSON."""
        summary = self.generate_summary()
        summary_file = self.work_dir / "runbook_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\n✅ Runbook summary saved to {summary_file}")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Priority-1 Complete Build Runbook")
    parser.add_argument("--queries", nargs="+", default=[
        "retrieval augmented generation",
        "dense passage retrieval",
        "semantic search",
        "neural information retrieval",
    ], help="Ingestion queries")
    parser.add_argument("--max_results_per_query", type=int, default=3000, help="Papers per query")
    parser.add_argument("--corpus_json", default=None, help="Pre-built corpus JSON (skip ingestion)")
    parser.add_argument("--skip_ingestion", action="store_true", help="Skip ingestion stage")
    parser.add_argument("--skip_embedding", action="store_true", help="Skip FAISS indexing")
    parser.add_argument("--skip_questions", action="store_true", help="Skip NIM question generation")
    parser.add_argument("--work_dir", default="data/priority1_build", help="Working directory")
    parser.add_argument("--dry_run", action="store_true", help="Show commands without executing")
    return parser.parse_args()


def main():
    args = parse_args()
    rb = Runbook(work_dir=args.work_dir)

    logger.info("=" * 70)
    logger.info("🚀 Sanshodhak Priority-1 Complete Build Runbook")
    logger.info("=" * 70)
    logger.info(f"Work directory: {rb.work_dir}")
    logger.info(f"Queries: {args.queries}")

    if args.dry_run:
        logger.info("\n[DRY RUN MODE]")
        logger.info("No commands will be executed.")
        return

    # Stage 1: Ingestion
    if not args.skip_ingestion and not args.corpus_json:
        logger.info("\nStarting ingestion stage...")
        if not rb.collect_ingestion_sources(args.queries, args.max_results_per_query):
            logger.error("❌ Ingestion failed")
            return

    # Stage 2: Merge & Deduplicate
    corpus_file = args.corpus_json or str(rb.work_dir / "corpus_merged.json")
    if not args.corpus_json:
        if not rb.merge_and_deduplicate():
            logger.error("❌ Merge & dedup failed")
            return

    # Stage 3: GPU Embeddings
    if not args.skip_embedding:
        if not rb.build_gpu_embeddings(corpus_file):
            logger.warning("⚠️  GPU embedding failed (non-blocking)")

    # Stage 4: NIM Questions
    if not args.skip_questions:
        if not rb.generate_nim_questions(corpus_file):
            logger.warning("⚠️  NIM question generation failed (non-blocking)")

    # Stage 5: Stratified Sampling + IAA
    if not rb.sample_300_questions():
        logger.warning("⚠️  Question sampling failed (non-blocking)")

    # Stage 6: Validation
    if not rb.validate_outputs():
        logger.warning("⚠️  Some outputs missing")

    # Summary
    summary = rb.save_summary()
    logger.info("\n" + "=" * 70)
    logger.info("📊 RUNBOOK EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(json.dumps(summary, indent=2))
    logger.info("=" * 70)

    if summary["success"]:
        logger.info("✅ Priority-1 build completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Priority-1 build had failures")
        sys.exit(1)


if __name__ == "__main__":
    main()
