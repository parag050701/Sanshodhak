#!/usr/bin/env python3
"""
Stratified 300-question sampling + IAA preparation.

Takes candidate questions, applies stratified sampling by paper source/quality,
and exports a 300-question set with annotation templates for inter-annotator agreement.

Usage:
  python sample_300_questions.py --candidates data/candidate_questions.json \
    --output data/annotation_set_300.json \
    --iaa_export data/iaa_template.xlsx
"""

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample 300 questions for annotation + IAA")
    parser.add_argument("--candidates", required=True, help="Path to candidate questions JSON")
    parser.add_argument("--output", required=True, help="Output 300-question JSON with annotation template")
    parser.add_argument("--iaa_export", default=None, help="Optional CSV export for IAA annotation")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of questions to sample")
    parser.add_argument("--stratify_by", choices=["source", "quality", "paper_doi"], default="source", help="Stratification key")
    return parser.parse_args()


def load_questions(path: str) -> dict:
    """Load candidate questions JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(
    questions: List[dict],
    stratify_by: str,
    num_samples: int,
) -> List[dict]:
    """Stratified random sampling by given key."""
    strata = {}
    for q in questions:
        key = q.get(stratify_by, "unknown")
        if key not in strata:
            strata[key] = []
        strata[key].append(q)

    logger.info(f"Stratification by '{stratify_by}':")
    for key, items in strata.items():
        logger.info(f"  {key}: {len(items)} questions")

    # Allocate proportionally
    total = len(questions)
    sampled = []
    for key, items in strata.items():
        allocation = int((len(items) / total) * num_samples)
        if allocation > 0:
            sampled.extend(random.sample(items, min(allocation, len(items))))

    # Fill remaining slots randomly
    remaining = num_samples - len(sampled)
    if remaining > 0:
        pool = [q for q in questions if q not in sampled]
        sampled.extend(random.sample(pool, min(remaining, len(pool))))

    return sampled[:num_samples]


def annotate_question_template(q: dict, idx: int) -> dict:
    """Add annotation fields to a question."""
    return {
        **q,
        "id": f"Q{idx+1:03d}",
        "annotations": {
            "answerable_by_corpus": None,  # True/False (IAA target)
            "relevance": None,  # 1-5 Likert scale
            "clarity": None,  # 1-5 Likert scale
            "annotator_1": {
                "answerable": None,
                "confidence": None,
                "notes": "",
            },
            "annotator_2": {
                "answerable": None,
                "confidence": None,
                "notes": "",
            },
            "consensus": None,  # After dual annotation
            "kappa": None,  # Pre-calculated for reference
        },
    }


def export_iaa_csv(questions: List[dict], output_path: str):
    """Export questions in CSV format for easy annotation."""
    fieldnames = [
        "id",
        "question",
        "paper_title",
        "paper_doi",
        "source",
        "resolvability_score",
        "annotator_1_answerable",
        "annotator_1_confidence",
        "annotator_1_notes",
        "annotator_2_answerable",
        "annotator_2_confidence",
        "annotator_2_notes",
        "consensus",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in questions:
            row = {
                "id": q.get("id"),
                "question": q.get("question", ""),
                "paper_title": q.get("paper_title", ""),
                "paper_doi": q.get("paper_doi", ""),
                "source": q.get("source", ""),
                "resolvability_score": q.get("resolvability_score", ""),
                "annotator_1_answerable": "",
                "annotator_1_confidence": "",
                "annotator_1_notes": "",
                "annotator_2_answerable": "",
                "annotator_2_confidence": "",
                "annotator_2_notes": "",
                "consensus": "",
            }
            writer.writerow(row)
    logger.info(f"✅ Exported {len(questions)} questions to {output_path}")


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading candidate questions from {args.candidates}...")
    data = load_questions(args.candidates)
    candidates = data.get("questions", [])
    logger.info(f"Loaded {len(candidates)} candidate questions")

    logger.info(f"Performing stratified sampling ({args.stratify_by})...")
    sampled = stratified_sample(candidates, args.stratify_by, args.num_samples)
    logger.info(f"Sampled {len(sampled)} questions")

    # Add annotation template
    annotated = [annotate_question_template(q, i) for i, q in enumerate(sampled)]

    # Compute summary stats
    sources_count = Counter(q.get("source") for q in annotated)
    quality_count = Counter(q.get("quality") for q in annotated)

    output_data = {
        "annotation_set": args.output,
        "num_questions": len(annotated),
        "source_candidate_file": args.candidates,
        "stratification": args.stratify_by,
        "stats": {
            "sources": dict(sources_count),
            "qualities": dict(quality_count),
            "mean_resolvability": sum(q.get("resolvability_score", 0) for q in annotated) / len(annotated) if annotated else 0,
        },
        "questions": annotated,
        "generated_at": time.time(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Annotation set saved to {output_path}")

    # Optional CSV export
    if args.iaa_export:
        export_iaa_csv(annotated, args.iaa_export)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("𝐀𝐧𝐧𝐨𝐭𝐚𝐭𝐢𝐨𝐧 𝐒𝐞𝐭 𝐑𝐞𝐚𝐝𝐲")
    logger.info("="*60)
    logger.info(f"Total questions: {len(annotated)}")
    logger.info(f"Source distribution: {dict(sources_count)}")
    logger.info(f"Quality distribution: {dict(quality_count)}")
    logger.info(f"Mean resolvability: {output_data['stats']['mean_resolvability']:.2f}")
    logger.info(f"Output JSON: {output_path}")
    if args.iaa_export:
        logger.info(f"CSV template: {args.iaa_export}")


if __name__ == "__main__":
    main()
