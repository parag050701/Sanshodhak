#!/usr/bin/env python3
"""
Inter-Annotator Agreement (IAA) Framework for Sanshodhak RAG Evaluation
========================================================================
This script manages the annotation process for evaluating retrieval relevance
judgments from two independent annotators and computes Cohen's kappa.

Workflow:
1. Annotator 1 runs this script and records binary relevance for each question.
2. Annotator 2 runs this independently on the same question set.
3. Run compute_kappa() to compare both annotation files and report agreement.

Target: Cohen's kappa >= 0.70 (substantial agreement threshold for IR evaluation)
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

RAG_QS_FILE = "rag_test_questions_100.json"
ANNOTATOR_1_FILE = "annotations_annotator1.json"
ANNOTATOR_2_FILE = "annotations_annotator2.json"
RESULTS_FILE = "iaa_results.json"


def load_questions() -> List[Dict]:
    """Load the 100-question evaluation set."""
    with open(RAG_QS_FILE, "r") as f:
        return json.load(f)


def run_annotation_session(annotator_id: str, output_file: str):
    """
    Interactive annotation session.
    Annotator labels each (question, document) pair as relevant (1) or not (0).
    """
    questions = load_questions()
    annotations = {}

    print(f"\n{'='*70}")
    print(f"SANSHODHAK IAA ANNOTATION SESSION — Annotator: {annotator_id}")
    print(f"{'='*70}")
    print("Instructions:")
    print("  For each question, you will be shown the question and the paper ID.")
    print("  Mark RELEVANT (1) if the listed paper is the primary source for the answer.")
    print("  Mark NOT RELEVANT (0) if the paper is not the primary source.")
    print("  Press 'q' at any time to save and quit (you can resume later).")
    print(f"{'='*70}\n")

    # Load existing progress if resuming
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing = json.load(f)
        annotations = existing.get("annotations", {})
        print(f"Resuming from previous session: {len(annotations)}/100 done.\n")

    for q in questions:
        qid = str(q["id"])
        if qid in annotations:
            continue  # Skip already annotated

        print(f"\n[Q{q['id']}/100] [{q['type'].upper()}/{q['difficulty'].upper()}]")
        print(f"Stratum: {q.get('stratum', 'RAG/IR')}")
        print(f"Question: {q['question']}")
        print(f"Expected paper: {q['paper_id']}")
        print(f"Relevant docs listed: {', '.join(q['relevant_docs'])}")

        while True:
            val = input("Is this paper the PRIMARY relevant source? [1=Yes / 0=No / q=Quit]: ").strip().lower()
            if val == "q":
                # Save and exit
                _save_annotations(annotator_id, annotations, output_file)
                print(f"\nProgress saved to {output_file}. Resume any time by running this script again.")
                sys.exit(0)
            elif val in ("0", "1"):
                annotations[qid] = int(val)
                break
            else:
                print("Please enter 1, 0, or q.")

    # Final save
    _save_annotations(annotator_id, annotations, output_file)
    print(f"\n✅ Annotation complete! Results saved to {output_file}")
    print(f"Total annotated: {len(annotations)}/100")


def _save_annotations(annotator_id: str, annotations: Dict, output_file: str):
    payload = {
        "annotator": annotator_id,
        "question_set": RAG_QS_FILE,
        "total_questions": 100,
        "completed": len(annotations),
        "annotations": annotations
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)


def compute_cohen_kappa(y1: List[int], y2: List[int]) -> Tuple[float, Dict]:
    """
    Compute Cohen's kappa for binary judgments.
    kappa = (P_obs - P_exp) / (1 - P_exp)
    """
    assert len(y1) == len(y2), "Annotation vectors must be the same length."
    n = len(y1)

    # Observed agreement
    p_obs = sum(a == b for a, b in zip(y1, y2)) / n

    # Expected agreement
    p1_yes = sum(y1) / n
    p1_no = 1 - p1_yes
    p2_yes = sum(y2) / n
    p2_no = 1 - p2_yes
    p_exp = (p1_yes * p2_yes) + (p1_no * p2_no)

    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 1.0

    # Confusion matrix
    a11 = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)  # Both relevant
    a10 = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 0)  # A1 only
    a01 = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 1)  # A2 only
    a00 = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)  # Both not relevant

    interpretation = (
        "Almost perfect (κ ≥ 0.80)" if kappa >= 0.80 else
        "✅ Substantial (κ ≥ 0.70) — MEETS TARGET" if kappa >= 0.70 else
        "⚠️ Moderate (0.60 ≤ κ < 0.70) — below target" if kappa >= 0.60 else
        "❌ Fair/Poor (κ < 0.60) — needs reconciliation"
    )

    return kappa, {
        "n_questions": n,
        "p_observed": round(p_obs, 4),
        "p_expected": round(p_exp, 4),
        "cohen_kappa": round(kappa, 4),
        "interpretation": interpretation,
        "confusion_matrix": {
            "both_relevant": a11,
            "annotator1_only": a10,
            "annotator2_only": a01,
            "both_not_relevant": a00
        }
    }


def evaluate_iaa():
    """Load both annotation files and compute agreement statistics."""
    if not os.path.exists(ANNOTATOR_1_FILE):
        print(f"ERROR: {ANNOTATOR_1_FILE} not found. Run this script with --annotate 1 first.")
        return
    if not os.path.exists(ANNOTATOR_2_FILE):
        print(f"ERROR: {ANNOTATOR_2_FILE} not found. Run this script with --annotate 2 first.")
        return

    with open(ANNOTATOR_1_FILE, "r") as f:
        a1_data = json.load(f)
    with open(ANNOTATOR_2_FILE, "r") as f:
        a2_data = json.load(f)

    a1_ann = a1_data["annotations"]
    a2_ann = a2_data["annotations"]

    # Find shared questions
    shared_ids = sorted(set(a1_ann.keys()) & set(a2_ann.keys()), key=int)
    print(f"\nShared annotated questions: {len(shared_ids)}/100")

    y1 = [a1_ann[qid] for qid in shared_ids]
    y2 = [a2_ann[qid] for qid in shared_ids]

    kappa, stats = compute_cohen_kappa(y1, y2)

    print(f"\n{'='*60}")
    print("INTER-ANNOTATOR AGREEMENT RESULTS")
    print(f"{'='*60}")
    print(f"  Questions compared:  {stats['n_questions']}")
    print(f"  Observed agreement:  {stats['p_observed']*100:.1f}%")
    print(f"  Expected agreement:  {stats['p_expected']*100:.1f}%")
    print(f"  Cohen's kappa (κ):   {stats['cohen_kappa']:.4f}")
    print(f"  Interpretation:      {stats['interpretation']}")
    print(f"\n  Confusion matrix:")
    cm = stats["confusion_matrix"]
    print(f"    Both relevant:      {cm['both_relevant']}")
    print(f"    Annotator 1 only:   {cm['annotator1_only']}")
    print(f"    Annotator 2 only:   {cm['annotator2_only']}")
    print(f"    Both not relevant:  {cm['both_not_relevant']}")
    print(f"{'='*60}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Full results saved to {RESULTS_FILE}")

    return stats


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--annotate":
        annotator_num = sys.argv[2] if len(sys.argv) > 2 else "1"
        out_file = ANNOTATOR_1_FILE if annotator_num == "1" else ANNOTATOR_2_FILE
        run_annotation_session(f"Annotator {annotator_num}", out_file)
    elif len(sys.argv) > 1 and sys.argv[1] == "--compute":
        evaluate_iaa()
    else:
        print("Usage:")
        print("  python iaa_annotation_framework.py --annotate 1   # Annotator 1 session")
        print("  python iaa_annotation_framework.py --annotate 2   # Annotator 2 session")
        print("  python iaa_annotation_framework.py --compute       # Compute Cohen's kappa")
