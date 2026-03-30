#!/usr/bin/env python3
"""Consolidate paper-ready metrics into a single score sheet (JSON + CSV)."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    eval_summary = load_json(ROOT / "eval_summary.json") or {}
    ablation = load_json(ROOT / "journal_claim_sync" / "ablation_report.json") or {}
    claim_sync = load_json(ROOT / "journal_claim_sync" / "claim_sync_report.json") or {}
    citation = load_json(ROOT / "journal_claim_sync" / "citation_activation_report.json") or {}
    latency_split = load_json(ROOT / "journal_claim_sync" / "latency_split_report.json") or {}
    iaa = load_json(ROOT / "journal_claim_sync" / "iaa_report.json") or {}
    beir = load_json(ROOT / "beir_results.json") or {}
    stat = load_json(ROOT / "stat_test_results.json") or {}
    scale_no_graph = load_json(ROOT / "scalability_results" / "scalability_results.json") or {}
    scale_graph_small = load_json(ROOT / "scalability_results_graph_small" / "scalability_results.json") or {}

    systems = ["VR-D (Dense)", "VR-H (Hybrid)", "GR (Graph)", "HGR (Hybrid+Graph)"]

    by_system = []
    for s in systems:
        row_eval = eval_summary.get(s, {})
        retrieval = row_eval.get("retrieval", {})
        generation = row_eval.get("generation", {})
        latency = row_eval.get("latency", {})

        by_system.append(
            {
                "system": s,
                "P@1": retrieval.get("P@1"),
                "R@5": retrieval.get("R@5"),
                "NDCG@5": retrieval.get("NDCG@5"),
                "MRR": retrieval.get("MRR"),
                "MAP": retrieval.get("MAP"),
                "ROUGE-1": generation.get("rouge1"),
                "ROUGE-2": generation.get("rouge2"),
                "ROUGE-L": generation.get("rougeL"),
                "BLEU": generation.get("bleu"),
                "BERTScore-F1": generation.get("bertscore_f1"),
                "Latency-mean-s": latency.get("mean_s"),
                "Latency-p95-s": latency.get("p95_s"),
            }
        )

    hgr = eval_summary.get("HGR (Hybrid+Graph)", {})
    hgr_ret = hgr.get("retrieval", {})
    hgr_gen = hgr.get("generation", {})
    hgr_lat = hgr.get("latency", {})

    score_sheet = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "eval_summary": "eval_summary.json",
            "ablation": "journal_claim_sync/ablation_report.json",
            "claim_sync": "journal_claim_sync/claim_sync_report.json",
            "citation_activation": "journal_claim_sync/citation_activation_report.json",
            "latency_split": "journal_claim_sync/latency_split_report.json",
            "iaa": "journal_claim_sync/iaa_report.json",
            "beir": "beir_results.json",
            "stat_test": "stat_test_results.json",
            "scalability_no_graph": "scalability_results/scalability_results.json",
            "scalability_graph_small": "scalability_results_graph_small/scalability_results.json",
        },
        "headline_hgr": {
            "P@1": hgr_ret.get("P@1"),
            "R@5": hgr_ret.get("R@5"),
            "NDCG@5": hgr_ret.get("NDCG@5"),
            "MRR": hgr_ret.get("MRR"),
            "MAP": hgr_ret.get("MAP"),
            "ROUGE-1": hgr_gen.get("rouge1"),
            "ROUGE-L": hgr_gen.get("rougeL"),
            "BERTScore-F1": hgr_gen.get("bertscore_f1"),
            "Latency-mean-s": hgr_lat.get("mean_s"),
            "Latency-p95-s": hgr_lat.get("p95_s"),
        },
        "ablation_delta_vs_dense": (ablation.get("total_improvement_over_baseline") or {}),
        "quality_gates": {
            "claim_sync_status": claim_sync.get("overall_status") or claim_sync.get("status"),
            "citation_activation_pass": (citation.get("check") or {}).get("passed"),
            "citation_edges": (citation.get("graph_stats") or {}).get("citation_edges"),
            "iaa_kappa": iaa.get("cohens_kappa"),
            "iaa_status": iaa.get("kappa_status"),
            "iaa_observed_agreement": iaa.get("observed_agreement"),
        },
        "beir_external": beir,
        "stat_significance": stat,
        "latency_split_hgr": ((latency_split.get("systems") or {}).get("HGR (Hybrid+Graph)") or {}),
        "scalability": {
            "no_graph": scale_no_graph,
            "graph_small": scale_graph_small,
        },
        "by_system": by_system,
        "notes": [
            "Generation metrics are sourced from eval_summary.json to avoid zeroed ablation generation placeholders.",
            "Latency split uses separate retrieval-only vs end-to-end invocations; generation estimate is approximate.",
        ],
    }

    out_json = ROOT / "paper_ready_score_sheet.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(score_sheet, f, indent=2)

    out_csv = ROOT / "paper_ready_score_sheet_systems.csv"
    fieldnames = list(by_system[0].keys()) if by_system else ["system"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_system)

    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
