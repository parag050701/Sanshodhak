#!/usr/bin/env python3
"""
Module 1: Baseline Freeze
Creates a reproducibility snapshot of the current Sanshodhak state.
"""

from __future__ import annotations

import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_INTEL = ROOT / "paper-intel"
OUT_DIR = PAPER_INTEL / "journal_baseline"
OUT_JSON = OUT_DIR / "baseline_manifest.json"
OUT_MD = OUT_DIR / "baseline_manifest.md"

REQUIRED_FILES = [
    ROOT / "sanshodhak_fixed.tex",
    ROOT / "Sanshodhak (2).pdf",
    ROOT / "JOURNAL_CLAIM_AUDIT.md",
    PAPER_INTEL / "eval_summary.json",
    PAPER_INTEL / "eval_report.txt",
    PAPER_INTEL / "PAPER_SCORE_SHEET.md",
    PAPER_INTEL / "eval_compare.py",
    PAPER_INTEL / "graph_rag.py",
    PAPER_INTEL / "core" / "adaptive_budget.py",
    PAPER_INTEL / "ingestion" / "discovery" / "search_engine.py",
    PAPER_INTEL / "rag_index" / "faiss.index",
    PAPER_INTEL / "rag_index" / "metadata.pkl",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_eval_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_metrics(summary: dict) -> dict:
    hgr = summary.get("HGR (Hybrid+Graph)", {})
    retrieval = hgr.get("retrieval", {})
    generation = hgr.get("generation", {})
    latency = hgr.get("latency", {})
    graph_stats = hgr.get("graph_stats", {})

    return {
        "retrieval": {
            "P@1": retrieval.get("P@1"),
            "R@5": retrieval.get("R@5"),
            "R@10": retrieval.get("R@10"),
            "NDCG@5": retrieval.get("NDCG@5"),
            "NDCG@10": retrieval.get("NDCG@10"),
            "MRR": retrieval.get("MRR"),
            "MAP": retrieval.get("MAP"),
        },
        "generation": {
            "rouge1": generation.get("rouge1"),
            "rouge2": generation.get("rouge2"),
            "rougeL": generation.get("rougeL"),
            "bleu": generation.get("bleu"),
            "bertscore_f1": generation.get("bertscore_f1"),
        },
        "latency_seconds": {
            "mean_s": latency.get("mean_s"),
            "median_s": latency.get("median_s"),
            "p95_s": latency.get("p95_s"),
        },
        "graph_stats": graph_stats,
    }


def build_manifest() -> dict:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]

    summary = load_eval_summary(PAPER_INTEL / "eval_summary.json")
    metrics = collect_metrics(summary)

    file_hashes = {}
    for p in REQUIRED_FILES:
        if p.exists():
            rel = str(p.relative_to(ROOT))
            file_hashes[rel] = {
                "size_bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            }

    manifest = {
        "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "status": {
            "required_files_present": len(missing) == 0,
            "missing_files": missing,
        },
        "paper_baseline": {
            "tex": "sanshodhak_fixed.tex",
            "pdf": "Sanshodhak (2).pdf",
            "audit": "JOURNAL_CLAIM_AUDIT.md",
        },
        "evaluation_baseline": {
            "summary_file": "paper-intel/eval_summary.json",
            "report_file": "paper-intel/eval_report.txt",
            "score_sheet": "paper-intel/PAPER_SCORE_SHEET.md",
            "hgr_metrics": metrics,
        },
        "runtime_baseline": {
            "index": "paper-intel/rag_index",
            "evaluation_script": "paper-intel/eval_compare.py",
            "graph_runtime": "paper-intel/graph_rag.py",
            "adaptive_budget": "paper-intel/core/adaptive_budget.py",
            "ingestion_search": "paper-intel/ingestion/discovery/search_engine.py",
        },
        "artifact_hashes": file_hashes,
    }

    return manifest


def write_markdown(manifest: dict) -> None:
    hgr = manifest["evaluation_baseline"]["hgr_metrics"]
    status = manifest["status"]

    lines = [
        "# Baseline Freeze Manifest",
        "",
        f"- Snapshot UTC: {manifest['snapshot_time_utc']}",
        f"- Required files present: {status['required_files_present']}",
        "",
        "## HGR Baseline Metrics",
        "",
        "### Retrieval",
    ]

    for k, v in hgr["retrieval"].items():
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "### Generation",
    ])

    for k, v in hgr["generation"].items():
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "### Latency (seconds)",
    ])

    for k, v in hgr["latency_seconds"].items():
        lines.append(f"- {k}: {v}")

    lines.extend([
        "",
        "### Graph Stats",
    ])

    for k, v in hgr["graph_stats"].items():
        lines.append(f"- {k}: {v}")

    if status["missing_files"]:
        lines.extend([
            "",
            "## Missing Files",
        ])
        for m in status["missing_files"]:
            lines.append(f"- {m}")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(manifest)

    print("[OK] Baseline manifest written:")
    print(f"  - {OUT_JSON}")
    print(f"  - {OUT_MD}")
    print(f"[OK] Required files present: {manifest['status']['required_files_present']}")
    if manifest["status"]["missing_files"]:
        print("[WARN] Missing files:")
        for m in manifest["status"]["missing_files"]:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
