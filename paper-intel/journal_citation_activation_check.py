#!/usr/bin/env python3
"""
Module 3: Citation-edge activation check
Verifies citation-derived graph edges are active in GraphRAG.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from graph_rag import GraphRAG

ROOT = Path(__file__).resolve().parents[1]
PAPER_INTEL = ROOT / "paper-intel"
INDEX_DIR = PAPER_INTEL / "rag_index"
OUT_DIR = PAPER_INTEL / "journal_claim_sync"
OUT_JSON = OUT_DIR / "citation_activation_report.json"
OUT_MD = OUT_DIR / "citation_activation_report.md"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rag = GraphRAG(use_bm25=False)
    rag.load(str(INDEX_DIR))
    stats = rag.get_graph_stats()

    citation_edges = int(stats.get("citation_edges", 0))
    citation_ratio = float(stats.get("citation_edge_ratio", 0.0))
    status = "pass" if citation_edges > 0 else "fail"

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "index_dir": str(INDEX_DIR.relative_to(ROOT)),
        "graph_stats": stats,
        "check": {
            "criterion": "citation_edges > 0",
            "citation_edges": citation_edges,
            "citation_edge_ratio": citation_ratio,
            "passed": citation_edges > 0,
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Citation Edge Activation Report",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Status: **{status.upper()}**",
        f"- Index: `{report['index_dir']}`",
        "",
        "## Check",
        "- Criterion: `citation_edges > 0`",
        f"- citation_edges: `{citation_edges}`",
        f"- citation_edge_ratio: `{citation_ratio:.6f}`",
        "",
        "## Graph Stats",
    ]

    for key in [
        "nodes",
        "edges",
        "citation_edges",
        "citation_edge_ratio",
        "avg_degree",
        "max_degree",
        "density",
        "connected_components",
    ]:
        if key in stats:
            lines.append(f"- {key}: `{stats[key]}`")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[OK] Citation activation report written:")
    print(f"  - {OUT_JSON}")
    print(f"  - {OUT_MD}")
    print(f"[OK] Status: {status.upper()} (citation_edges={citation_edges})")


if __name__ == "__main__":
    main()
