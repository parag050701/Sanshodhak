#!/usr/bin/env python3
"""
Module 2: Claim-Evidence Sync
Checks manuscript claim-safety against verified artifact baselines.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_INTEL = ROOT / "paper-intel"
TEX = ROOT / "sanshodhak_fixed.tex"
EVAL = PAPER_INTEL / "eval_summary.json"
OUT_DIR = PAPER_INTEL / "journal_claim_sync"
OUT_JSON = OUT_DIR / "claim_sync_report.json"
OUT_MD = OUT_DIR / "claim_matrix.md"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tex = load_text(TEX)
    summary = load_json(EVAL)
    hgr = summary.get("HGR (Hybrid+Graph)", {})

    checks = []

    # 1) Novelty phrase safety
    checks.append({
        "claim": "Avoid absolute novelty phrase 'first published'",
        "status": "pass" if not contains(tex, r"first\s+published") else "fail",
        "evidence": "Manuscript phrase scan",
    })

    # 2) Dual-edge wording safety
    dual_edge_hits = contains(tex, r"dual-edge")
    limitation_present = contains(tex, r"citation.*zero|citation.*0|citation.*inactive|citation.*yielded\s+zero")
    checks.append({
        "claim": "If dual-edge is mentioned, citation-edge limitation must be explicit",
        "status": "pass" if (not dual_edge_hits or limitation_present) else "fail",
        "evidence": "Manuscript graph claim scan",
    })

    # 3) Latency wording safety
    has_152ms = contains(tex, r"152\\,ms|152\s*ms")
    checks.append({
        "claim": "No unsupported 152ms headline latency in manuscript",
        "status": "pass" if not has_152ms else "fail",
        "evidence": "Manuscript latency phrase scan",
    })

    # 4) Adaptive budget consistency
    adaptive_mentioned = contains(tex, r"adaptive\s+budget|adaptive\s+graph\s+budget|use_adaptive_budget")
    checks.append({
        "claim": "Runtime graph-slot strategy mentions adaptive budget",
        "status": "pass" if adaptive_mentioned else "warn",
        "evidence": "Manuscript method scan",
    })

    # 5) Core retrieval metrics consistency
    retrieval = hgr.get("retrieval", {})
    expected_strings = {
        "P@1": f"P@1\\,=\\,{retrieval.get('P@1')}",
        "NDCG@5": f"NDCG@5\\,=\\,{round(retrieval.get('NDCG@5', 0), 3)}",
        "NDCG@10": f"NDCG@10\\,=\\,{round(retrieval.get('NDCG@10', 0), 3)}",
        "MRR": f"MRR\\,=\\,{round(retrieval.get('MRR', 0), 3)}",
        "MAP": f"MAP\\,=\\,{round(retrieval.get('MAP', 0), 3)}",
    }

    metric_hits = 0
    for s in expected_strings.values():
        if s in tex:
            metric_hits += 1

    checks.append({
        "claim": "Core HGR metrics align with eval_summary baseline",
        "status": "pass" if metric_hits >= 4 else "warn",
        "evidence": f"{metric_hits}/5 expected metric strings found",
    })

    overall = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall = "warn"

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "manuscript": str(TEX.relative_to(ROOT)),
        "eval_baseline": str(EVAL.relative_to(ROOT)),
        "checks": checks,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Claim-Evidence Sync Matrix",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Overall status: **{overall.upper()}**",
        f"- Manuscript: `{report['manuscript']}`",
        f"- Baseline: `{report['eval_baseline']}`",
        "",
        "| Claim Check | Status | Evidence |",
        "|---|---|---|",
    ]

    for c in checks:
        lines.append(f"| {c['claim']} | {c['status']} | {c['evidence']} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[OK] Claim sync report written:")
    print(f"  - {OUT_JSON}")
    print(f"  - {OUT_MD}")
    print(f"[OK] Overall status: {overall.upper()}")


if __name__ == "__main__":
    main()
