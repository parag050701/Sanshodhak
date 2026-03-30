#!/usr/bin/env python3
"""
Module 7 — Full Ablation Report with LaTeX Table.

Reads eval_summary.json (4-system comparative evaluation already run on
all 20 questions) and produces:
  - Structured ablation JSON with delta/improvement columns
  - Markdown ablation table
  - LaTeX tabular environment ready for manuscript insertion

Component analysis:
  Baseline: VR-D (Dense-only)
  +BM25:    VR-H (Hybrid = Dense + BM25)
  +Graph:   GR   (Graph = Dense + Graph expansion)
  +Both:    HGR  (Hybrid+Graph = Dense + BM25 + Graph, full system)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
EVAL_SUMMARY = BASE_DIR / "eval_summary.json"
OUTPUT_DIR = BASE_DIR / "journal_claim_sync"

SYSTEMS = [
    ("VR-D (Dense)",      "VR-D",  "Dense only (FAISS)"),
    ("VR-H (Hybrid)",     "VR-H",  "+BM25 (Hybrid)"),
    ("GR (Graph)",        "GR",    "+Graph expansion (no BM25)"),
    ("HGR (Hybrid+Graph)","HGR",   "+BM25 +Graph (full system)"),
]

RETRIEVAL_METRICS = ["P@1", "R@5", "NDCG@5", "MRR", "MAP"]
GEN_METRICS = ["ROUGE-1", "ROUGE-L", "BERTScore-F1"]


def fmt_delta(d: float) -> str:
    """Format a delta with sign."""
    if abs(d) < 0.0005:
        return "—"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_SUMMARY) as f:
        raw = json.load(f)

    # Extract data for each system
    rows = []
    baseline = None
    for sys_key, abbrev, description in SYSTEMS:
        sys_data = raw[sys_key]
        retr = sys_data["retrieval"]
        gen = sys_data.get("generation", {})
        lat = sys_data["latency"]

        row = {
            "system": sys_key,
            "abbrev": abbrev,
            "description": description,
            "P@1": retr.get("P@1", 0.0),
            "R@5": retr.get("R@5", 0.0),
            "NDCG@5": retr.get("NDCG@5", 0.0),
            "MRR": retr.get("MRR", 0.0),
            "MAP": retr.get("MAP", 0.0),
            "ROUGE-1": gen.get("ROUGE-1", 0.0),
            "ROUGE-L": gen.get("ROUGE-L", 0.0),
            "BERTScore-F1": gen.get("BERTScore-F1", 0.0),
            "latency_mean_s": lat.get("mean_s", 0.0),
            "latency_p95_s": lat.get("p95_s", 0.0),
        }

        if baseline is None:
            baseline = row
            row["delta_P@1"] = 0.0
            row["delta_NDCG@5"] = 0.0
            row["delta_MRR"] = 0.0
        else:
            row["delta_P@1"] = round(row["P@1"] - baseline["P@1"], 4)
            row["delta_NDCG@5"] = round(row["NDCG@5"] - baseline["NDCG@5"], 4)
            row["delta_MRR"] = round(row["MRR"] - baseline["MRR"], 4)

        rows.append(row)

    hgr = rows[-1]
    baseln = rows[0]

    total_improvement = {
        "P@1": round(hgr["P@1"] - baseln["P@1"], 4),
        "NDCG@5": round(hgr["NDCG@5"] - baseln["NDCG@5"], 4),
        "MRR": round(hgr["MRR"] - baseln["MRR"], 4),
        "R@5": round(hgr["R@5"] - baseln["R@5"], 4),
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_questions": raw.get("_meta", {}).get("n_questions", 20),
        "baseline_system": baseln["system"],
        "full_system": hgr["system"],
        "total_improvement_over_baseline": total_improvement,
        "rows": rows,
    }

    # ── JSON ─────────────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "ablation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_lines = [
        "# Module 7 — Ablation Study Report",
        "",
        f"**Generated:** {report['generated_utc']}",
        f"**Questions:** {report['n_questions']} (domain-specific RAG evaluation set)",
        "",
        "## Component Contribution Analysis",
        "",
        "Four system configurations evaluated by incrementally adding components:",
        "",
        "| System | Description | P@1 | R@5 | NDCG@5 | MRR | ΔP@1 | ΔNDCG@5 | ΔMRR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dp = fmt_delta(r.get("delta_P@1", 0.0))
        dn = fmt_delta(r.get("delta_NDCG@5", 0.0))
        dm = fmt_delta(r.get("delta_MRR", 0.0))
        md_lines.append(
            f"| **{r['abbrev']}** | {r['description']} "
            f"| {r['P@1']:.3f} | {r['R@5']:.3f} | {r['NDCG@5']:.3f} | {r['MRR']:.3f} "
            f"| {dp} | {dn} | {dm} |"
        )

    md_lines += [
        "",
        "## Generation Quality",
        "",
        "| System | ROUGE-1 | ROUGE-L | BERTScore-F1 | Latency Mean (s) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['abbrev']}** | {r['ROUGE-1']:.3f} | {r['ROUGE-L']:.3f} "
            f"| {r['BERTScore-F1']:.3f} | {r['latency_mean_s']:.2f} |"
        )

    md_lines += [
        "",
        "## Total HGR Improvement over Dense Baseline",
        "",
        f"| Metric | Baseline (VR-D) | HGR | Δ |",
        f"|---|---|---|---|",
        f"| P@1 | {baseln['P@1']:.3f} | {hgr['P@1']:.3f} | **{fmt_delta(total_improvement['P@1'])}** |",
        f"| R@5 | {baseln['R@5']:.3f} | {hgr['R@5']:.3f} | **{fmt_delta(total_improvement['R@5'])}** |",
        f"| NDCG@5 | {baseln['NDCG@5']:.3f} | {hgr['NDCG@5']:.3f} | **{fmt_delta(total_improvement['NDCG@5'])}** |",
        f"| MRR | {baseln['MRR']:.3f} | {hgr['MRR']:.3f} | **{fmt_delta(total_improvement['MRR'])}** |",
        "",
    ]

    # ── LaTeX table ──────────────────────────────────────────────────────────
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: contribution of each retrieval component.",
        r"$\Delta$ columns show improvement over the Dense-only baseline (VR-D).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"System & Description & P@1 & R@5 & NDCG@5 & MRR & $\Delta$P@1 & $\Delta$NDCG@5 & $\Delta$MRR \\",
        r"\midrule",
    ]
    for r in rows:
        sys_tex = r["abbrev"].replace("+", r"\texttt{+}")
        dp = fmt_delta(r.get("delta_P@1", 0.0))
        dn = fmt_delta(r.get("delta_NDCG@5", 0.0))
        dm = fmt_delta(r.get("delta_MRR", 0.0))
        # Bold the full system row
        if r["abbrev"] == "HGR":
            latex_lines.append(
                rf"\textbf{{{r['abbrev']}}} & {r['description']} "
                rf"& \textbf{{{r['P@1']:.3f}}} & \textbf{{{r['R@5']:.3f}}} "
                rf"& \textbf{{{r['NDCG@5']:.3f}}} & \textbf{{{r['MRR']:.3f}}} "
                rf"& \textbf{{{dp}}} & \textbf{{{dn}}} & \textbf{{{dm}}} \\"
            )
        else:
            latex_lines.append(
                rf"{r['abbrev']} & {r['description']} "
                rf"& {r['P@1']:.3f} & {r['R@5']:.3f} "
                rf"& {r['NDCG@5']:.3f} & {r['MRR']:.3f} "
                rf"& {dp} & {dn} & {dm} \\"
            )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex_table = "\n".join(latex_lines)

    md_lines += [
        "## LaTeX Table (for manuscript)",
        "",
        "```latex",
        latex_table,
        "```",
        "",
    ]

    with open(OUTPUT_DIR / "ablation_report.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Save LaTeX separately too
    with open(OUTPUT_DIR / "ablation_table.tex", "w") as f:
        f.write(latex_table + "\n")

    print("[OK] Ablation report written:")
    print(f"  - {OUTPUT_DIR}/ablation_report.json")
    print(f"  - {OUTPUT_DIR}/ablation_report.md")
    print(f"  - {OUTPUT_DIR}/ablation_table.tex")
    print()
    print("Component contributions (Δ vs Dense baseline):")
    for r in rows[1:]:
        print(f"  {r['abbrev']:5s}: ΔP@1={fmt_delta(r['delta_P@1'])}  ΔNDCG@5={fmt_delta(r['delta_NDCG@5'])}  ΔMRR={fmt_delta(r['delta_MRR'])}")
    print()
    print(f"HGR total improvement: ΔP@1={fmt_delta(total_improvement['P@1'])}  ΔNDCG@5={fmt_delta(total_improvement['NDCG@5'])}  ΔMRR={fmt_delta(total_improvement['MRR'])}")


if __name__ == "__main__":
    main()
