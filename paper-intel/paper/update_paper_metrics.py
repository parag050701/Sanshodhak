#!/usr/bin/env python3
"""
Update generation metric numbers in main.tex from eval_summary.json.
Run this after: python eval_compare.py (with generation enabled)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_JSON = ROOT.parent / "eval_summary.json"
TEX_FILE  = ROOT / "main.tex"

def load_metrics():
    with open(EVAL_JSON) as f:
        data = json.load(f)
    rows = {}
    for sys_name, sys_data in data.items():
        if sys_name.startswith("_"):
            continue
        gen = sys_data.get("generation", {})
        if gen:
            rows[sys_name] = {
                "rouge1":  gen.get("rouge1", 0.0),
                "rouge2":  gen.get("rouge2", 0.0),
                "rougeL":  gen.get("rougeL", 0.0),
                "bleu":    gen.get("bleu", 0.0),
                "bscore_p": gen.get("bertscore_precision", 0.0),
                "bscore_r": gen.get("bertscore_recall", 0.0),
                "bscore_f": gen.get("bertscore_f1", 0.0),
            }
    return rows


def build_table(rows):
    """Build the new LaTeX table rows for generation metrics."""
    sys_order = ["VR-D (Dense)", "VR-H (Hybrid)", "GR (Graph)", "HGR (Hybrid+Graph)"]
    col_labels = ["VR-D", "VR-H", "GR", "HGR"]

    metric_keys = ["rouge1", "rouge2", "rougeL", "bleu",
                   "bscore_p", "bscore_r", "bscore_f"]
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU",
                     "BERTScore P", "BERTScore R", "BERTScore F1"]

    lines = []
    for mkey, mlabel in zip(metric_keys, metric_labels):
        vals = [rows.get(s, {}).get(mkey, 0.0) for s in sys_order]
        best_idx = vals.index(max(vals))
        cells = []
        for i, v in enumerate(vals):
            cell = f"{v:.3f}"
            if i == best_idx:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{mlabel:<16} & {' & '.join(cells)} \\\\")
    return "\n".join(lines)


def update_tex(new_rows):
    tex = TEX_FILE.read_text()

    # Replace the tabular content between \midrule and \bottomrule
    pattern = (
        r"(ROUGE-1\s*&.*?BERTScore F1\s*&.*?\\\\)"
    )
    new_table_body = build_table(new_rows)
    updated = re.sub(pattern, new_table_body, tex, flags=re.DOTALL)

    if updated == tex:
        print("⚠  Pattern not found — table body unchanged.")
        print("   New table body would be:\n", new_table_body)
        return

    TEX_FILE.write_text(updated)
    print(f"✅ Updated {TEX_FILE.name} with real generation metrics.")


def main():
    if not EVAL_JSON.exists():
        print(f"❌ {EVAL_JSON} not found. Run eval_compare.py first.")
        sys.exit(1)

    rows = load_metrics()
    if not rows:
        print("⚠  No generation metrics found in eval_summary.json")
        print("   Did you run eval_compare.py without --no-generation?")
        sys.exit(1)

    print("Generation metrics found for:", list(rows.keys()))
    for sys_name, m in rows.items():
        print(f"  {sys_name}: ROUGE-1={m['rouge1']:.3f}, BERTScore-F1={m['bscore_f']:.3f}")

    update_tex(rows)


if __name__ == "__main__":
    main()
