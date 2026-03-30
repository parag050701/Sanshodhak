#!/usr/bin/env python3
"""
Merge multi-topic ingestion outputs and deduplicate by DOI + title fingerprint.
Also emits source-level ingestion stats for paper table construction.

Example:
  python merge_corpus.py \
    --inputs data/corpus/rag.json data/corpus/kg.json data/corpus/biomed.json data/corpus/summarization.json \
    --output data/corpus/corpus_10k_dedup.json \
    --stats_output data/corpus/ingestion_stats.json
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List

from ingestion.discovery.doi_utils import normalize_doi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and deduplicate corpus JSON files")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input ingestion JSONs")
    parser.add_argument("--output", required=True, help="Output deduplicated JSON")
    parser.add_argument("--stats_output", required=True, help="Output stats JSON")
    return parser.parse_args()


def title_fingerprint(title: str) -> str:
    title = (title or "").lower()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in title)
    tokens = [t for t in cleaned.split() if len(t) > 2]
    key = " ".join(tokens[:20])
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def dedup_key(paper: Dict) -> str:
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return f"doi:{doi}"
    return f"title:{title_fingerprint(paper.get('title', ''))}"


def aggregate_stats(run: Dict, run_name: str) -> Dict:
    papers = run.get("papers", [])
    source_counts = run.get("sources_used", {})
    oa_count = sum(1 for p in papers if p.get("is_open_access"))
    return {
        "run_name": run_name,
        "query": run.get("query"),
        "raw_total_found": run.get("total_found", 0),
        "post_rank_selected": len(papers),
        "oa_fraction": (oa_count / len(papers)) if papers else 0.0,
        "elapsed_time_seconds": run.get("elapsed_time_seconds", 0.0),
        "source_counts": source_counts,
    }


def main() -> None:
    args = parse_args()

    merged: List[Dict] = []
    run_stats: List[Dict] = []

    for path_str in args.inputs:
        path = Path(path_str)
        with open(path, "r", encoding="utf-8") as f:
            run = json.load(f)
        merged.extend(run.get("papers", []))
        run_stats.append(aggregate_stats(run, path.name))

    unique_map: Dict[str, Dict] = {}
    duplicate_count = 0

    for paper in merged:
        key = dedup_key(paper)
        if key in unique_map:
            duplicate_count += 1
            prev = unique_map[key]
            prev_cit = prev.get("citations", 0) or 0
            cur_cit = paper.get("citations", 0) or 0
            prev_oa = bool(prev.get("is_open_access"))
            cur_oa = bool(paper.get("is_open_access"))
            if (cur_oa and not prev_oa) or (cur_cit > prev_cit):
                unique_map[key] = paper
        else:
            unique_map[key] = paper

    deduped = list(unique_map.values())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_input_records": len(merged),
                "total_unique_records": len(deduped),
                "duplicates_removed": duplicate_count,
                "overlap_fraction": (duplicate_count / len(merged)) if merged else 0.0,
                "papers": deduped,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    source_summary: Dict[str, int] = {}
    for rs in run_stats:
        for src, cnt in rs.get("source_counts", {}).items():
            source_summary[src] = source_summary.get(src, 0) + int(cnt)

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": run_stats,
                "source_summary_post_selection": source_summary,
                "raw_records": len(merged),
                "unique_records": len(deduped),
                "duplicates_removed": duplicate_count,
                "overlap_fraction": (duplicate_count / len(merged)) if merged else 0.0,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[OK] Corpus merged and deduplicated")
    print(f"  Input records: {len(merged)}")
    print(f"  Unique records: {len(deduped)}")
    print(f"  Duplicates removed: {duplicate_count}")
    print(f"  Dedup corpus: {output_path}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
