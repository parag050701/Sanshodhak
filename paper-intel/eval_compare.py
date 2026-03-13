"""
Comparative Evaluation: Vector RAG vs Graph RAG
================================================

Systems under comparison:
  1. Dense-only RAG  (VR-D)   — FAISS cosine similarity,  no BM25, no graph
  2. Hybrid RAG      (VR-H)   — BM25 + FAISS + RRF,       no graph
  3. Graph RAG       (GR)     — FAISS + graph expansion,   no BM25
  4. Hybrid+Graph    (HGR)    — BM25 + FAISS + graph + RRF (full system)

Retrieval metrics  : P@1, P@5, P@10, R@1, R@5, R@10, NDCG@5, NDCG@10, MRR, MAP
Generation metrics : ROUGE-1, ROUGE-2, ROUGE-L, BLEU, BERTScore-P/R/F1
Latency metrics    : mean, median, P95 (seconds)

Output files:
  eval_results.json          — full per-question results
  eval_summary.json          — aggregated metric table (ready for paper)
  eval_report.txt            — human-readable summary table

Usage:
  cd /home/admin-/Desktop/Sanshodhak/paper-intel
  python eval_compare.py [--index rag_index] [--questions rag_test_questions.json]
                         [--sample N] [--no-generation] [--no-bertscore]
"""

import argparse
import json
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

# ── optional generation-quality dependencies ─────────────────────────────────
try:
    from rouge_score import rouge_scorer as _rouge_scorer_module
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠  rouge-score not installed — ROUGE metrics will be 0.0")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠  nltk not installed — BLEU will be 0.0")

try:
    from bert_score import score as _bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠  bert-score not installed — BERTScore will be 0.0")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0 or not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in set(relevant))
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in set(relevant))
    return hits / len(set(relevant))


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum(
        (1.0 if d in relevant_set else 0.0) / math.log2(i + 2)
        for i, d in enumerate(retrieved[:k])
    )
    ideal_k = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    n_relevant, sum_p = 0, 0.0
    for k, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            n_relevant += 1
            sum_p += n_relevant / k
    return sum_p / len(relevant_set)


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    if not ROUGE_AVAILABLE or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = _rouge_scorer_module.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    s = scorer.score(reference, prediction)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }


def compute_bleu(prediction: str, reference: str) -> float:
    if not BLEU_AVAILABLE or not reference:
        return 0.0
    try:
        return sentence_bleu(
            [reference.lower().split()],
            prediction.lower().split(),
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        return 0.0


def compute_bertscore_batch(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    if not BERTSCORE_AVAILABLE or not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        P, R, F1 = _bert_score_fn(
            predictions, references, lang="en", verbose=False
        )
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }
    except Exception as e:
        print(f"⚠  BERTScore failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper — uniform interface over all four RAG systems
# ─────────────────────────────────────────────────────────────────────────────

class RAGWrapper:
    """
    Wraps a loaded RAG object (OllamaRAG / EnhancedRAG / GraphRAG) and
    exposes:
      .retrieve(query, top_k) → List[str]   (deduplicated paper filenames, ranked)
      .generate(query, top_k) → (str, float) (answer, latency_seconds)
    """

    def __init__(self, name: str, rag_obj):
        self.name = name
        self._rag = rag_obj
        self._is_enhanced = hasattr(rag_obj, "use_bm25") or hasattr(
            rag_obj, "bm25_docs"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Return deduplicated paper filenames in retrieval rank order."""
        try:
            if hasattr(self._rag, "search"):
                if self._is_enhanced:
                    # EnhancedRAG / GraphRAG signature
                    try:
                        results = self._rag.search(
                            query, top_k=top_k, use_hybrid=True, use_rerank=False
                        )
                    except TypeError:
                        results = self._rag.search(query, top_k=top_k)
                else:
                    results = self._rag.search(query, top_k=top_k)
            else:
                return []

            # Deduplicate at paper level while preserving rank order
            seen, ordered = set(), []
            for r in results:
                fname = r["metadata"]["file"]
                if fname not in seen:
                    seen.add(fname)
                    ordered.append(fname)
            return ordered
        except Exception as e:
            logger.error(f"[{self.name}] retrieve error: {e}")
            return []

    def generate(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Run full query and return (answer, latency_seconds)."""
        t0 = time.time()
        try:
            if hasattr(self._rag, "query"):
                try:
                    answer, _ = self._rag.query(
                        query, top_k=top_k, verbose=False,
                        use_hybrid=True, use_rerank=False
                    )
                except TypeError:
                    try:
                        answer, _ = self._rag.query(
                            query, top_k=top_k, verbose=False
                        )
                    except TypeError:
                        answer, _ = self._rag.query(query)
            else:
                answer = "[no query method]"
        except Exception as e:
            logger.error(f"[{self.name}] generate error: {e}")
            answer = ""
        return answer, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    systems: List[RAGWrapper],
    test_questions: List[Dict],
    top_k_values: List[int],
    run_generation: bool = True,
    run_bertscore: bool = True,
    sample_size: Optional[int] = None,
) -> Dict:
    """
    Run full evaluation across all systems.

    Returns nested dict:
      results[system_name] = {
          "retrieval": { "P@1": ..., "R@5": ..., ... },
          "generation": { "rouge1": ..., "bleu": ..., ... },
          "latency": { "mean": ..., "median": ..., "p95": ... },
          "per_question": [ ... ]
      }
    """
    if sample_size:
        test_questions = test_questions[:sample_size]

    total = len(test_questions)
    print(f"\n📋 {total} test questions — {len(systems)} systems")

    all_results: Dict = {}

    for wrapper in systems:
        name = wrapper.name
        print(f"\n{'='*70}")
        print(f"  System: {name}")
        print(f"{'='*70}")

        per_q: List[Dict] = []
        latencies: List[float] = []
        predictions: List[str] = []
        references: List[str] = []

        for i, item in enumerate(test_questions):
            question = item["question"]
            relevant_docs = item.get("relevant_docs", [])
            reference_answer = item.get("answer", "")

            # ── Retrieval ──────────────────────────────────────────────
            t0 = time.time()
            retrieved = wrapper.retrieve(question, top_k=max(top_k_values))
            retrieval_latency = time.time() - t0

            # ── Retrieval metrics ──────────────────────────────────────
            q_ret: Dict = {}
            for k in top_k_values:
                q_ret[f"P@{k}"] = precision_at_k(retrieved, relevant_docs, k)
                q_ret[f"R@{k}"] = recall_at_k(retrieved, relevant_docs, k)
                q_ret[f"NDCG@{k}"] = ndcg_at_k(retrieved, relevant_docs, k)
            q_ret["RR"] = reciprocal_rank(retrieved, relevant_docs)
            q_ret["AP"] = average_precision(retrieved, relevant_docs)

            # ── Generation ─────────────────────────────────────────────
            q_gen: Dict = {}
            if run_generation:
                answer, gen_latency = wrapper.generate(question, top_k=5)
                total_latency = retrieval_latency + gen_latency
                latencies.append(total_latency)

                if reference_answer:
                    q_gen.update(compute_rouge(answer, reference_answer))
                    q_gen["bleu"] = compute_bleu(answer, reference_answer)
                    predictions.append(answer)
                    references.append(reference_answer)
            else:
                latencies.append(retrieval_latency)
                answer = ""

            per_q.append(
                {
                    "question": question,
                    "retrieved": retrieved,
                    "relevant_docs": relevant_docs,
                    "retrieval_metrics": q_ret,
                    "generation_metrics": q_gen,
                    "answer": answer,
                    "reference_answer": reference_answer,
                }
            )

            prog = f"[{i+1}/{total}]"
            p1 = q_ret.get("P@1", 0)
            ndcg5 = q_ret.get("NDCG@5", q_ret.get(f"NDCG@{top_k_values[0]}", 0))
            print(f"  {prog} P@1={p1:.2f} NDCG@5={ndcg5:.2f}  {question[:55]}…")

        # ── Aggregate retrieval ────────────────────────────────────────
        agg_ret: Dict = {}
        ret_keys = list(per_q[0]["retrieval_metrics"].keys()) if per_q else []
        for key in ret_keys:
            vals = [q["retrieval_metrics"].get(key, 0.0) for q in per_q]
            agg_ret[key] = float(np.mean(vals))
        # Rename RR→MRR, AP→MAP
        agg_ret["MRR"] = agg_ret.pop("RR", 0.0)
        agg_ret["MAP"] = agg_ret.pop("AP", 0.0)

        # ── Aggregate generation ───────────────────────────────────────
        agg_gen: Dict = {}
        if run_generation and per_q[0]["generation_metrics"]:
            gen_keys = list(per_q[0]["generation_metrics"].keys())
            for key in gen_keys:
                vals = [
                    q["generation_metrics"].get(key, 0.0) for q in per_q
                    if q["generation_metrics"]
                ]
                agg_gen[key] = float(np.mean(vals)) if vals else 0.0

            # BERTScore (batch — more efficient)
            if run_bertscore and predictions and references:
                bscores = compute_bertscore_batch(predictions, references)
                agg_gen["bertscore_precision"] = bscores["precision"]
                agg_gen["bertscore_recall"] = bscores["recall"]
                agg_gen["bertscore_f1"] = bscores["f1"]

        # ── Aggregate latency ──────────────────────────────────────────
        agg_lat = {
            "mean_s": float(np.mean(latencies)),
            "median_s": float(np.median(latencies)),
            "p95_s": float(np.percentile(latencies, 95)),
        }

        all_results[name] = {
            "retrieval": agg_ret,
            "generation": agg_gen,
            "latency": agg_lat,
            "per_question": per_q,
        }

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(all_results: Dict, top_k_values: List[int]) -> Dict:
    """Build a flat metric table suitable for a LaTeX/paper table."""
    systems = list(all_results.keys())
    summary: Dict = {}

    # Retrieval metric keys in display order
    ret_keys = (
        [f"P@{k}" for k in top_k_values]
        + [f"R@{k}" for k in top_k_values]
        + [f"NDCG@{k}" for k in top_k_values]
        + ["MRR", "MAP"]
    )
    gen_keys = [
        "rouge1", "rouge2", "rougeL", "bleu",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
    ]
    lat_keys = ["mean_s", "median_s", "p95_s"]

    for sys in systems:
        summary[sys] = {}
        ret = all_results[sys]["retrieval"]
        gen = all_results[sys]["generation"]
        lat = all_results[sys]["latency"]
        for k in ret_keys:
            summary[sys][k] = round(ret.get(k, 0.0), 4)
        for k in gen_keys:
            summary[sys][k] = round(gen.get(k, 0.0), 4)
        for k in lat_keys:
            summary[sys][k] = round(lat.get(k, 0.0), 3)

    return summary


def print_report(summary: Dict, all_results: Dict):
    """Print a compact comparison table to stdout and return as string."""
    systems = list(summary.keys())
    col_w = max(12, max(len(s) for s in systems))

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("COMPARATIVE EVALUATION — SANSHODHAK RAG SYSTEMS")
    lines.append("=" * 90)

    # ── Retrieval table ───────────────────────────────────────────────────
    ret_metrics = [
        "P@1", "P@5", "P@10",
        "R@1", "R@5", "R@10",
        "NDCG@5", "NDCG@10",
        "MRR", "MAP",
    ]
    header = f"{'Metric':<14}" + "".join(f"{s:>{col_w}}" for s in systems)
    sep = "-" * len(header)
    lines.append("\nRetrieval Metrics:")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for m in ret_metrics:
        if m not in summary[systems[0]]:
            continue
        row = f"{m:<14}" + "".join(
            f"{summary[s].get(m, 0.0):>{col_w}.4f}" for s in systems
        )
        lines.append(row)
    lines.append(sep)

    # ── Generation table ──────────────────────────────────────────────────
    gen_metrics = [
        "rouge1", "rouge2", "rougeL", "bleu",
        "bertscore_f1",
    ]
    any_gen = any(summary[s].get("rouge1", 0.0) > 0 for s in systems)
    if any_gen:
        lines.append("\nGeneration Metrics:")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        labels = {
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2",
            "rougeL": "ROUGE-L",
            "bleu": "BLEU",
            "bertscore_f1": "BERTScore F1",
        }
        for m in gen_metrics:
            if not any(summary[s].get(m, 0.0) > 0 for s in systems):
                continue
            row = f"{labels[m]:<14}" + "".join(
                f"{summary[s].get(m, 0.0):>{col_w}.4f}" for s in systems
            )
            lines.append(row)
        lines.append(sep)

    # ── Latency table ─────────────────────────────────────────────────────
    lines.append("\nLatency (seconds):")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for m, lbl in [("mean_s", "Mean"), ("median_s", "Median"), ("p95_s", "P95")]:
        row = f"{lbl:<14}" + "".join(
            f"{summary[s].get(m, 0.0):>{col_w}.3f}" for s in systems
        )
        lines.append(row)
    lines.append(sep)

    # ── Graph stats (if GraphRAG present) ─────────────────────────────────
    for sys_name, res in all_results.items():
        if "graph_stats" in res:
            gs = res["graph_stats"]
            lines.append(f"\nKnowledge Graph ({sys_name}):")
            for k, v in gs.items():
                lines.append(f"  {k}: {v}")

    lines.append("=" * 90)
    report = "\n".join(lines)
    print(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Comparative RAG evaluation")
    p.add_argument("--index", default="rag_index",
                   help="Path to pre-built FAISS index directory")
    p.add_argument("--questions", default="rag_test_questions.json",
                   help="Test questions JSON file")
    p.add_argument("--sample", type=int, default=None,
                   help="Evaluate on first N questions only (default: all)")
    p.add_argument("--no-generation", action="store_true",
                   help="Skip LLM generation (retrieval metrics only)")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Skip BERTScore (slow, needs GPU for best results)")
    p.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10],
                   help="Top-K values for retrieval metrics")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Jaccard similarity threshold for graph edges")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).parent
    index_dir = str(base_dir / args.index)
    questions_path = base_dir / args.questions

    # ── Load test questions ────────────────────────────────────────────────
    print(f"\n📄 Loading questions from {questions_path} …")
    with open(questions_path) as f:
        test_questions = json.load(f)
    print(f"   {len(test_questions)} questions loaded")

    # ── Instantiate systems ────────────────────────────────────────────────
    print("\n🔧 Loading RAG systems …")

    systems: List[RAGWrapper] = []

    # 1. Dense-only (OllamaRAG — FAISS cosine, no BM25, no graph)
    print("  [1/4] Dense-only (VR-D) …")
    from ollama_rag import OllamaRAG
    vrd = OllamaRAG(embed_model="bge-m3", llm_model="deepseek-r1:7b",
                    use_openrouter=False)
    vrd.load(index_dir)
    systems.append(RAGWrapper("VR-D (Dense)", vrd))

    # 2. Hybrid RAG (EnhancedRAG — BM25 + FAISS + RRF, no graph)
    print("  [2/4] Hybrid RAG (VR-H) …")
    from enhanced_rag import EnhancedRAG
    vrh = EnhancedRAG(use_ollama_fallback=True,
                      ollama_embed="bge-m3",
                      ollama_llm="deepseek-r1:7b")
    vrh.load(index_dir)
    systems.append(RAGWrapper("VR-H (Hybrid)", vrh))

    # 3. Graph RAG (FAISS + graph expansion, no BM25)
    print("  [3/4] Graph RAG (GR) …")
    from graph_rag import GraphRAG
    gr = GraphRAG(use_bm25=False,
                  ollama_embed="bge-m3",
                  ollama_llm="deepseek-r1:7b",
                  similarity_threshold=args.threshold)
    gr.load(index_dir)
    graph_stats = gr.get_graph_stats()
    systems.append(RAGWrapper("GR (Graph)", gr))

    # 4. Hybrid + Graph (BM25 + FAISS + graph expansion + RRF)
    print("  [4/4] Hybrid+Graph (HGR) …")
    hgr = GraphRAG(use_bm25=True,
                   ollama_embed="bge-m3",
                   ollama_llm="deepseek-r1:7b",
                   similarity_threshold=args.threshold)
    hgr.load(index_dir)
    systems.append(RAGWrapper("HGR (Hybrid+Graph)", hgr))

    print("\n✅ All systems ready")

    # ── Run evaluation ─────────────────────────────────────────────────────
    all_results = run_evaluation(
        systems=systems,
        test_questions=test_questions,
        top_k_values=args.top_k,
        run_generation=not args.no_generation,
        run_bertscore=not args.no_bertscore,
        sample_size=args.sample,
    )

    # Attach graph stats for reporting
    all_results["GR (Graph)"]["graph_stats"] = graph_stats
    all_results["HGR (Hybrid+Graph)"]["graph_stats"] = graph_stats

    # ── Build summary ──────────────────────────────────────────────────────
    summary = build_summary_table(all_results, args.top_k)
    report_str = print_report(summary, all_results)

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir = base_dir

    # Strip per-question details for summary file (keep file sizes manageable)
    summary_out = {
        sys: {k: v for k, v in data.items() if k != "per_question"}
        for sys, data in all_results.items()
    }
    summary_out["_meta"] = {
        "questions_file": str(questions_path),
        "index_dir": index_dir,
        "n_questions": len(test_questions),
        "top_k_values": args.top_k,
        "graph_similarity_threshold": args.threshold,
    }

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    with open(out_dir / "eval_results.json", "w") as f:
        # per_question answers can be large; save separately
        json.dump(all_results, f, indent=2, default=str)

    with open(out_dir / "eval_report.txt", "w") as f:
        f.write(report_str)

    print(f"\n✅ Results saved:")
    print(f"   eval_summary.json  — aggregated metrics (for paper tables)")
    print(f"   eval_results.json  — per-question results")
    print(f"   eval_report.txt    — human-readable comparison table")


if __name__ == "__main__":
    main()
