"""
Sanshodhak Core — Modular IR system components.

Modules
-------
retrieval        : DenseRetriever (FAISS), BM25Retriever, HybridRetriever
graph            : GraphIndex (4 edge types + 4 traversal modes)
fusion           : reciprocal_rank_fusion (RRF) + confidence_weighted_rrf (CW-RRF)
adaptive_budget  : compute_graph_budget — dynamic per-query graph slot allocation
reranker         : CrossEncoderReranker (sentence-transformers cross-encoder)
evaluation       : NDCG, MAP, MRR, Recall metrics + Wilcoxon/t-test
diversity        : mmr_rerank — Maximal Marginal Relevance post-retrieval diversity
query_classifier : classify_query — lightweight query difficulty classification
"""

from .retrieval import EmbeddingBackend, BM25Retriever, DenseRetriever, HybridRetriever
from .graph import GraphIndex
from .fusion import reciprocal_rank_fusion, confidence_weighted_rrf
from .adaptive_budget import compute_graph_budget
from .diversity import mmr_rerank
from .query_classifier import classify_query
from .reranker import CrossEncoderReranker
from .evaluation import (
    ndcg_at_k,
    ndcg_at_k_graded,
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    evaluate_run,
    wilcoxon_test,
    paired_ttest,
)

__all__ = [
    "EmbeddingBackend",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "GraphIndex",
    "reciprocal_rank_fusion",
    "confidence_weighted_rrf",
    "compute_graph_budget",
    "CrossEncoderReranker",
    "mmr_rerank",
    "classify_query",
    "ndcg_at_k",
    "ndcg_at_k_graded",
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "reciprocal_rank",
    "evaluate_run",
    "wilcoxon_test",
    "paired_ttest",
]
