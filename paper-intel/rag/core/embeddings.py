"""
Advanced embedding models for RAG system.
"""
import logging
from typing import List, Optional
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class DenseEmbedder:
    """Dense embeddings using BGE-M3."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        
        logger.info(f"Loading dense model: {model_name} on {self.device}")
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=self.device == "cuda"
        )
        logger.info("Dense model loaded")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed documents in batches."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed_documents([query])[0]


class SparseEmbedder:
    """Sparse embeddings using BM25."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
    
    def fit(self, texts: List[str]):
        """Fit BM25 on corpus."""
        logger.info(f"Fitting BM25 on {len(texts)} documents")
        self.corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        logger.info("BM25 fitted")
    
    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for query."""
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        query_tokens = query.lower().split()
        return np.array(self.bm25.get_scores(query_tokens))


class HybridEmbedder:
    """Combines dense and sparse embeddings."""
    
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        self.dense = dense_embedder
        self.sparse = sparse_embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        logger.info(
            f"Hybrid embedder: dense={dense_weight}, sparse={sparse_weight}"
        )
    
    def fit_sparse(self, texts: List[str]):
        """Fit sparse embedder."""
        self.sparse.fit(texts)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed documents (dense only for indexing)."""
        return self.dense.embed_documents(texts)
    
    def hybrid_search(
        self,
        query: str,
        dense_embeddings: np.ndarray,
        top_k: int = 100
    ) -> List[tuple]:
        """
        Perform hybrid search combining dense and sparse.
        
        Returns:
            List of (doc_idx, combined_score) tuples
        """
        # Dense search
        query_emb = self.dense.embed_query(query)
        dense_scores = np.dot(dense_embeddings, query_emb)
        
        # Sparse search
        sparse_scores = self.sparse.get_scores(query)
        
        # Normalize scores
        dense_scores = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-9
        )
        sparse_scores = (sparse_scores - sparse_scores.min()) / (
            sparse_scores.max() - sparse_scores.min() + 1e-9
        )
        
        # Combine
        combined_scores = (
            self.dense_weight * dense_scores + 
            self.sparse_weight * sparse_scores
        )
        
        # Get top k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = [(int(idx), float(combined_scores[idx])) for idx in top_indices]
        
        return results
