"""
Reranker using BGE Reranker and Cross-Encoder
Improves retrieval quality by reranking initial results
"""
import numpy as np
from typing import List, Dict, Optional
from FlagEmbedding import FlagReranker


class BGEReranker:
    """BGE reranker for result reranking"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize BGE reranker
        
        Args:
            model_name: Reranker model name
            use_fp16: Use FP16 for faster inference
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"📥 Loading BGE reranker: {model_name}")
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16
        )
        print("✅ Reranker loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents based on query
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Return top-k results (None = all)
            
        Returns:
            List of reranked results with scores
        """
        if not documents:
            return []
        
        # Prepare pairs
        pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        scores = self.model.compute_score(
            pairs,
            batch_size=self.batch_size,
            normalize=True
        )
        
        # Ensure scores is a list
        if not isinstance(scores, list):
            scores = [scores]
        
        # Sort by score
        results = [
            {"document": doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def rerank_results(
        self,
        query: str,
        search_results: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank search results with metadata
        
        Args:
            query: Query text
            search_results: Results from vector search
            top_k: Return top-k results
            
        Returns:
            Reranked results with updated scores
        """
        if not search_results:
            return []
        
        # Extract documents
        documents = [r["document"] for r in search_results]
        
        # Rerank
        reranked = self.rerank(query, documents, top_k=None)
        
        # Merge with original metadata
        results = []
        for rerank_result in reranked:
            # Find original result
            for orig_result in search_results:
                if orig_result["document"] == rerank_result["document"]:
                    results.append({
                        **orig_result,
                        "rerank_score": rerank_result["score"],
                        "original_score": orig_result["score"]
                    })
                    break
        
        if top_k:
            results = results[:top_k]
        
        return results


class HybridReranker:
    """Hybrid reranker combining multiple strategies"""
    
    def __init__(
        self,
        use_bge: bool = True,
        bge_weight: float = 0.7
    ):
        """
        Initialize hybrid reranker
        
        Args:
            use_bge: Use BGE reranker
            bge_weight: Weight for BGE score
        """
        self.use_bge = use_bge
        self.bge_weight = bge_weight
        
        if use_bge:
            self.bge_reranker = BGEReranker()
    
    def rerank(
        self,
        query: str,
        search_results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Hybrid reranking with multiple signals
        
        Args:
            query: Query text
            search_results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not search_results:
            return []
        
        # BGE reranking
        if self.use_bge:
            results = self.bge_reranker.rerank_results(query, search_results)
        else:
            results = search_results
        
        # Apply hybrid scoring
        for result in results:
            original_score = result.get("original_score", result.get("score", 0))
            rerank_score = result.get("rerank_score", original_score)
            
            # Combine scores
            result["final_score"] = (
                self.bge_weight * rerank_score +
                (1 - self.bge_weight) * original_score
            )
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return results[:top_k]
