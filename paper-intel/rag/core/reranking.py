"""
Reranking using BGE reranker model.
"""
import logging
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class Reranker:
    """BGE Reranker for improving retrieval quality."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 16
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        logger.info(f"Loading reranker: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Reranker loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Query string
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of (doc_idx, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Prepare pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score in batches
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get scores
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                all_scores.extend(scores.tolist())
        
        # Sort by score
        scored_docs = [(idx, score) for idx, score in enumerate(all_scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


class MaximalMarginalRelevance:
    """MMR for diversity in results."""
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
                         1.0 = pure relevance, 0.0 = pure diversity
        """
        self.lambda_param = lambda_param
    
    def select(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
        scores: List[float],
        top_k: int = 10
    ) -> List[int]:
        """
        Select diverse documents using MMR.
        
        Args:
            query_embedding: Query embedding (D,)
            doc_embeddings: Document embeddings (N, D)
            scores: Initial relevance scores
            top_k: Number of documents to select
            
        Returns:
            List of selected document indices
        """
        selected_indices = []
        remaining_indices = list(range(len(scores)))
        
        # Select first document (highest relevance)
        first_idx = max(remaining_indices, key=lambda i: scores[i])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select diverse documents
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = scores[idx]
                
                # Max similarity to selected documents
                similarities = []
                for sel_idx in selected_indices:
                    sim = torch.cosine_similarity(
                        doc_embeddings[idx].unsqueeze(0),
                        doc_embeddings[sel_idx].unsqueeze(0)
                    ).item()
                    similarities.append(sim)
                
                max_similarity = max(similarities) if similarities else 0
                
                # MMR score
                mmr = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity
                )
                mmr_scores.append(mmr)
            
            # Select document with highest MMR
            best_idx_pos = max(range(len(mmr_scores)), key=lambda i: mmr_scores[i])
            best_idx = remaining_indices[best_idx_pos]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices
