"""
Retrieval Diversity via Maximal Marginal Relevance (MMR).

Penalises retrieved chunks that are overly similar to already-selected
chunks to maximise information coverage in the context window.
"""

from typing import List, Dict, Any
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def mmr_rerank(
    results: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    lambda_param: float = 0.5,
    embeddings: List[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Reorder results using Maximal Marginal Relevance (MMR).

    Formula:
        MMR = argmax_{D_i in R\\S} [ λ * Sim1(D_i, Q) - (1 - λ) * max_{D_j in S} Sim2(D_i, D_j) ]

    Parameters
    ----------
    results         : List of retrieved document dictionaries.
                      Should contain original relevance metadata.
    query_embedding : 1D numpy array representing the query.
    lambda_param    : Float in [0, 1]. Trade-off between relevance to query (λ=1)
                      and diversity from already selected documents (λ=0).
    embeddings      : List of 1D numpy arrays parallel to `results`.

    Returns
    -------
    List[Dict] : Re-ordered results.
    """
    if not results or not embeddings or len(results) != len(embeddings):
        return results

    # Pre-compute document-query similarities
    doc_query_sims = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    selected_indices = []
    unselected_indices = list(range(len(results)))
    
    # First selected document is the most relevant one
    first_idx = int(np.argmax(doc_query_sims))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)
    
    while unselected_indices:
        best_idx = -1
        max_mmr = -float("inf")
        
        for cand_idx in unselected_indices:
            # Relevance to query
            relevance = doc_query_sims[cand_idx]
            
            # Max similarity to already selected docs
            max_sim_to_selected = max(
                cosine_similarity(embeddings[cand_idx], embeddings[sel_idx])
                for sel_idx in selected_indices
            )
            
            # MMR Score
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected
            
            if mmr_score > max_mmr:
                max_mmr = mmr_score
                best_idx = cand_idx
                
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
    return [results[i] for i in selected_indices]
