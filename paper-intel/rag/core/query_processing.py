"""
Query processing: expansion, rewriting, HyDE.
"""
import logging
from typing import List, Tuple
import httpx
import os
from dotenv import load_dotenv

load_dotenv("config/.env")

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Query expansion, rewriting, and HyDE."""
    
    def __init__(
        self,
        model: str = "meta-llama/llama-3.2-3b-instruct",
        provider: str = "openrouter",
        api_key: str = None
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def expand_query(
        self,
        query: str,
        num_expansions: int = 3
    ) -> List[str]:
        """
        Expand query with related terms.
        
        Returns:
            List of expanded queries including original
        """
        prompt = f"""Given this search query: "{query}"

Generate {num_expansions} alternative ways to search for the same information.
Each alternative should use different words but capture the same intent.

Return only the alternatives, one per line."""
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    expansions = [line.strip() for line in content.split("\n") if line.strip()]
                    return [query] + expansions[:num_expansions]
        
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return [query]
    
    async def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval.
        
        Returns:
            Rewritten query
        """
        prompt = f"""Rewrite this search query to be more effective for retrieving research papers:

Original: "{query}"

Rewritten query (be specific and use academic terminology):"""
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    }
                )
                
                if response.status_code == 200:
                    rewritten = response.json()["choices"][0]["message"]["content"].strip()
                    return rewritten if rewritten else query
        
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
        
        return query
    
    async def generate_hyde(
        self,
        query: str,
        num_generations: int = 2
    ) -> List[str]:
        """
        Generate hypothetical documents (HyDE).
        
        Returns:
            List of hypothetical document passages
        """
        prompt = f"""Write a detailed academic passage that would answer this question:

Question: {query}

Write a scholarly paragraph with specific details, citations, and technical terms:"""
        
        hypotheticals = []
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for i in range(num_generations):
                    response = await client.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.7 + (i * 0.1)  # Increase diversity
                        }
                    )
                    
                    if response.status_code == 200:
                        content = response.json()["choices"][0]["message"]["content"]
                        hypotheticals.append(content.strip())
        
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
        
        return hypotheticals


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using RRF.
    
    Args:
        ranked_lists: List of ranked results [(doc_id, score), ...]
        k: RRF constant (default 60)
        
    Returns:
        Fused ranking
    """
    doc_scores = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return fused
