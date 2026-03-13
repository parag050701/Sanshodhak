"""
Answer generation with citations.
"""
import logging
from typing import List, Dict, Any
import httpx
import os
from dotenv import load_dotenv

load_dotenv("config/.env")

logger = logging.getLogger(__name__)


class Generator:
    """Generate answers with citations."""
    
    def __init__(
        self,
        model: str = "meta-llama/llama-3.2-3b-instruct",
        provider: str = "openrouter",
        system_prompt: str = None,
        api_key: str = None
    ):
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer from contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts with paper_id and text
            
        Returns:
            Generated answer with citations
        """
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1} - Paper: {ctx['paper_id']}]\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""Context from research papers:

{context_str}

Question: {query}

Answer the question based on the context above. Include citations to specific papers [Paper: ID] when referencing information. If the context doesn't contain enough information, say so.

Answer:"""
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 2000
                    }
                )
                
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    return answer.strip()
                else:
                    logger.error(f"Generation failed: {response.status_code}")
                    return "Error generating answer"
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Error generating answer"
