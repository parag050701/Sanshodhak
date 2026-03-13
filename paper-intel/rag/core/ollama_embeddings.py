"""
Ollama-based embeddings for RAG system.
"""
import logging
from typing import List
import numpy as np
import requests

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Embeddings using Ollama's BGE-M3."""
    
    def __init__(
        self,
        model_name: str = "bge-m3:latest",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        
        logger.info(f"Using Ollama embedder: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed documents using Ollama."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for text in batch:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    embeddings.append(embedding)
                else:
                    logger.error(f"Ollama error: {response.status_code}")
                    # Return zero vector on error
                    embeddings.append([0.0] * 1024)
            
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Embedded {i + len(batch)}/{len(texts)} documents")
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed_documents([query])[0]
