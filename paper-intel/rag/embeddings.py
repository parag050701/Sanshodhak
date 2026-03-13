"""
BGE-M3 Embedding Manager
Handles document embedding with BGE-M3 model with caching and batch processing
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel


class BGEEmbedder:
    """BGE-M3 embedding model with optimizations"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 8192
    ):
        """
        Initialize BGE-M3 embedder
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model
            use_fp16: Use FP16 for faster inference
            batch_size: Batch size for encoding
            max_length: Maximum token length
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "models/bge-m3"
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        print(f"📥 Loading BGE-M3 model: {model_name}")
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 and self.device == "cuda"
        )
        print("✅ BGE-M3 model loaded")
        
    def encode(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to dense embeddings
        
        Args:
            texts: List of texts to encode
            show_progress: Show progress bar
            
        Returns:
            Dense embeddings array (N x D)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        
        # Extract dense embeddings
        if isinstance(embeddings, dict):
            embeddings = embeddings['dense_vecs']
        
        return np.array(embeddings)
    
    def encode_with_metadata(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> Dict:
        """
        Encode texts with all embedding types (dense, sparse, colbert)
        
        Args:
            texts: List of texts to encode
            show_progress: Show progress bar
            
        Returns:
            Dictionary with dense, sparse, and colbert embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return 1024  # BGE-M3 dimension


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation"""
    
    def __init__(self, cache_dir: str = "rag/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, text_hash: str) -> Path:
        """Get cache file path for text hash"""
        return self.cache_dir / f"{text_hash}.npy"
    
    def save(self, text_hash: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_path = self.get_cache_path(text_hash)
        np.save(cache_path, embedding)
    
    def load(self, text_hash: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        cache_path = self.get_cache_path(text_hash)
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def exists(self, text_hash: str) -> bool:
        """Check if embedding exists in cache"""
        return self.get_cache_path(text_hash).exists()
