"""
FAISS-based vector database with GPU support.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class VectorDatabase:
    """FAISS vector database with GPU support."""
    
    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "IVF1024,Flat",
        metric: str = "cosine",
        use_gpu: bool = True,
        nprobe: int = 64
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.nprobe = nprobe
        
        self.index: Optional[faiss.Index] = None
        self.gpu_index: Optional[faiss.Index] = None
        self.doc_ids: List[str] = []
        
        logger.info(f"Vector DB: dim={dimension}, type={index_type}, gpu={self.use_gpu}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index."""
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            measure = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            measure = faiss.METRIC_L2
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Parse index type
        if self.index_type == "Flat":
            index = faiss.IndexFlat(self.dimension, measure)
        elif self.index_type.startswith("IVF"):
            # IVF index (inverted file index)
            parts = self.index_type.split(",")
            n_list = int(parts[0].replace("IVF", ""))
            
            quantizer = faiss.IndexFlat(self.dimension, measure)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, n_list, measure)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def build(self, embeddings: np.ndarray, doc_ids: List[str]):
        """
        Build index from embeddings.
        
        Args:
            embeddings: Document embeddings (N x D)
            doc_ids: Document identifiers
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError("Embeddings and doc_ids must have same length")
        
        logger.info(f"Building index with {len(embeddings)} vectors")
        
        # Create index
        self.index = self._create_index()
        self.doc_ids = doc_ids
        
        # Train if needed
        if isinstance(self.index, faiss.IndexIVF):
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        # Add vectors
        self.index.add(embeddings)
        
        # Move to GPU if available
        if self.use_gpu:
            logger.info("Moving index to GPU")
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            self.gpu_index = self.index
        
        logger.info(f"Index built: {self.index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding (D,)
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of (doc_id, score) tuples
        """
        if self.gpu_index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.gpu_index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]
        
        # Filter by score and convert to doc_ids
        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and score >= min_score:
                results.append((self.doc_ids[idx], float(score)))
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100
    ) -> List[List[Tuple[str, float]]]:
        """Batch search for multiple queries."""
        if self.gpu_index is None:
            raise ValueError("Index not built. Call build() first.")
        
        scores, indices = self.gpu_index.search(query_embeddings, top_k)
        
        results = []
        for batch_scores, batch_indices in zip(scores, indices):
            batch_results = [
                (self.doc_ids[idx], float(score))
                for idx, score in zip(batch_indices, batch_scores)
                if idx >= 0
            ]
            results.append(batch_results)
        
        return results
    
    def save(self, path: Path):
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save CPU index
        logger.info(f"Saving index to {path}")
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save doc_ids
        with open(path / "doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nprobe": self.nprobe,
            "ntotal": self.index.ntotal
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info("Index saved")
    
    def load(self, path: Path):
        """Load index from disk."""
        logger.info(f"Loading index from {path}")
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.metric = metadata["metric"]
        self.nprobe = metadata["nprobe"]
        
        # Load index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = self.nprobe
        
        # Load doc_ids
        with open(path / "doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)
        
        # Move to GPU if available
        if self.use_gpu:
            logger.info("Moving index to GPU")
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            self.gpu_index = self.index
        
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
