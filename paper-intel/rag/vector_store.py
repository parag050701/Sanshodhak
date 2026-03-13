"""
Vector Store with FAISS and ChromaDB
Handles vector storage and similarity search
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
import pickle
from dataclasses import asdict


class FAISSVectorStore:
    """FAISS-based vector store for fast similarity search"""
    
    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "IVF",
        nlist: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_type: Index type (Flat, IVF, HNSW)
            nlist: Number of clusters for IVF
            use_gpu: Use GPU acceleration
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        
        # Initialize index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Move to GPU if available
        if use_gpu and faiss.get_num_gpus() > 0:
            print("🚀 Using GPU for FAISS")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        # Metadata storage
        self.documents = []
        self.metadatas = []
        self.ids = []
        
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to vector store
        
        Args:
            embeddings: Document embeddings (N x D)
            documents: Document texts
            metadatas: Document metadata
            ids: Document IDs
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            print(f"🔧 Training FAISS index with {len(embeddings)} vectors")
            self.index.train(embeddings)
        
        # Add to index
        start_id = len(self.documents)
        self.index.add(embeddings)
        
        # Store metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        if ids:
            self.ids.extend(ids)
        else:
            self.ids.extend([f"doc_{start_id + i}" for i in range(len(documents))])
        
        print(f"✅ Added {len(documents)} documents to vector store (total: {len(self.documents)})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding (1 x D)
            k: Number of results
            filter_metadata: Filter by metadata
            
        Returns:
            List of results with documents, scores, and metadata
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        search_k = min(k * 3, len(self.documents))  # Oversample for filtering
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            metadata = self.metadatas[idx]
            
            # Apply metadata filter
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            results.append({
                "id": self.ids[idx],
                "document": self.documents[idx],
                "metadata": metadata,
                "score": float(score)
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def save(self, directory: str):
        """Save vector store to disk"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, str(directory / "index.faiss"))
        else:
            faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Save metadata
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self.ids,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f)
        
        print(f"✅ Vector store saved to {directory}")
    
    @classmethod
    def load(cls, directory: str, use_gpu: bool = False):
        """Load vector store from disk"""
        directory = Path(directory)
        
        # Load metadata
        with open(directory / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Create instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            use_gpu=use_gpu
        )
        
        # Load FAISS index
        index = faiss.read_index(str(directory / "index.faiss"))
        if use_gpu and faiss.get_num_gpus() > 0:
            store.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        else:
            store.index = index
        
        # Restore metadata
        store.documents = metadata["documents"]
        store.metadatas = metadata["metadatas"]
        store.ids = metadata["ids"]
        
        print(f"✅ Vector store loaded from {directory} ({len(store.documents)} documents)")
        return store


class HybridVectorStore:
    """Hybrid vector store combining dense and sparse search"""
    
    def __init__(
        self,
        dimension: int = 1024,
        use_gpu: bool = False
    ):
        self.dense_store = FAISSVectorStore(
            dimension=dimension,
            index_type="IVF",
            use_gpu=use_gpu
        )
        self.sparse_store = None  # BM25 or similar
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """Add documents to hybrid store"""
        self.dense_store.add_documents(embeddings, documents, metadatas, ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        alpha: float = 0.7
    ) -> List[Dict]:
        """
        Hybrid search combining dense and sparse
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            alpha: Weight for dense search (1-alpha for sparse)
        """
        # For now, just use dense search
        return self.dense_store.search(query_embedding, k)
    
    def save(self, directory: str):
        """Save hybrid store"""
        self.dense_store.save(directory)
    
    @classmethod
    def load(cls, directory: str, use_gpu: bool = False):
        """Load hybrid store"""
        store = cls(use_gpu=use_gpu)
        store.dense_store = FAISSVectorStore.load(directory, use_gpu)
        return store
