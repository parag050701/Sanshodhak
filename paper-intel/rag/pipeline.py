"""
RAG Pipeline
Complete RAG system with BGE-M3, FAISS, and reranking
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np

from .embeddings import BGEEmbedder
from .chunking import PaperChunker, Chunk
from .vector_store import FAISSVectorStore, HybridVectorStore
from .reranker import HybridReranker


class RAGPipeline:
    """Complete RAG pipeline for paper search"""
    
    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_gpu: bool = False,
        use_reranking: bool = True
    ):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store_path: Path to load existing vector store
            chunk_size: Chunk size for documents
            chunk_overlap: Overlap between chunks
            use_gpu: Use GPU acceleration
            use_reranking: Enable reranking
        """
        print("🚀 Initializing RAG Pipeline")
        
        # Initialize components
        self.embedder = BGEEmbedder(use_fp16=use_gpu)
        self.chunker = PaperChunker(chunk_size, chunk_overlap)
        
        # Load or create vector store
        if vector_store_path and Path(vector_store_path).exists():
            print(f"📂 Loading vector store from {vector_store_path}")
            self.vector_store = FAISSVectorStore.load(vector_store_path, use_gpu)
        else:
            print("🔨 Creating new vector store")
            self.vector_store = FAISSVectorStore(
                dimension=self.embedder.embedding_dim,
                index_type="IVF",
                use_gpu=use_gpu
            )
        
        # Initialize reranker
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = HybridReranker(use_bge=True)
        
        print("✅ RAG Pipeline ready")
    
    def index_papers(
        self,
        text_dir: str,
        metadata_dir: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Index papers from text directory
        
        Args:
            text_dir: Directory containing text files
            metadata_dir: Directory containing paper metadata
            save_path: Path to save vector store
        """
        text_dir = Path(text_dir)
        text_files = sorted(text_dir.glob("*.txt"))
        
        print(f"📚 Indexing {len(text_files)} papers")
        
        all_chunks = []
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        for text_file in tqdm(text_files, desc="Processing papers"):
            # Read text
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract paper ID
            paper_id = text_file.stem
            
            # Load metadata if available
            metadata = {"paper_id": paper_id, "source_file": text_file.name}
            if metadata_dir:
                metadata_file = Path(metadata_dir) / f"{paper_id}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        paper_meta = json.load(f)
                        metadata.update(paper_meta)
            
            # Chunk paper
            chunks = self.chunker.chunk_paper(text, paper_id, metadata)
            
            # Collect chunks
            for chunk in chunks:
                all_chunks.append(chunk)
                all_texts.append(chunk.text)
                all_metadatas.append(chunk.metadata)
                all_ids.append(f"{paper_id}_chunk_{chunk.chunk_id}")
        
        print(f"📝 Generated {len(all_chunks)} chunks from {len(text_files)} papers")
        
        # Encode chunks in batches
        print("🔢 Encoding chunks with BGE-M3")
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
            batch_texts = all_texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch_texts, show_progress=False)
            all_embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(all_embeddings)
        print(f"✅ Encoded {len(all_embeddings)} chunks")
        
        # Add to vector store
        print("💾 Adding to vector store")
        self.vector_store.add_documents(
            embeddings=all_embeddings,
            documents=all_texts,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        # Save vector store
        if save_path:
            print(f"💾 Saving vector store to {save_path}")
            self.vector_store.save(save_path)
        
        print("✅ Indexing complete!")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 50,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of final results
            rerank_k: Number of results to retrieve before reranking
            filter_metadata: Filter by metadata
            
        Returns:
            List of search results
        """
        # Encode query
        query_embedding = self.embedder.encode([query])[0]
        
        # Search vector store
        search_k = rerank_k if self.use_reranking else top_k
        results = self.vector_store.search(
            query_embedding,
            k=search_k,
            filter_metadata=filter_metadata
        )
        
        # Rerank if enabled
        if self.use_reranking and len(results) > 0:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        max_context_length: int = 4000
    ) -> Dict:
        """
        Answer question using RAG
        
        Args:
            question: Question to answer
            top_k: Number of chunks to retrieve
            max_context_length: Maximum context length
            
        Returns:
            Dictionary with answer and sources
        """
        # Search for relevant chunks
        results = self.search(question, top_k=top_k)
        
        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "context": ""
            }
        
        # Build context
        context_parts = []
        sources = []
        total_length = 0
        
        for result in results:
            text = result["document"]
            text_length = len(text.split())
            
            if total_length + text_length > max_context_length:
                break
            
            context_parts.append(text)
            sources.append({
                "paper_id": result["metadata"].get("paper_id"),
                "section": result["metadata"].get("section", "unknown"),
                "score": result.get("final_score", result.get("score"))
            })
            total_length += text_length
        
        context = "\n\n".join(context_parts)
        
        return {
            "question": question,
            "context": context,
            "sources": sources,
            "num_chunks": len(sources)
        }
    
    def get_paper_summary(self, paper_id: str, max_chunks: int = 10) -> Dict:
        """
        Get summary of a specific paper
        
        Args:
            paper_id: Paper identifier
            max_chunks: Maximum chunks to retrieve
            
        Returns:
            Paper summary with key chunks
        """
        # Search for chunks from this paper
        results = self.vector_store.search(
            query_embedding=np.zeros((1, self.embedder.embedding_dim)),  # Dummy query
            k=max_chunks,
            filter_metadata={"paper_id": paper_id}
        )
        
        return {
            "paper_id": paper_id,
            "chunks": results
        }
