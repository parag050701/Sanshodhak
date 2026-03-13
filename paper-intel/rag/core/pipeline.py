"""
Advanced RAG pipeline with all optimizations.
"""
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import json
from dataclasses import dataclass

from .embeddings import DenseEmbedder, SparseEmbedder, HybridEmbedder
from .ollama_embeddings import OllamaEmbedder
from .chunking import SemanticChunker, Chunk
from .vectordb import VectorDatabase
from .reranking import Reranker, MaximalMarginalRelevance
from .query_processing import QueryProcessor, reciprocal_rank_fusion
from .generation import Generator

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """RAG pipeline result."""
    query: str
    answer: str
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AdvancedRAG:
    """Research-grade RAG with all optimizations."""
    
    def __init__(self, config_path: str = "rag/config/rag_config.yaml"):
        """Initialize RAG pipeline from config."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        self.embedder: Optional[HybridEmbedder] = None
        self.vectordb: Optional[VectorDatabase] = None
        self.reranker: Optional[Reranker] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.generator: Optional[Generator] = None
        self.mmr: Optional[MaximalMarginalRelevance] = None
        
        logger.info("RAG pipeline initialized")
    
    def build_index(self, text_dir: Path, output_dir: Path):
        """
        Build RAG index from text files.
        
        Args:
            text_dir: Directory with text files
            output_dir: Output directory for index
        """
        logger.info("=" * 80)
        logger.info("BUILDING RAG INDEX")
        logger.info("=" * 80)
        
        # 1. Load and chunk documents
        logger.info("\n[1/6] Loading and chunking documents...")
        chunker = SemanticChunker(**self.config['chunking'])
        
        text_files = list(text_dir.glob("*.txt"))
        logger.info(f"Found {len(text_files)} text files")
        
        for text_file in text_files:
            paper_id = text_file.stem
            text = text_file.read_text(encoding='utf-8', errors='ignore')
            
            chunks = chunker.chunk_document(text, paper_id)
            self.chunks.extend(chunks)
        
        self.chunk_texts = [chunk.text for chunk in self.chunks]
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # 2. Initialize embedders
        logger.info("\n[2/6] Initializing embedders...")
        dense_config = self.config['embeddings']['dense']
        sparse_config = self.config['embeddings']['sparse']
        
        # Use Ollama embedder
        dense = OllamaEmbedder(
            model_name=dense_config.get('model_name', 'bge-m3:latest'),
            batch_size=dense_config.get('batch_size', 32)
        )
        sparse = SparseEmbedder(**sparse_config)
        
        hybrid_config = self.config['retrieval']['hybrid']
        self.embedder = HybridEmbedder(
            dense, sparse,
            hybrid_config['dense_weight'],
            hybrid_config['sparse_weight']
        )
        
        # 3. Generate embeddings
        logger.info("\n[3/6] Generating embeddings...")
        embeddings = self.embedder.embed_documents(self.chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
        
        # 4. Fit sparse embedder
        logger.info("\n[4/6] Fitting sparse embedder...")
        self.embedder.fit_sparse(self.chunk_texts)
        
        # 5. Build vector database
        logger.info("\n[5/6] Building vector database...")
        db_config = self.config['vectordb']
        self.vectordb = VectorDatabase(
            dimension=embeddings.shape[1],
            **db_config
        )
        
        doc_ids = [f"{chunk.paper_id}::{chunk.chunk_id}" for chunk in self.chunks]
        self.vectordb.build(embeddings, doc_ids)
        
        # 6. Save index
        logger.info("\n[6/6] Saving index...")
        output_dir = Path(output_dir)
        
        self.vectordb.save(output_dir / "vectordb")
        
        # Save chunks
        chunks_data = [
            {
                'text': c.text,
                'paper_id': c.paper_id,
                'chunk_id': c.chunk_id,
                'start_char': c.start_char,
                'end_char': c.end_char,
                'metadata': c.metadata
            }
            for c in self.chunks
        ]
        with open(output_dir / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save config
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
        
        logger.info(f"\n✅ Index saved to {output_dir}")
        logger.info("=" * 80)
    
    def load_index(self, index_dir: Path):
        """Load pre-built index."""
        logger.info(f"Loading index from {index_dir}")
        
        index_dir = Path(index_dir)
        
        # Load chunks
        with open(index_dir / "chunks.json") as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            Chunk(
                text=c['text'],
                paper_id=c['paper_id'],
                chunk_id=c['chunk_id'],
                start_char=c['start_char'],
                end_char=c['end_char'],
                metadata=c['metadata']
            )
            for c in chunks_data
        ]
        self.chunk_texts = [c.text for c in self.chunks]
        
        # Load config
        with open(index_dir / "config.yaml") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize embedders
        dense_config = self.config['embeddings']['dense']
        sparse_config = self.config['embeddings']['sparse']
        
        # Use Ollama embedder
        dense = OllamaEmbedder(
            model_name=dense_config.get('model_name', 'bge-m3:latest'),
            batch_size=dense_config.get('batch_size', 32)
        )
        sparse = SparseEmbedder(**sparse_config)
        
        # Fit sparse on loaded chunks
        sparse.fit(self.chunk_texts)
        
        hybrid_config = self.config['retrieval']['hybrid']
        self.embedder = HybridEmbedder(
            dense, sparse,
            hybrid_config['dense_weight'],
            hybrid_config['sparse_weight']
        )
        
        # Load vector database
        db_config = self.config['vectordb']
        self.vectordb = VectorDatabase(
            dimension=1024,  # BGE-M3 dimension
            **db_config
        )
        self.vectordb.load(index_dir / "vectordb")
        
        # Initialize other components
        if self.config['retrieval']['reranking']['enabled']:
            rerank_config = self.config['retrieval']['reranking']
            self.reranker = Reranker(
                rerank_config['model'],
                dense_config['device'],
                rerank_config['batch_size']
            )
        
        if self.config['retrieval']['diversity']['enabled']:
            div_config = self.config['retrieval']['diversity']
            self.mmr = MaximalMarginalRelevance(div_config['lambda_param'])
        
        self.query_processor = QueryProcessor(
            self.config['generation']['model'],
            self.config['generation']['provider']
        )
        
        self.generator = Generator(
            self.config['generation']['model'],
            self.config['generation']['provider'],
            self.config['generation']['system_prompt']
        )
        
        logger.info(f"Index loaded: {len(self.chunks)} chunks")
    
    async def query(
        self,
        query: str,
        use_expansion: bool = None,
        use_rewrite: bool = None,
        use_hyde: bool = None,
        use_reranking: bool = None,
        use_diversity: bool = None
    ) -> RAGResult:
        """
        Query the RAG system.
        
        Args:
            query: User query
            use_expansion: Enable query expansion (default from config)
            use_rewrite: Enable query rewriting (default from config)
            use_hyde: Enable HyDE (default from config)
            use_reranking: Enable reranking (default from config)
            use_diversity: Enable MMR diversity (default from config)
            
        Returns:
            RAGResult with answer and contexts
        """
        if self.vectordb is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Use config defaults if not specified
        if use_expansion is None:
            use_expansion = self.config['query']['expansion']['enabled']
        if use_rewrite is None:
            use_rewrite = self.config['query']['rewrite']['enabled']
        if use_hyde is None:
            use_hyde = self.config['query']['hyde']['enabled']
        if use_reranking is None:
            use_reranking = self.config['retrieval']['reranking']['enabled']
        if use_diversity is None:
            use_diversity = self.config['retrieval']['diversity']['enabled']
        
        original_query = query
        
        # Query processing
        if use_rewrite:
            query = await self.query_processor.rewrite_query(query)
            logger.info(f"Rewritten: {query}")
        
        queries_to_search = [query]
        
        if use_expansion:
            expansions = await self.query_processor.expand_query(
                query,
                self.config['query']['expansion']['num_expansions']
            )
            queries_to_search.extend(expansions[1:])  # Skip original
            logger.info(f"Expanded to {len(queries_to_search)} queries")
        
        if use_hyde:
            hypotheticals = await self.query_processor.generate_hyde(
                query,
                self.config['query']['hyde']['num_generations']
            )
            queries_to_search.extend(hypotheticals)
            logger.info(f"Added {len(hypotheticals)} HyDE documents")
        
        # Retrieve for all queries
        all_results = []
        for q in queries_to_search:
            results = self.embedder.hybrid_search(
                q,
                self.vectordb.index.reconstruct_n(0, len(self.chunks)),
                self.config['retrieval']['top_k']
            )
            all_results.append([(self.vectordb.doc_ids[idx], score) for idx, score in results])
        
        # Fusion
        if len(all_results) > 1:
            fused = reciprocal_rank_fusion(all_results)
            top_k = self.config['retrieval']['top_k']
            retrieved = fused[:top_k]
        else:
            retrieved = all_results[0]
        
        # Get chunks
        retrieved_chunks = []
        for doc_id, score in retrieved:
            paper_id, chunk_id = doc_id.split("::")
            chunk_id = int(chunk_id)
            
            for chunk in self.chunks:
                if chunk.paper_id == paper_id and chunk.chunk_id == chunk_id:
                    retrieved_chunks.append((chunk, score))
                    break
        
        # Reranking
        if use_reranking and self.reranker:
            texts = [chunk.text for chunk, _ in retrieved_chunks]
            reranked_indices = self.reranker.rerank(
                query,
                texts,
                self.config['retrieval']['reranking']['top_k']
            )
            retrieved_chunks = [retrieved_chunks[idx] for idx, _ in reranked_indices]
        
        # Diversity (MMR)
        final_k = self.config['retrieval']['diversity']['final_top_k']
        if use_diversity and self.mmr and len(retrieved_chunks) > final_k:
            # This would need embeddings - simplified for now
            retrieved_chunks = retrieved_chunks[:final_k]
        else:
            retrieved_chunks = retrieved_chunks[:final_k]
        
        # Generate answer
        contexts = [
            {
                'text': chunk.text,
                'paper_id': chunk.paper_id,
                'score': score
            }
            for chunk, score in retrieved_chunks
        ]
        
        answer = await self.generator.generate(query, contexts)
        
        return RAGResult(
            query=original_query,
            answer=answer,
            contexts=contexts,
            metadata={
                'num_queries': len(queries_to_search),
                'num_retrieved': len(retrieved),
                'num_reranked': len(retrieved_chunks),
                'used_expansion': use_expansion,
                'used_rewrite': use_rewrite,
                'used_hyde': use_hyde,
                'used_reranking': use_reranking,
                'used_diversity': use_diversity
            }
        )
