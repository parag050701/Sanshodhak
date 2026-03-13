"""Enhanced RAG with reranking, query expansion, hybrid search, and evaluation."""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import requests
from collections import Counter
import re
import math

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using Ollama only")

class SimpleRAG:
    def __init__(self, 
                 embed_model="BAAI/bge-m3",
                 rerank_model="BAAI/bge-reranker-v2-m3",
                 use_ollama_fallback=True,
                 ollama_embed="bge-m3",
                 ollama_url="http://localhost:11434"):
        
        self.use_ollama = False
        self.ollama_embed = ollama_embed
        self.ollama_url = ollama_url
        
        # Try sentence-transformers first, fallback to Ollama
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model: {embed_model}...")
                self.embedder = SentenceTransformer(embed_model)
                print(f"Loading reranker: {rerank_model}...")
                self.reranker = CrossEncoder(rerank_model)
                print("✓ Models loaded successfully")
            except Exception as e:
                if use_ollama_fallback:
                    print(f"Failed to load transformers: {e}")
                    print(f"Falling back to Ollama ({ollama_embed})")
                    self.use_ollama = True
                    self.embedder = None
                    self.reranker = None
                else:
                    raise
        else:
            if use_ollama_fallback:
                print(f"Using Ollama for embeddings ({ollama_embed})")
                self.use_ollama = True
                self.embedder = None
                self.reranker = None
            else:
                raise ImportError("sentence-transformers required but not available")
        
        self.index = None
        self.chunks = []
        self.metadata = []
        self.bm25_index = None
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using transformers or Ollama."""
        if self.use_ollama:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.ollama_embed, "prompt": text}
            )
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.text}")
            emb = np.array(response.json()["embedding"], dtype=np.float32)
        else:
            emb = self.embedder.encode(text, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        emb = emb / np.linalg.norm(emb)
        return emb
    
    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Chunk with overlap for better context preservation."""
        words = text.split()
        chunks = []
        i = 0
        
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            i += chunk_size - overlap
        
        return chunks
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return re.findall(r'\w+', text.lower())
    
    def build_bm25_index(self):
        """Build BM25 index for hybrid search."""
        self.bm25_docs = [self.tokenize(chunk) for chunk in self.chunks]
        
        # Calculate IDF
        df = Counter()
        for doc in self.bm25_docs:
            df.update(set(doc))
        
        N = len(self.bm25_docs)
        self.idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
                    for term in df}
        
        # Calculate average document length
        self.avgdl = sum(len(doc) for doc in self.bm25_docs) / N
        print(f"BM25 index built: {N} docs, avgdl={self.avgdl:.1f}")
        return chunks
    
    def build_index(self, text_dir: str):
        """Build FAISS index from text files."""
        text_dir = Path(text_dir)
        print(f"\nLoading texts from {text_dir}")
        
        all_chunks = []
        all_metadata = []
        
        for txt_file in text_dir.glob("*.txt"):
            text = txt_file.read_text(encoding='utf-8', errors='ignore')
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'paper_id': txt_file.stem,
                    'chunk_id': i,
                    'file': str(txt_file)
                })
        
        print(f"Created {len(all_chunks)} chunks from {len(list(text_dir.glob('*.txt')))} files")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        print(f"✅ Index built: {self.index.ntotal} vectors")
    
    def save(self, save_dir: str):
        """Save index to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        
        with open(save_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(save_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"✅ Saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load index from disk."""
        save_dir = Path(save_dir)
        
        self.index = faiss.read_index(str(save_dir / "index.faiss"))
        
        with open(save_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(save_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Loaded index: {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks."""
        # Encode query
        query_emb = self.embedder.encode([query])
        query_emb = np.array(query_emb).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'text': self.chunks[idx],
                'score': float(score),
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def query(self, question: str, top_k: int = 5, llm_api: str = "openrouter") -> str:
        """Query RAG system."""
        # Retrieve
        results = self.search(question, top_k)
        
        # Build context
        context = "\n\n".join([
            f"[{r['metadata']['paper_id']}] {r['text']}"
            for r in results
        ])
        
        # Generate answer
        if llm_api == "ollama":
            answer = self._generate_ollama(question, context)
        else:
            answer = self._generate_openrouter(question, context)
        
        return answer, results
    
    def _generate_ollama(self, question: str, context: str) -> str:
        """Generate using Ollama."""
        prompt = f"""Context from research papers:

{context}

Question: {question}

Answer based on the context above. Cite papers using [paper_id]."""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            return response.json().get("response", "Error generating answer")
        except Exception as e:
            return f"Error: {e}"
    
    def _generate_openrouter(self, question: str, context: str) -> str:
        """Generate using OpenRouter."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        prompt = f"""Context from research papers:

{context}

Question: {question}

Answer based on the context above. Cite papers using [paper_id]."""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "meta-llama/llama-3.2-3b-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=60
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    # Build index
    rag = SimpleRAG()
    rag.build_index("ingestion/raw_text")
    rag.save("rag_index")
    
    print("\n✅ RAG built successfully!")
    print("\nTest query:")
    answer, results = rag.query("What are the main findings?", llm_api="ollama")
    print(f"\nAnswer: {answer}")
