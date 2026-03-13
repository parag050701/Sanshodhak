"""
Simple RAG with Ollama embeddings and FAISS.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAG:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.dimension = 1024  # BGE-M3 dimension
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple chunking by character count."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            
            if current_len >= chunk_size:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama."""
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': 'bge-m3', 'prompt': text}
        )
        return np.array(response.json()['embedding'], dtype=np.float32)
    
    def build_index(self, text_dir: Path, output_dir: Path):
        """Build FAISS index."""
        logger.info("=== Building Simple RAG ===")
        
        # 1. Load and chunk
        logger.info("Loading text files...")
        text_files = list(text_dir.glob("*.txt"))
        logger.info(f"Found {len(text_files)} files")
        
        for text_file in text_files:
            paper_id = text_file.stem
            text = text_file.read_text(encoding='utf-8', errors='ignore')
            
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                self.chunks.append({
                    'text': chunk,
                    'paper_id': paper_id,
                    'chunk_id': i
                })
        
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # 2. Generate embeddings
        logger.info("Generating embeddings with Ollama...")
        embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            if i % 100 == 0:
                logger.info(f"  Progress: {i}/{len(self.chunks)}")
            
            emb = self.get_ollama_embedding(chunk['text'])
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # 3. Build FAISS index
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
        # 4. Save
        output_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        
        with open(output_dir / "chunks.json", 'w') as f:
            json.dump(self.chunks, f)
        
        logger.info(f"✅ Saved to {output_dir}")
    
    def load_index(self, index_dir: Path):
        """Load FAISS index."""
        logger.info(f"Loading index from {index_dir}")
        
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        
        with open(index_dir / "chunks.json") as f:
            self.chunks = json.load(f)
        
        logger.info(f"Loaded {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks."""
        # Get query embedding
        query_emb = self.get_ollama_embedding(query).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(score)
            })
        
        return results
    
    def query(self, question: str, llm_api_key: str = None) -> Dict:
        """Full RAG query."""
        # Retrieve
        results = self.search(question, top_k=5)
        
        # Build context
        context = "\n\n".join([
            f"[Paper: {r['chunk']['paper_id']}]\n{r['chunk']['text']}"
            for r in results
        ])
        
        # Generate answer with OpenRouter
        if llm_api_key:
            import httpx
            
            prompt = f"""Context from research papers:

{context}

Question: {question}

Answer based on the context above. Cite papers using [Paper: ID] format.

Answer:"""
            
            try:
                response = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {llm_api_key}"},
                    json={
                        "model": "meta-llama/llama-3.2-3b-instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                else:
                    answer = "Error generating answer"
            except Exception as e:
                answer = f"Error: {e}"
        else:
            answer = "No LLM API key provided. Retrieved contexts only."
        
        return {
            'question': question,
            'answer': answer,
            'contexts': results
        }


if __name__ == "__main__":
    import sys
    
    rag = SimpleRAG()
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # Build index
        rag.build_index(
            Path("ingestion/raw_text"),
            Path("rag/simple_index")
        )
    else:
        # Query
        rag.load_index(Path("rag/simple_index"))
        
        # Interactive
        print("Simple RAG loaded. Ask questions (or 'quit' to exit):")
        
        while True:
            q = input("\nQuestion: ").strip()
            if q.lower() in ['quit', 'exit']:
                break
            
            import os
            api_key = os.getenv("OPENROUTER_API_KEY")
            
            result = rag.query(q, api_key)
            
            print(f"\nAnswer: {result['answer']}\n")
            print("Sources:")
            for ctx in result['contexts'][:3]:
                print(f"  - {ctx['chunk']['paper_id']} (score: {ctx['score']:.3f})")
