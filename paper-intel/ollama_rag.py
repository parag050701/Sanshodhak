"""Ultra-simple RAG using Ollama with OpenRouter fallback."""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
import requests

class OllamaRAG:
    def __init__(self, embed_model="bge-m3", llm_model="qwen3:4b", use_openrouter=True):
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.use_openrouter = use_openrouter
        self.ollama_url = "http://localhost:11434"
        
        # Get OpenRouter key from environment or use default
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if use_openrouter and self.openrouter_key:
            print(f"✅ OpenRouter PRIMARY: meta-llama/llama-3.2-3b-instruct:free")
            print(f"🔄 Ollama FALLBACK: {llm_model}")
        else:
            print(f"Using Ollama with embed={embed_model}, llm={llm_model}")
        
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using sentence-transformers (bge-m3) or fallback to Ollama."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            if not hasattr(self, '_bge_model'):
                print("[INFO] Loading bge-m3 embedding model (sentence-transformers)...")
                self._bge_model = SentenceTransformer("BAAI/bge-m3")
            emb = self._bge_model.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            print(f"[WARN] sentence-transformers embedding failed: {e}. Falling back to Ollama API.")
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text}
            )
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")
            embedding = np.array(response.json()["embedding"], dtype=np.float32)
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple chunking by characters."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        
        if current:
            chunks.append(" ".join(current))
        return chunks
    
    def build_index(self, text_dir: str):
        """Build FAISS index from text files."""
        print(f"Loading texts from {text_dir}...")
        text_files = list(Path(text_dir).glob("*.txt"))
        print(f"Found {len(text_files)} text files")
        
        all_chunks = []
        all_metadata = []
        
        for text_file in text_files:
            print(f"Processing {text_file.name}...")
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'file': text_file.name,
                    'chunk_id': i,
                    'text': chunk
                })
        
        print(f"Total chunks: {len(all_chunks)}")
        print("Generating embeddings...")
        
        embeddings = []
        valid_chunks = []
        valid_metadata = []
        for i, chunk in enumerate(all_chunks):
            if i % 10 == 0:
                print(f"  {i}/{len(all_chunks)}...", end='\r')
            # Skip empty, very short, or very long chunks
            # Skip empty, too short, or all-whitespace
            if not chunk or len(chunk.strip()) < 20:
                print(f"\n[WARN] Skipping chunk {i} (empty or too short)")
                continue
            # Skip all-whitespace
            if chunk.strip() == '':
                print(f"\n[WARN] Skipping chunk {i} (all whitespace)")
                continue
            # Skip highly repetitive chunks (e.g., >80% one char)
            from collections import Counter
            chars = [c for c in chunk if not c.isspace()]
            if chars:
                most_common = Counter(chars).most_common(1)[0][1]
                if most_common / len(chars) > 0.8:
                    print(f"\n[WARN] Skipping chunk {i} (highly repetitive content)")
                    continue
            if len(chunk) > 4000:
                print(f"\n[WARN] Skipping chunk {i} (too long: {len(chunk)} chars)")
                continue
            try:
                emb = self.get_embedding(chunk)
            except Exception as e:
                print(f"\n[WARN] Embedding error for chunk {i}: {e} (retrying once)")
                # Retry once after short delay
                import time
                time.sleep(1)
                try:
                    emb = self.get_embedding(chunk)
                except Exception as e2:
                    print(f"\n[WARN] Skipping chunk {i} after retry: {e2}")
                    continue
            # Check for NaN or inf in embedding
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                print(f"\n[WARN] Skipping chunk {i} (NaN/Inf in embedding)")
                continue
            embeddings.append(emb)
            valid_chunks.append(chunk)
            valid_metadata.append(all_metadata[i])
        if not embeddings:
            raise RuntimeError("No valid embeddings generated! Check input data.")
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print("Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        self.index.add(embeddings)
        self.chunks = valid_chunks
        self.metadata = valid_metadata
        print(f"Index built with {self.index.ntotal} vectors")
        return self
    
    def save(self, save_dir: str):
        """Save index and metadata."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        print(f"Saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load index and metadata."""
        save_path = Path(save_dir)
        self.index = faiss.read_index(str(save_path / "faiss.index"))
        
        with open(save_path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
        
        print(f"Loaded from {save_dir}")
        return self
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks."""
        query_emb = self.get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'score': float(dist),
                'text': self.chunks[idx],
                'metadata': self.metadata[idx]
            })
        return results
    
    def generate(self, question: str, context: str) -> str:
        """Generate answer using OpenRouter or Ollama LLM."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Try OpenRouter first if enabled and key available
        if self.use_openrouter and self.openrouter_key:
            try:
                print("🌐 Trying OpenRouter API (meta-llama/llama-3.2-3b-instruct:free)...", end=" ")
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/sanshodhak",
                        "X-Title": "Sanshodhak Research Assistant"
                    },
                    json={
                        "model": "meta-llama/llama-3.2-3b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    print("✅ Success!")
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"❌ Failed (Status {response.status_code})")
                    print(f"   Error: {response.text[:200]}")
            except Exception as e:
                print(f"❌ Error: {str(e)[:100]}")
        
        # Fallback to NVIDIA OpenAI-compatible API
        print(f"🔄 Falling back to NVIDIA OpenAI-compatible API (meta/llama3-8b-instruct)...")
        try:
            from openai import OpenAI
        except ImportError:
            return "Error: openai package not installed. Please install with 'pip install openai'"
        nv_api_key = os.getenv("NVIDIA_API_KEY", "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")
        client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = nv_api_key
        )
        try:
            completion = client.chat.completions.create(
                model="meta/llama3-8b-instruct",
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
                stream=False
            )
            # Non-streaming: get the full answer
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: NVIDIA API call failed: {e}"
    
    def query(self, question: str, top_k: int = 5, verbose: bool = True) -> Tuple[str, List[Dict]]:
        """Search and generate answer."""
        if verbose:
            print(f"\nQuery: {question}")
            print("Searching...")
        
        results = self.search(question, top_k)
        
        if verbose:
            print(f"Found {len(results)} results")
            for i, r in enumerate(results[:3]):
                print(f"\n[{i+1}] Score: {r['score']:.3f}")
                print(f"File: {r['metadata']['file']}")
                print(f"Text: {r['text'][:200]}...")
        
        # Combine context
        context = "\n\n".join([r['text'] for r in results])
        
        if verbose:
            print("\nGenerating answer...")
        
        answer = self.generate(question, context)
        
        if verbose:
            print(f"\nAnswer: {answer}")
        
        return answer, results


if __name__ == "__main__":
    # Build index
    print("=" * 60)
    print("BUILDING RAG INDEX")
    print("=" * 60)
    
    rag = OllamaRAG()
    rag.build_index("ingestion/raw_text")
    rag.save("rag_index")
    
    print("\n" + "=" * 60)
    print("TESTING QUERY")
    print("=" * 60)
    
    # Test query
    answer, results = rag.query(
        "What are the main findings about quantum computing in these papers?"
    )
    
    print("\n" + "=" * 60)
    print("DONE - Index saved to rag_index/")
    print("=" * 60)
