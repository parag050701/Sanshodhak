"""Enhanced RAG with hybrid search, reranking, query expansion, and comprehensive evaluation."""
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
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available, using Ollama only")


class EnhancedRAG:
    def __init__(self, 
                 embed_model="BAAI/bge-m3",
                 rerank_model="BAAI/bge-reranker-v2-m3",
                 use_ollama_fallback=True,
                 ollama_embed="bge-m3",
                 ollama_llm="deepseek-r1:7b",
                 ollama_url="http://localhost:11434"):
        
        self.use_ollama = False
        self.ollama_embed = ollama_embed
        self.ollama_llm = ollama_llm
        self.ollama_url = ollama_url
        
        # Try sentence-transformers first, fallback to Ollama
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"📦 Loading embedding model: {embed_model}...")
                self.embedder = SentenceTransformer(embed_model)
                print(f"📦 Loading reranker: {rerank_model}...")
                self.reranker = CrossEncoder(rerank_model)
                print("✅ Models loaded successfully")
            except Exception as e:
                if use_ollama_fallback:
                    print(f"❌ Failed to load transformers: {e}")
                    print(f"🔄 Falling back to Ollama ({ollama_embed})")
                    self.use_ollama = True
                    self.embedder = None
                    self.reranker = None
                else:
                    raise
        else:
            if use_ollama_fallback:
                print(f"🦙 Using Ollama for embeddings ({ollama_embed})")
                self.use_ollama = True
                self.embedder = None
                self.reranker = None
            else:
                raise ImportError("sentence-transformers required but not available")
        
        self.index = None
        self.chunks = []
        self.metadata = []
        self.bm25_docs = None
        self.idf = None
        self.avgdl = None
        
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
        print("📊 Building BM25 index...")
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
        print(f"✅ BM25 index built: {N} docs, avgdl={self.avgdl:.1f}")
    
    def bm25_score(self, query_tokens: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score for a document."""
        doc = self.bm25_docs[doc_idx]
        score = 0.0
        doc_len = len(doc)
        
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            tf = doc.count(term)
            idf = self.idf[term]
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / self.avgdl))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def build_index(self, text_dir: str):
        """Build FAISS and BM25 indices from text files."""
        print(f"\n📁 Loading texts from {text_dir}...")
        text_files = list(Path(text_dir).glob("*.txt"))
        print(f"📄 Found {len(text_files)} text files")
        
        all_chunks = []
        all_metadata = []
        
        for text_file in text_files:
            print(f"  Processing {text_file.name}...", end='\r')
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
        
        print(f"\n📝 Total chunks: {len(all_chunks)}")
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        print("🔢 Generating embeddings...")
        embeddings = []
        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            if self.use_ollama:
                batch_embs = [self.get_embedding(chunk) for chunk in batch]
                embeddings.extend(batch_embs)
            else:
                batch_embs = self.embedder.encode(batch, convert_to_numpy=True)
                batch_embs = batch_embs / np.linalg.norm(batch_embs, axis=1, keepdims=True)
                embeddings.extend(batch_embs)
            print(f"  {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}...", end='\r')
        
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"\n✅ Embeddings shape: {embeddings.shape}")
        
        print("🔍 Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"✅ FAISS index built: {self.index.ntotal} vectors")
        
        self.build_bm25_index()
        
        return self
    
    def save(self, save_dir: str):
        """Save index and metadata."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'bm25_docs': self.bm25_docs,
                'idf': self.idf,
                'avgdl': self.avgdl,
                'use_ollama': self.use_ollama
            }, f)
        
        print(f"💾 Saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load index and metadata."""
        save_path = Path(save_dir)
        self.index = faiss.read_index(str(save_path / "faiss.index"))
        
        with open(save_path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.bm25_docs = data.get('bm25_docs')
            self.idf = data.get('idf')
            self.avgdl = data.get('avgdl')
            if 'use_ollama' in data:
                self.use_ollama = data['use_ollama']
        
        print(f"✅ Loaded from {save_dir}")
        return self
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        expansions = [query]
        
        # Simple rule-based expansion
        terms_map = {
            "machine learning": ["ML", "deep learning", "neural network"],
            "performance": ["accuracy", "efficiency", "results", "metrics"],
            "dataset": ["data", "corpus", "benchmark"],
            "model": ["architecture", "network", "system"],
            "training": ["learning", "optimization", "fine-tuning"],
        }
        
        query_lower = query.lower()
        for key, values in terms_map.items():
            if key in query_lower:
                expansions.extend(values)
        
        return " ".join(expansions)
    
    def search(self, query: str, top_k: int = 20, use_hybrid: bool = True,
               use_rerank: bool = True) -> List[Dict]:
        """Hybrid search with optional reranking."""
        
        # Expand query
        expanded_query = self.expand_query(query)
        
        # Dense retrieval
        query_emb = self.get_embedding(expanded_query).reshape(1, -1)
        dense_scores, dense_indices = self.index.search(query_emb, top_k * 2)
        
        results = {}
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            results[idx] = {'dense_score': float(score), 'bm25_score': 0.0}
        
        # BM25 retrieval (if available and hybrid enabled)
        if use_hybrid and self.bm25_docs:
            query_tokens = self.tokenize(expanded_query)
            bm25_scores = [(i, self.bm25_score(query_tokens, i)) 
                          for i in range(len(self.chunks))]
            bm25_scores.sort(key=lambda x: x[1], reverse=True)
            
            for idx, score in bm25_scores[:top_k * 2]:
                if idx in results:
                    results[idx]['bm25_score'] = score
                else:
                    results[idx] = {'dense_score': 0.0, 'bm25_score': score}
        
        # Combine scores (RRF - Reciprocal Rank Fusion)
        for idx in results:
            dense_rank = list(dense_indices[0]).index(idx) + 1 if idx in dense_indices[0] else 999
            results[idx]['rrf_score'] = 1.0 / (60 + dense_rank)
            
            if use_hybrid and self.bm25_docs:
                bm25_rank = next((i for i, (idx2, _) in enumerate(bm25_scores) if idx2 == idx), 999) + 1
                results[idx]['rrf_score'] += 1.0 / (60 + bm25_rank)
        
        # Sort by RRF score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['rrf_score'], reverse=True)[:top_k]
        
        # Rerank if available
        if use_rerank and not self.use_ollama and self.reranker:
            pairs = [(query, self.chunks[idx]) for idx, _ in sorted_results]
            rerank_scores = self.reranker.predict(pairs)
            sorted_results = [(idx, {**scores, 'rerank_score': float(rs)}) 
                            for (idx, scores), rs in zip(sorted_results, rerank_scores)]
            sorted_results.sort(key=lambda x: x[1]['rerank_score'], reverse=True)
        
        # Format results
        final_results = []
        for idx, scores in sorted_results[:top_k]:
            final_results.append({
                'score': scores.get('rerank_score', scores['rrf_score']),
                'text': self.chunks[idx],
                'metadata': self.metadata[idx],
                'scores': scores
            })
        
        return final_results
    
    def _generate_ollama(self, question: str, context: str) -> str:
        """Generate answer using Ollama."""
        prompt = f"""Based on the following research paper excerpts, answer the question concisely and accurately.

Context from papers:
{context}

Question: {question}

Answer:"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_llm,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            return f"Error: {response.text}"
        
        return response.json()["response"]
    
    def _generate_openrouter(self, question: str, context: str) -> str:
        """Generate answer using OpenRouter API."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "Error: OPENROUTER_API_KEY not set"
        
        prompt = f"""Based on the following research paper excerpts, answer the question concisely and accurately.

Context from papers:
{context}

Question: {question}

Answer:"""
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code != 200:
            return f"Error: {response.text}"
        
        return response.json()["choices"][0]["message"]["content"]
    
    def query(self, question: str, llm_api: str = "ollama", top_k: int = 5,
              verbose: bool = True, use_hybrid: bool = True, use_rerank: bool = True) -> Tuple[str, List[Dict]]:
        """Search and generate answer with enhanced retrieval."""
        if verbose:
            print(f"\n❓ Query: {question}")
            print("🔍 Searching...")
        
        results = self.search(question, top_k=top_k, use_hybrid=use_hybrid, use_rerank=use_rerank)
        
        if verbose:
            print(f"✅ Found {len(results)} results")
            for i, r in enumerate(results[:3]):
                print(f"\n[{i+1}] Score: {r['score']:.3f}")
                print(f"📄 File: {r['metadata']['file']}")
                if 'scores' in r:
                    print(f"   Scores: {r['scores']}")
                print(f"📖 Text: {r['text'][:150]}...")
        
        # Combine context with source attribution
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(f"[Source {i+1}] {r['metadata']['file']}:\n{r['text']}")
        context = "\n\n".join(context_parts)
        
        if verbose:
            print("\n💭 Generating answer...")
        
        if llm_api == "ollama":
            answer = self._generate_ollama(question, context)
        else:
            answer = self._generate_openrouter(question, context)
        
        if verbose:
            print(f"\n✨ Answer: {answer}")
        
        return answer, results


class RAGEvaluator:
    """Comprehensive RAG evaluation metrics."""
    
    def __init__(self, rag: EnhancedRAG):
        self.rag = rag
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Precision@K metric."""
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / k if k > 0 else 0.0
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Recall@K metric."""
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / len(relevant_set) if relevant_set else 0.0
    
    def mrr(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Mean Reciprocal Rank."""
        relevant_set = set(relevant_docs)
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """NDCG@K metric."""
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        # DCG
        dcg = sum((1 if doc in relevant_set else 0) / math.log2(i + 2) 
                  for i, doc in enumerate(retrieved_k))
        
        # IDCG
        ideal_k = min(k, len(relevant_docs))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval(self, test_questions: List[Dict], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate retrieval performance on test questions."""
        print(f"\n📊 Evaluating retrieval on {len(test_questions)} questions...")
        
        results = {f"P@{k}": [] for k in k_values}
        results.update({f"R@{k}": [] for k in k_values})
        results.update({f"NDCG@{k}": [] for k in k_values})
        results["MRR"] = []
        
        for i, item in enumerate(test_questions):
            question = item["question"]
            relevant_docs = item.get("relevant_docs", [])
            
            if not relevant_docs:
                continue
            
            print(f"  {i+1}/{len(test_questions)}...", end='\r')
            
            # Retrieve
            search_results = self.rag.search(question, top_k=max(k_values), use_hybrid=True, use_rerank=True)
            retrieved_docs = [r['metadata']['file'] for r in search_results]
            
            # Calculate metrics
            for k in k_values:
                results[f"P@{k}"].append(self.precision_at_k(retrieved_docs, relevant_docs, k))
                results[f"R@{k}"].append(self.recall_at_k(retrieved_docs, relevant_docs, k))
                results[f"NDCG@{k}"].append(self.ndcg_at_k(retrieved_docs, relevant_docs, k))
            
            results["MRR"].append(self.mrr(retrieved_docs, relevant_docs))
        
        # Average results
        avg_results = {metric: np.mean(values) for metric, values in results.items()}
        
        print(f"\n\n📈 Retrieval Evaluation Results:")
        for metric, value in avg_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return avg_results
    
    def evaluate_generation(self, test_questions: List[Dict], llm_api: str = "ollama") -> Dict:
        """Evaluate generation quality."""
        print(f"\n📊 Evaluating generation on {len(test_questions)} questions...")
        
        predictions = []
        references = []
        
        for i, item in enumerate(test_questions):
            question = item["question"]
            reference = item.get("answer", "")
            
            if not reference:
                continue
            
            print(f"  {i+1}/{len(test_questions)}...", end='\r')
            
            answer, _ = self.rag.query(question, llm_api=llm_api, verbose=False, top_k=5)
            predictions.append(answer)
            references.append(reference)
        
        print(f"\n\n📈 Generation Evaluation Results:")
        print(f"  Total questions: {len(predictions)}")
        print(f"  Avg answer length: {np.mean([len(p.split()) for p in predictions]):.1f} words")
        
        return {
            "total_questions": len(predictions),
            "avg_answer_length": np.mean([len(p.split()) for p in predictions])
        }
    
    def full_evaluation(self, test_questions: List[Dict], llm_api: str = "ollama") -> Dict:
        """Run full evaluation."""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE RAG EVALUATION")
        print("="*60)
        
        retrieval_results = self.evaluate_retrieval(test_questions)
        generation_results = self.evaluate_generation(test_questions, llm_api)
        
        return {
            "retrieval": retrieval_results,
            "generation": generation_results,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Build index
    print("=" * 60)
    print("🚀 BUILDING ENHANCED RAG INDEX")
    print("=" * 60)
    
    rag = EnhancedRAG()
    rag.build_index("ingestion/raw_text")
    rag.save("enhanced_rag_index")
    
    print("\n" + "=" * 60)
    print("🧪 TESTING QUERY")
    print("=" * 60)
    
    # Test query
    answer, results = rag.query(
        "What machine learning techniques are discussed in these papers?",
        use_hybrid=True,
        use_rerank=True
    )
    
    print("\n" + "=" * 60)
    print("✅ DONE - Index saved to enhanced_rag_index/")
    print("=" * 60)
