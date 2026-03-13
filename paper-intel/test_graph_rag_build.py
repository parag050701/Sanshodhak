import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

from graph_rag import GraphRAG
from core.ollama_rag import OllamaRAG

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--data_dir", type=str, default="paper-intel/research_papers/existing_root")
    parser.add_argument("--query", type=str, default="What are the key advantages of hybrid retrieval in RAG systems?")
    args = parser.parse_args()

    # Build dense vector FAISS index and BM25 baseline
    base_rag = OllamaRAG()
    
    if args.build:
        print(f"Building index from {args.data_dir}...")
        base_rag.build_index(args.data_dir)
        base_rag.save("rag_index")
    else:
        base_rag.load("rag_index")
        
    # Test Graph RAG queries using base index
    rag = GraphRAG(use_bm25=True, similarity_threshold=0.10)
    rag.index = base_rag.index
    rag.documents = base_rag.documents
    
    answer, sources = rag.query(
        args.query,
        top_k=5,
        verbose=True,
    )
    print(f"\nAnswer:\n{answer}")
