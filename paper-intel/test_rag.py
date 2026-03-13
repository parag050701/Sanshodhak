"""Test the RAG with various queries."""
from ollama_rag import OllamaRAG

# Load the built index
rag = OllamaRAG()
rag.load("rag_index")

# Test queries
queries = [
    "What machine learning techniques are discussed in these papers?",
    "What are the main research findings?",
    "What datasets were used in the experiments?",
]

for q in queries:
    print("\n" + "=" * 80)
    answer, results = rag.query(q, top_k=3)
    print("=" * 80)
