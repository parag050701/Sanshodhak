"""
Demo RAG System
Quick demo of RAG capabilities
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline


def demo_search(rag):
    """Demo search functionality"""
    print("\n" + "=" * 80)
    print("🔍 DEMO: SEMANTIC SEARCH")
    print("=" * 80)
    
    queries = [
        "machine learning for healthcare",
        "deep learning techniques",
        "data preprocessing methods",
        "model evaluation metrics"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 80)
        
        results = rag.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            paper_id = result["metadata"].get("paper_id", "Unknown")
            section = result["metadata"].get("section", "unknown")
            score = result.get("final_score", result.get("score", 0))
            
            print(f"\n{i}. Paper: {paper_id}")
            print(f"   Section: {section}")
            print(f"   Score: {score:.4f}")
            
            doc = result["document"]
            preview = doc[:150] + "..." if len(doc) > 150 else doc
            print(f"   Text: {preview}")


def demo_question_answering(rag):
    """Demo question answering"""
    print("\n" + "=" * 80)
    print("💬 DEMO: QUESTION ANSWERING")
    print("=" * 80)
    
    questions = [
        "What are the main challenges in healthcare AI?",
        "How do neural networks work?",
        "What is cross-validation?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 80)
        
        answer_data = rag.answer_question(question, top_k=3)
        
        print(f"\n📖 Context ({answer_data['num_chunks']} chunks):")
        context = answer_data['context']
        preview = context[:500] + "..." if len(context) > 500 else context
        print(preview)
        
        print("\n📚 Sources:")
        for source in answer_data['sources']:
            print(f"  - {source['paper_id']} ({source['section']}) - score: {source['score']:.4f}")


def main():
    """Run RAG demo"""
    
    vector_store_path = "rag/vector_store"
    
    # Check if vector store exists
    if not Path(vector_store_path).exists():
        print(f"❌ Vector store not found: {vector_store_path}")
        print("Please run: python build_rag_index.py")
        return
    
    print("=" * 80)
    print("🚀 RAG SYSTEM DEMO")
    print("=" * 80)
    
    # Load RAG pipeline
    print("\n📥 Loading RAG system...")
    rag = RAGPipeline(
        vector_store_path=vector_store_path,
        use_gpu=False,
        use_reranking=True
    )
    
    print("✅ RAG system loaded!")
    
    # Run demos
    demo_search(rag)
    demo_question_answering(rag)
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Run interactive mode: python query_rag.py")
    print("  - Search papers: python query_rag.py 'your query here'")


if __name__ == "__main__":
    main()
