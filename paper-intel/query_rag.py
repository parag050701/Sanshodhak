"""
Query RAG System
Interactive CLI for querying papers
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline


def print_results(results, show_context=False):
    """Print search results"""
    if not results:
        print("❌ No results found")
        return
    
    print(f"\n🔍 Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        paper_id = result["metadata"].get("paper_id", "Unknown")
        section = result["metadata"].get("section", "unknown")
        score = result.get("final_score", result.get("rerank_score", result.get("score", 0)))
        
        print(f"{i}. Paper: {paper_id}")
        print(f"   Section: {section}")
        print(f"   Score: {score:.4f}")
        
        if show_context:
            doc = result["document"]
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"   Preview: {preview}")
        
        print()


def interactive_mode(rag):
    """Interactive query mode"""
    print("\n" + "=" * 80)
    print("🤖 INTERACTIVE RAG QUERY MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  - Enter a question to search")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'help' for more options")
    print()
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nCommands:")
                print("  search <query>     - Search papers")
                print("  answer <question>  - Get answer with context")
                print("  paper <paper_id>   - Get paper info")
                print("  quit/exit          - Exit")
                print()
                continue
            
            # Parse command
            if query.startswith("search "):
                query = query[7:]
                results = rag.search(query, top_k=5)
                print_results(results, show_context=True)
            
            elif query.startswith("answer "):
                question = query[7:]
                answer_data = rag.answer_question(question, top_k=5)
                
                print(f"\n📖 Context ({answer_data['num_chunks']} chunks):")
                print("-" * 80)
                print(answer_data['context'][:1000] + "..." if len(answer_data['context']) > 1000 else answer_data['context'])
                print("-" * 80)
                
                print("\n📚 Sources:")
                for source in answer_data['sources']:
                    print(f"  - {source['paper_id']} ({source['section']}) - score: {source['score']:.4f}")
                print()
            
            elif query.startswith("paper "):
                paper_id = query[6:].strip()
                results = rag.vector_store.search(
                    query_embedding=rag.embedder.encode([f"Summary of {paper_id}"])[0],
                    k=5,
                    filter_metadata={"paper_id": paper_id}
                )
                print(f"\n📄 Paper: {paper_id}")
                print_results(results, show_context=True)
            
            else:
                # Default: search
                results = rag.search(query, top_k=5)
                print_results(results, show_context=True)
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Query RAG system"""
    
    vector_store_path = "rag/vector_store"
    
    # Check if vector store exists
    if not Path(vector_store_path).exists():
        print(f"❌ Vector store not found: {vector_store_path}")
        print("Please run: python build_rag_index.py")
        return
    
    print("=" * 80)
    print("🔍 LOADING RAG SYSTEM")
    print("=" * 80)
    
    # Load RAG pipeline
    rag = RAGPipeline(
        vector_store_path=vector_store_path,
        use_gpu=False,
        use_reranking=True
    )
    
    print("\n✅ RAG system loaded!")
    
    # Check for command-line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n🔍 Query: {query}\n")
        
        results = rag.search(query, top_k=10)
        print_results(results, show_context=True)
    else:
        # Interactive mode
        interactive_mode(rag)


if __name__ == "__main__":
    main()
