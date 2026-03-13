"""
Interactive RAG query interface.
"""
import logging
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.core.pipeline import AdvancedRAG

logging.basicConfig(level=logging.WARNING)


async def main():
    """Interactive query interface."""
    print("=" * 80)
    print("RESEARCH-GRADE RAG SYSTEM")
    print("=" * 80)
    print()
    
    # Load RAG
    print("Loading RAG system...")
    rag = AdvancedRAG(config_path="rag/config/rag_config.yaml")
    rag.load_index(Path("rag/index"))
    print("✅ RAG system ready!")
    print()
    
    # Interactive loop
    while True:
        print("-" * 80)
        query = input("\n📝 Enter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        print("\n🔍 Searching...")
        result = await rag.query(query)
        
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(result.answer)
        
        print("\n" + "=" * 80)
        print("SOURCES:")
        print("=" * 80)
        for i, ctx in enumerate(result.contexts[:5], 1):
            print(f"\n[{i}] Paper: {ctx['paper_id']} (Score: {ctx['score']:.3f})")
            print(f"    {ctx['text'][:200]}...")
        
        print("\n" + "=" * 80)
        print("METADATA:")
        print("=" * 80)
        for k, v in result.metadata.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
