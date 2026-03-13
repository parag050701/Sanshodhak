"""
Build RAG Index from Papers
Indexes all papers from ingestion/raw_text/
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline


def main():
    """Build RAG index from papers"""
    
    print("=" * 80)
    print("🚀 BUILDING RAG INDEX")
    print("=" * 80)
    
    # Paths (relative to script location)
    base_dir = Path(__file__).parent
    text_dir = base_dir / "ingestion/raw_text"
    metadata_dir = base_dir / "ingestion/output_json"
    vector_store_path = base_dir / "rag/vector_store"
    
    # Check if text directory exists
    if not text_dir.exists():
        print(f"❌ Text directory not found: {text_dir}")
        return
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        chunk_size=512,
        chunk_overlap=128,
        use_gpu=False,  # Set to True if you have GPU
        use_reranking=True
    )
    
    # Index papers
    rag.index_papers(
        text_dir=str(text_dir),
        metadata_dir=str(metadata_dir),
        save_path=str(vector_store_path)
    )
    
    print("\n" + "=" * 80)
    print("✅ RAG INDEX BUILT SUCCESSFULLY")
    print("=" * 80)
    print(f"\n📂 Vector store saved to: {vector_store_path}")
    print(f"🔍 Ready to search!")
    print("\nNext steps:")
    print("  1. Run: python query_rag.py")
    print("  2. Or use: python demo_rag.py")


if __name__ == "__main__":
    main()
