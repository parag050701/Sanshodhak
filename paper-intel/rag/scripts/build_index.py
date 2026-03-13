"""
Build RAG index from text files.
"""
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.core.pipeline import AdvancedRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Build RAG index."""
    # Paths
    text_dir = Path("ingestion/raw_text")
    output_dir = Path("rag/index")
    
    # Initialize RAG
    rag = AdvancedRAG(config_path="rag/config/rag_config.yaml")
    
    # Build index
    rag.build_index(text_dir, output_dir)
    
    print("\n✅ RAG index built successfully!")
    print(f"   Index location: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Generate test data: python rag/evaluation/generate_test_data.py")
    print(f"  2. Run evaluation: python rag/scripts/run_evaluation.py")
    print(f"  3. Query system: python rag/scripts/query_rag.py")


if __name__ == "__main__":
    main()
