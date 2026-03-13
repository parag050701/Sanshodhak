"""Test enhanced RAG with evaluation."""
from enhanced_rag import EnhancedRAG, RAGEvaluator
import json

# Load existing index (or build new one)
print("Loading RAG...")
rag = EnhancedRAG()

try:
    rag.load("enhanced_rag_index")
except:
    print("Building new index...")
    rag.build_index("ingestion/raw_text")
    rag.save("enhanced_rag_index")

# Test queries with different configurations
print("\n" + "="*80)
print("TEST 1: Dense only (no hybrid, no rerank)")
print("="*80)
answer, results = rag.query(
    "What are the main findings about neural networks?",
    use_hybrid=False,
    use_rerank=False,
    top_k=3
)

print("\n" + "="*80)
print("TEST 2: Hybrid search (dense + BM25, no rerank)")
print("="*80)
answer, results = rag.query(
    "What are the main findings about neural networks?",
    use_hybrid=True,
    use_rerank=False,
    top_k=3
)

print("\n" + "="*80)
print("TEST 3: Full pipeline (hybrid + rerank)")
print("="*80)
answer, results = rag.query(
    "What are the main findings about neural networks?",
    use_hybrid=True,
    use_rerank=True,
    top_k=3
)

# Run evaluation if test dataset exists
print("\n" + "="*80)
print("EVALUATION")
print("="*80)

try:
    with open("rag/evaluation/test_dataset.json", "r") as f:
        test_questions = json.load(f)
    
    print(f"Loaded {len(test_questions)} test questions")
    
    evaluator = RAGEvaluator(rag)
    eval_results = evaluator.full_evaluation(test_questions[:10])  # Test on first 10
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✅ Evaluation results saved to evaluation_results.json")
    
except FileNotFoundError:
    print("⚠️  Test dataset not found at rag/evaluation/test_dataset.json")
    print("   Skipping evaluation")

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)
