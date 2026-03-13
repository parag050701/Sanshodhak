"""Quick comprehensive eval using working Ollama RAG."""
import json
import numpy as np
from ollama_rag import OllamaRAG
from comprehensive_eval import ComprehensiveRAGEvaluator

print("="*80)
print("🚀 COMPREHENSIVE RAG EVALUATION (Ollama)")
print("="*80)

# Load Ollama RAG (we know this works)
print("\n📦 Loading Ollama RAG...")
rag = OllamaRAG()
rag.load("rag_index")

# Load proper test questions
print("\n📄 Loading test questions...")
with open("/home/admin-/Desktop/Sanshodhak/paper-intel/rag_test_questions.json") as f:
    test_questions = json.load(f)

print(f"✅ Loaded {len(test_questions)} test questions")

# Create evaluator
evaluator = ComprehensiveRAGEvaluator(rag)

# Run evaluation (on first 10 questions)
print("\n" + "="*80)
results = evaluator.full_evaluation(test_questions, sample_size=10)

# Save results
with open("/home/admin-/Desktop/Sanshodhak/paper-intel/comprehensive_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to comprehensive_eval_results.json")

# Generate demo answers
evaluator.create_demo_answers(test_questions)

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
