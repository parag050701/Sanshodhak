"""
Comprehensive RAG evaluation with ROUGE, BLEU, BERTScore, and retrieval metrics.
Designed for Graph RAG systems with proper ground truth alignment.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import re

# Import evaluation metrics (with fallbacks)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge-score not available")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  NLTK not available")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️  bert-score not available (PyTorch issue)")

import math


class ComprehensiveRAGEvaluator:
    """Complete evaluation suite for RAG systems."""
    
    def __init__(self, rag):
        self.rag = rag
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        
        # BLEU smoothing
        if BLEU_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if not ROUGE_AVAILABLE:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    
    def compute_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score."""
        if not BLEU_AVAILABLE:
            return 0.0
        
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        
        try:
            score = sentence_bleu([ref_tokens], pred_tokens, 
                                 smoothing_function=self.smoothing)
            return score
        except:
            return 0.0
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore (batch)."""
        if not BERTSCORE_AVAILABLE or len(predictions) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()
            }
        except Exception as e:
            print(f"⚠️  BERTScore failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Precision@K metric."""
        if k == 0 or not relevant_docs:
            return 0.0
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / k
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Recall@K metric."""
        if not relevant_docs:
            return 0.0
        retrieved_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / len(relevant_set)
    
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
    
    def evaluate_retrieval(self, test_questions: List[Dict], 
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate retrieval performance."""
        print(f"\n📊 RETRIEVAL EVALUATION on {len(test_questions)} questions")
        print("="*80)
        
        results = {f"P@{k}": [] for k in k_values}
        results.update({f"R@{k}": [] for k in k_values})
        results.update({f"NDCG@{k}": [] for k in k_values})
        results["MRR"] = []
        
        for i, item in enumerate(test_questions):
            question = item["question"]
            relevant_docs = item.get("relevant_docs", [])
            
            if not relevant_docs:
                print(f"  ⚠️  Question {i+1}: No relevant docs specified")
                continue
            
            print(f"  [{i+1}/{len(test_questions)}] {question[:60]}...", end='\r')
            
            # Retrieve
            try:
                # Try enhanced RAG first, fallback to simple API
                try:
                    search_results = self.rag.search(question, top_k=max(k_values), 
                                                    use_hybrid=True, use_rerank=False)
                except TypeError:
                    # OllamaRAG or simple API
                    search_results = self.rag.search(question, top_k=max(k_values))
                
                retrieved_docs = [r['metadata']['file'] for r in search_results]
                
                # Calculate metrics
                for k in k_values:
                    results[f"P@{k}"].append(self.precision_at_k(retrieved_docs, relevant_docs, k))
                    results[f"R@{k}"].append(self.recall_at_k(retrieved_docs, relevant_docs, k))
                    results[f"NDCG@{k}"].append(self.ndcg_at_k(retrieved_docs, relevant_docs, k))
                
                results["MRR"].append(self.mrr(retrieved_docs, relevant_docs))
                
            except Exception as e:
                print(f"\n  ❌ Error on question {i+1}: {e}")
                continue
        
        # Average results
        avg_results = {metric: np.mean(values) if values else 0.0 
                      for metric, values in results.items()}
        
        print(f"\n\n{'='*80}")
        print("📈 RETRIEVAL RESULTS:")
        print(f"{'='*80}")
        for metric, value in avg_results.items():
            print(f"  {metric:15s}: {value:.4f}")
        
        return avg_results
    
    def evaluate_generation(self, test_questions: List[Dict], 
                           llm_api: str = "ollama",
                           sample_size: int = None) -> Dict:
        """Evaluate generation quality with ROUGE, BLEU, BERTScore."""
        
        if sample_size:
            test_questions = test_questions[:sample_size]
        
        print(f"\n📊 GENERATION EVALUATION on {len(test_questions)} questions")
        print("="*80)
        
        predictions = []
        references = []
        rouge_scores = []
        bleu_scores = []
        
        for i, item in enumerate(test_questions):
            question = item["question"]
            reference = item.get("answer", "")
            
            if not reference:
                print(f"  ⚠️  Question {i+1}: No reference answer")
                continue
            
            print(f"  [{i+1}/{len(test_questions)}] Generating...", end='\r')
            
            try:
                # Try enhanced RAG first, fallback to simple API
                try:
                    answer, _ = self.rag.query(question, llm_api=llm_api, 
                                              verbose=False, top_k=5)
                except TypeError:
                    # OllamaRAG or simple API
                    answer, _ = self.rag.query(question, verbose=False, top_k=5)
                
                predictions.append(answer)
                references.append(reference)
                
                # ROUGE
                rouge = self.compute_rouge(answer, reference)
                rouge_scores.append(rouge)
                
                # BLEU
                bleu = self.compute_bleu(answer, reference)
                bleu_scores.append(bleu)
                
            except Exception as e:
                print(f"\n  ❌ Error on question {i+1}: {e}")
                continue
        
        print(f"\n\n{'='*80}")
        print("📈 GENERATION RESULTS:")
        print(f"{'='*80}")
        
        results = {
            "total_questions": len(predictions),
            "avg_answer_length": np.mean([len(p.split()) for p in predictions]) if predictions else 0
        }
        
        # Average ROUGE
        if rouge_scores:
            results["rouge1"] = np.mean([s["rouge1"] for s in rouge_scores])
            results["rouge2"] = np.mean([s["rouge2"] for s in rouge_scores])
            results["rougeL"] = np.mean([s["rougeL"] for s in rouge_scores])
            print(f"  ROUGE-1        : {results['rouge1']:.4f}")
            print(f"  ROUGE-2        : {results['rouge2']:.4f}")
            print(f"  ROUGE-L        : {results['rougeL']:.4f}")
        
        # Average BLEU
        if bleu_scores:
            results["bleu"] = np.mean(bleu_scores)
            print(f"  BLEU           : {results['bleu']:.4f}")
        
        # BERTScore (batch)
        if predictions and references:
            bert_scores = self.compute_bertscore(predictions, references)
            results["bertscore_precision"] = bert_scores["precision"]
            results["bertscore_recall"] = bert_scores["recall"]
            results["bertscore_f1"] = bert_scores["f1"]
            print(f"  BERTScore P    : {bert_scores['precision']:.4f}")
            print(f"  BERTScore R    : {bert_scores['recall']:.4f}")
            print(f"  BERTScore F1   : {bert_scores['f1']:.4f}")
        
        print(f"  Avg Length     : {results['avg_answer_length']:.1f} words")
        
        return results
    
    def full_evaluation(self, test_questions: List[Dict], 
                       llm_api: str = "ollama",
                       sample_size: int = None) -> Dict:
        """Run complete evaluation suite."""
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE RAG EVALUATION")
        print("="*80)
        
        retrieval_results = self.evaluate_retrieval(test_questions)
        generation_results = self.evaluate_generation(test_questions, llm_api, sample_size)
        
        return {
            "retrieval": retrieval_results,
            "generation": generation_results,
            "metrics_available": {
                "rouge": ROUGE_AVAILABLE,
                "bleu": BLEU_AVAILABLE,
                "bertscore": BERTSCORE_AVAILABLE
            }
        }
    
    def create_demo_answers(self, test_questions: List[Dict], 
                           output_file: str = "demo_answers.json") -> None:
        """Generate demo answers for presentation."""
        print(f"\n📝 GENERATING DEMO ANSWERS")
        print("="*80)
        
        demo_results = []
        
        for i, item in enumerate(test_questions[:5]):  # First 5 questions
            question = item["question"]
            reference = item.get("answer", "")
            
            print(f"\n[{i+1}/5] {question[:70]}...")
            
            try:
                # Try enhanced RAG first, fallback to simple API
                try:
                    answer, search_results = self.rag.query(
                        question, 
                        llm_api="ollama",
                        verbose=False,
                        use_hybrid=True,
                        use_rerank=True,
                        top_k=3
                    )
                except TypeError:
                    # OllamaRAG or simple API
                    answer, search_results = self.rag.query(
                        question,
                        verbose=False,
                        top_k=3
                    )
                
                demo_results.append({
                    "question": question,
                    "reference_answer": reference,
                    "generated_answer": answer,
                    "top_sources": [
                        {
                            "file": r['metadata']['file'],
                            "score": r['score'],
                            "snippet": r['text'][:200]
                        }
                        for r in search_results
                    ],
                    "rouge_scores": self.compute_rouge(answer, reference),
                    "bleu_score": self.compute_bleu(answer, reference)
                })
                
                print(f"  ✅ Generated ({len(answer.split())} words)")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                demo_results.append({
                    "question": question,
                    "error": str(e)
                })
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"\n✅ Demo answers saved to {output_file}")


if __name__ == "__main__":
    from enhanced_rag import EnhancedRAG
    
    print("="*80)
    print("🚀 COMPREHENSIVE RAG EVALUATION")
    print("="*80)
    
    # Load RAG
    print("\n📦 Loading RAG system...")
    rag = EnhancedRAG()
    
    try:
        rag.load("enhanced_rag_index")
    except:
        print("Building index...")
        rag.build_index("ingestion/raw_text")
        rag.save("enhanced_rag_index")
    
    # Load test questions (use the proper ones)
    print("\n📄 Loading test questions...")
    with open("rag_test_questions.json") as f:
        test_questions = json.load(f)
    
    print(f"✅ Loaded {len(test_questions)} test questions")
    
    # Create evaluator
    evaluator = ComprehensiveRAGEvaluator(rag)
    
    # Run evaluation
    results = evaluator.full_evaluation(test_questions, sample_size=10)
    
    # Save results
    with open("comprehensive_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to comprehensive_eval_results.json")
    
    # Generate demo answers for presentation
    evaluator.create_demo_answers(test_questions)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
