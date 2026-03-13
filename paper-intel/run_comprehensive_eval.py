#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation System
Includes retrieval metrics, generation quality, and demo answers
"""

import json
import time
try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None
from typing import Dict, List, Any
from pathlib import Path
import traceback

# Import RAG systems
try:
    from ollama_rag import OllamaRAG
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

try:
    from enhanced_rag import EnhancedRAG
    ENHANCED_AVAILABLE = True
except:
    ENHANCED_AVAILABLE = False


class ComprehensiveEvaluator:
    """Evaluate RAG systems with multiple metrics."""
    
    def __init__(self, rag_system, test_questions_file: str):
        self.rag = rag_system
        
        # Load test questions
        with open(test_questions_file, 'r') as f:
            self.test_questions = json.load(f)
        
        print(f"✅ Loaded {len(self.test_questions)} test questions")
        
        # Try to import evaluation libraries
        self.rouge_available = False
        self.bleu_available = False
        self.bertscore_available = False
        
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.rouge_available = True
            print("✅ ROUGE available")
        except:
            print("⚠️  ROUGE not available")
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction()
            self.bleu_available = True
            print("✅ BLEU available")
        except:
            print("⚠️  BLEU not available")
        
        try:
            from bert_score import score as bertscore
            self.bertscore = bertscore
            self.bertscore_available = False  # Disable by default (too slow)
            print("⚠️  BERTScore available but disabled (too slow)")
        except:
            print("⚠️  BERTScore not available")
    
    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if not self.rouge_available:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_bleu(self, prediction: str, reference: str) -> float:
        """Compute BLEU score."""
        if not self.bleu_available:
            return 0.0
        
        try:
            reference_tokens = [reference.lower().split()]
            prediction_tokens = prediction.lower().split()
            return self.sentence_bleu(reference_tokens, prediction_tokens, 
                                     smoothing_function=self.smoothing.method1)
        except:
            return 0.0
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore (batch)."""
        if not self.bertscore_available:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        try:
            P, R, F1 = self.bertscore(predictions, references, lang='en', verbose=False)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def evaluate_retrieval(self, k_values=[1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate retrieval quality with P@K, R@K, NDCG, MRR."""
        
        print("\n" + "="*80)
        print("📊 RETRIEVAL EVALUATION")
        print("="*80)
        
        results = {k: {'precision': [], 'recall': []} for k in k_values}
        mrr_scores = []
        ndcg_scores = {k: [] for k in k_values}
        
        for i, question in enumerate(self.test_questions, 1):
            query = question['question']
            relevant_docs = set(question.get('relevant_docs', []))
            
            if not relevant_docs:
                continue
            
            print(f"\n[{i}/{len(self.test_questions)}] {query[:60]}...")
            
            try:
                # Search with different methods
                try:
                    search_results = self.rag.search(query, top_k=max(k_values))
                except TypeError:
                    # Fallback for simpler API
                    search_results = self.rag.search(query)
                
                retrieved_docs = []
                for result in search_results:
                    # Handle different result formats
                    if isinstance(result, dict):
                        doc = result.get('file') or result.get('metadata', {}).get('file', '')
                    else:
                        doc = str(result)
                    retrieved_docs.append(doc)
                
                print(f"  Retrieved: {len(retrieved_docs)} docs")
                print(f"  Relevant: {len(relevant_docs)} docs")
                
                # Calculate metrics at different K values
                for k in k_values:
                    retrieved_at_k = set(retrieved_docs[:k])
                    
                    # Precision@K
                    true_positives = len(retrieved_at_k & relevant_docs)
                    precision = true_positives / k if k > 0 else 0
                    results[k]['precision'].append(precision)
                    
                    # Recall@K
                    recall = true_positives / len(relevant_docs) if relevant_docs else 0
                    results[k]['recall'].append(recall)
                    
                    # NDCG@K
                    dcg = sum([(1 if doc in relevant_docs else 0) / (j+1) 
                              for j, doc in enumerate(retrieved_docs[:k])])
                    idcg = sum([1/(j+1) for j in range(min(k, len(relevant_docs)))])
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcg_scores[k].append(ndcg)
                
                # MRR (Mean Reciprocal Rank)
                for j, doc in enumerate(retrieved_docs, 1):
                    if doc in relevant_docs:
                        mrr_scores.append(1.0 / j)
                        break
                else:
                    mrr_scores.append(0.0)
                
                print(f"  P@1={results[1]['precision'][-1]:.3f}, "
                      f"R@5={results[5]['recall'][-1]:.3f}, "
                      f"MRR={mrr_scores[-1]:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                traceback.print_exc()
                continue
        
        # Compute averages
        metrics = {}
        for k in k_values:
            if results[k]['precision']:
                metrics[f'P@{k}'] = sum(results[k]['precision']) / len(results[k]['precision'])
                metrics[f'R@{k}'] = sum(results[k]['recall']) / len(results[k]['recall'])
                metrics[f'NDCG@{k}'] = sum(ndcg_scores[k]) / len(ndcg_scores[k]) if ndcg_scores[k] else 0
            else:
                metrics[f'P@{k}'] = 0.0
                metrics[f'R@{k}'] = 0.0
                metrics[f'NDCG@{k}'] = 0.0
        
        metrics['MRR'] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        
        # Print summary
        print("\n" + "-"*80)
        print("📈 RETRIEVAL METRICS SUMMARY")
        print("-"*80)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return {
            'metrics': metrics,
            'details': {
                'total_questions': len(self.test_questions),
                'evaluated_questions': len(mrr_scores),
                'k_values': k_values
            }
        }
    
    def evaluate_generation(self, max_questions: int = 20) -> Dict[str, Any]:
        """Evaluate generation quality with ROUGE, BLEU."""
        
        print("\n" + "="*80)
        print("📝 GENERATION EVALUATION")
        print("="*80)
        
        results = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bleu': []
        }
        
        generated_answers = []
        reference_answers = []
        
        for i, question in enumerate(self.test_questions[:max_questions], 1):
            query = question['question']
            reference = question.get('answer', '')
            
            if not reference:
                continue
            
            print(f"\n[{i}/{min(max_questions, len(self.test_questions))}] {query[:60]}...")
            
            try:
                # Generate answer
                try:
                    answer = self.rag.query(query)
                except TypeError:
                    # Fallback for different API
                    try:
                        answer = self.rag.query(query, llm_api="ollama")
                    except:
                        search_results = self.rag.search(query, top_k=3)
                        answer = " ".join([r.get('text', '')[:200] for r in search_results])
                
                print(f"  Generated: {answer[:100]}...")
                print(f"  Reference: {reference[:100]}...")
                
                # Compute metrics
                rouge_scores = self.compute_rouge(answer, reference)
                bleu_score = self.compute_bleu(answer, reference)
                
                results['rouge1'].append(rouge_scores['rouge1'])
                results['rouge2'].append(rouge_scores['rouge2'])
                results['rougeL'].append(rouge_scores['rougeL'])
                results['bleu'].append(bleu_score)
                
                generated_answers.append(answer)
                reference_answers.append(reference)
                
                print(f"  ROUGE-1={rouge_scores['rouge1']:.3f}, "
                      f"ROUGE-L={rouge_scores['rougeL']:.3f}, "
                      f"BLEU={bleu_score:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                traceback.print_exc()
                continue
        
        # Compute averages
        metrics = {
            'rouge1': sum(results['rouge1']) / len(results['rouge1']) if results['rouge1'] else 0,
            'rouge2': sum(results['rouge2']) / len(results['rouge2']) if results['rouge2'] else 0,
            'rougeL': sum(results['rougeL']) / len(results['rougeL']) if results['rougeL'] else 0,
            'bleu': sum(results['bleu']) / len(results['bleu']) if results['bleu'] else 0
        }
        
        # BERTScore (optional, on subset)
        if self.bertscore_available and generated_answers:
            print("\n  Computing BERTScore...")
            bertscore_metrics = self.compute_bertscore(
                generated_answers[:5], 
                reference_answers[:5]
            )
            metrics.update({f'bertscore_{k}': v for k, v in bertscore_metrics.items()})
        
        # Print summary
        print("\n" + "-"*80)
        print("📈 GENERATION METRICS SUMMARY")
        print("-"*80)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return {
            'metrics': metrics,
            'details': {
                'total_questions': len(self.test_questions),
                'evaluated_questions': len(generated_answers)
            }
        }
    
    def generate_demo_answers(self, output_file: str, num_demos: int = 5):
        """Generate demo answers for presentation."""
        
        print("\n" + "="*80)
        print("🎬 GENERATING DEMO ANSWERS")
        print("="*80)
        
        demos = []
        
        for i, question in enumerate(self.test_questions[:num_demos], 1):
            query = question['question']
            reference = question.get('answer', '')
            
            print(f"\n[{i}/{num_demos}] {query}")
            
            try:
                # Get search results
                try:
                    search_results = self.rag.search(query, top_k=5)
                except TypeError:
                    search_results = self.rag.search(query)
                
                # Generate answer
                try:
                    answer = self.rag.query(query)
                except:
                    answer = "Answer generation not available for this RAG system."
                
                demos.append({
                    'question': query,
                    'reference_answer': reference,
                    'generated_answer': answer,
                    'retrieved_docs': [
                        {
                            'file': r.get('file', 'unknown'),
                            'score': r.get('score', 0),
                            'snippet': r.get('text', '')[:300]
                        }
                        for r in search_results[:3]
                    ],
                    'difficulty': question.get('difficulty', 'unknown'),
                    'type': question.get('type', 'unknown')
                })
                
                print(f"  ✅ Generated answer: {answer[:100]}...")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                demos.append({
                    'question': query,
                    'error': str(e)
                })
        
        # Save demos
        with open(output_file, 'w') as f:
            json.dump(demos, f, indent=2)
        
        print(f"\n💾 Saved {len(demos)} demos to {output_file}")
        
        return demos
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE RAG EVALUATION")
        print("="*80)
        print(f"RAG System: {type(self.rag).__name__}")
        print(f"Test Questions: {len(self.test_questions)}")
        
        start_time = time.time()
        
        # Run evaluations
        retrieval_results = self.evaluate_retrieval()
        generation_results = self.evaluate_generation(max_questions=15)
        
        # Generate demos
        demos = self.generate_demo_answers('demo_answers.json', num_demos=5)
        
        elapsed = time.time() - start_time
        
        # Combine results
        results = {
            'system': type(self.rag).__name__,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_time_seconds': elapsed,
            'retrieval': retrieval_results,
            'generation': generation_results,
            'demo_count': len(demos)
        }
        
        # Print final summary
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"\n📊 Key Metrics:")
        print(f"  Retrieval:")
        print(f"    P@1:  {retrieval_results['metrics']['P@1']:.4f}")
        print(f"    P@5:  {retrieval_results['metrics']['P@5']:.4f}")
        print(f"    MRR:  {retrieval_results['metrics']['MRR']:.4f}")
        print(f"  Generation:")
        print(f"    ROUGE-1: {generation_results['metrics']['rouge1']:.4f}")
        print(f"    ROUGE-L: {generation_results['metrics']['rougeL']:.4f}")
        print(f"    BLEU:    {generation_results['metrics']['bleu']:.4f}")
        
        return results


def main():
    """Main evaluation pipeline."""
    
    print("="*80)
    print("  RAG EVALUATION SYSTEM")
    print("="*80)
    
    # Load RAG system
    print("\n📦 Loading RAG system...")
    
    if OLLAMA_AVAILABLE:
        print("  Using OllamaRAG...")
        rag = OllamaRAG()
        rag.load('rag_index')
    elif ENHANCED_AVAILABLE:
        print("  Using EnhancedRAG...")
        rag = EnhancedRAG()
        # Try to load index
        try:
            rag.load('enhanced_rag_index')
        except:
            print("  ⚠️  No index found, building...")
            rag.build_index('processed_text')
    else:
        print("❌ No RAG system available!")
        return
    
    # Load test questions
    test_file = 'rag_test_questions.json'
    if not Path(test_file).exists():
        print(f"❌ Test questions not found: {test_file}")
        return
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(rag, test_file)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Save results
    output_file = f'evaluation_results_{type(rag).__name__}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"💾 Demo answers saved to demo_answers.json")
    
    return results


if __name__ == "__main__":
    main()
