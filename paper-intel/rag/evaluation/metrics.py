"""
Comprehensive RAG evaluation metrics.
"""
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score
import torch
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Comprehensive RAG evaluation."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
    
    # ========================================================================
    # RETRIEVAL METRICS
    # ========================================================================
    
    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int = 10
    ) -> float:
        """
        Precision@K: proportion of retrieved docs that are relevant.
        
        Args:
            retrieved: Retrieved document IDs (ordered)
            relevant: Relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Precision score (0-1)
        """
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not retrieved_k:
            return 0.0
        
        return len(retrieved_k & relevant_set) / len(retrieved_k)
    
    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int = 10
    ) -> float:
        """
        Recall@K: proportion of relevant docs that are retrieved.
        
        Args:
            retrieved: Retrieved document IDs (ordered)
            relevant: Relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Recall score (0-1)
        """
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_k & relevant_set) / len(relevant_set)
    
    def mean_reciprocal_rank(
        self,
        retrieved_list: List[List[str]],
        relevant_list: List[List[str]]
    ) -> float:
        """
        MRR: average reciprocal rank of first relevant document.
        
        Args:
            retrieved_list: List of retrieved document lists (per query)
            relevant_list: List of relevant document lists (per query)
            
        Returns:
            MRR score (0-1)
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Dict[str, float],
        k: int = 10
    ) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved: Retrieved document IDs (ordered)
            relevant: Relevant document IDs
            relevance_scores: Relevance scores for each doc
            k: Number of top results to consider
            
        Returns:
            NDCG score (0-1)
        """
        retrieved_k = retrieved[:k]
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k, start=1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)
        
        # IDCG (ideal DCG)
        ideal_scores = sorted(
            [relevance_scores.get(doc_id, 0.0) for doc_id in relevant],
            reverse=True
        )[:k]
        
        idcg = sum(
            score / np.log2(i + 2)
            for i, score in enumerate(ideal_scores)
        )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mean_average_precision(
        self,
        retrieved_list: List[List[str]],
        relevant_list: List[List[str]]
    ) -> float:
        """
        MAP: Mean Average Precision across queries.
        
        Args:
            retrieved_list: List of retrieved document lists
            relevant_list: List of relevant document lists
            
        Returns:
            MAP score (0-1)
        """
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            
            if not relevant_set:
                continue
            
            precisions = []
            num_relevant = 0
            
            for k, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant_set:
                    num_relevant += 1
                    precision = num_relevant / k
                    precisions.append(precision)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    # ========================================================================
    # GENERATION METRICS
    # ========================================================================
    
    def faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Faithfulness: Are answer claims grounded in context?
        
        Simple approximation: keyword overlap with contexts.
        
        Returns:
            Faithfulness score (0-1)
        """
        answer_words = set(answer.lower().split())
        if not answer_words:
            return 0.0
        
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)
    
    def answer_relevance(
        self,
        answer: str,
        query: str
    ) -> float:
        """
        Answer Relevance: Does answer address the query?
        
        Simple approximation: keyword overlap.
        
        Returns:
            Relevance score (0-1)
        """
        answer_words = set(answer.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(answer_words & query_words)
        return overlap / len(query_words)
    
    def context_relevance(
        self,
        contexts: List[str],
        query: str
    ) -> float:
        """
        Context Relevance: Are retrieved contexts relevant to query?
        
        Simple approximation: average keyword overlap.
        
        Returns:
            Relevance score (0-1)
        """
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        
        relevances = []
        for context in contexts:
            context_words = set(context.lower().split())
            if context_words:
                overlap = len(context_words & query_words)
                relevances.append(overlap / len(query_words))
        
        return np.mean(relevances) if relevances else 0.0
    
    def bleu_score(
        self,
        answer: str,
        reference: str
    ) -> float:
        """
        BLEU: Measure n-gram overlap with reference.
        
        Returns:
            BLEU score (0-1)
        """
        answer_tokens = answer.lower().split()
        reference_tokens = [reference.lower().split()]
        
        return sentence_bleu(
            reference_tokens,
            answer_tokens,
            smoothing_function=self.smoothing.method1
        )
    
    def rouge_scores(
        self,
        answer: str,
        reference: str
    ) -> Dict[str, float]:
        """
        ROUGE: Recall-oriented n-gram overlap.
        
        Returns:
            Dict with rouge1, rouge2, rougeL F1 scores
        """
        scores = self.rouge_scorer.score(reference, answer)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def bert_score(
        self,
        answer: str,
        reference: str,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        BERTScore: Semantic similarity using BERT embeddings.
        
        Returns:
            Dict with precision, recall, F1
        """
        try:
            P, R, F1 = bertscore(
                [answer],
                [reference],
                lang="en",
                device=device,
                verbose=False
            )
            
            return {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # ========================================================================
    # END-TO-END METRICS
    # ========================================================================
    
    def answer_correctness(
        self,
        answer: str,
        reference: str
    ) -> float:
        """
        Answer Correctness: Combined semantic and lexical similarity.
        
        Returns:
            Correctness score (0-1)
        """
        # Combine multiple metrics
        rouge = self.rouge_scores(answer, reference)
        bert = self.bert_score(answer, reference)
        
        # Weighted average
        score = (
            0.3 * rouge['rouge1'] +
            0.3 * rouge['rougeL'] +
            0.4 * bert['f1']
        )
        
        return score
    
    def hallucination_rate(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Hallucination Rate: Proportion of answer NOT grounded in context.
        
        Returns:
            Hallucination rate (0-1, lower is better)
        """
        faithfulness_score = self.faithfulness(answer, contexts)
        return 1.0 - faithfulness_score
    
    def citation_accuracy(
        self,
        answer: str,
        contexts: List[Dict[str, Any]]
    ) -> float:
        """
        Citation Accuracy: Are citations in answer actually from contexts?
        
        Returns:
            Accuracy score (0-1)
        """
        # Extract paper IDs mentioned in answer
        import re
        cited_papers = set(re.findall(r'Paper:\s*(\w+)', answer))
        
        # Get available paper IDs from contexts
        available_papers = set(ctx.get('paper_id', '') for ctx in contexts)
        
        if not cited_papers:
            return 1.0  # No citations means no errors
        
        correct_citations = len(cited_papers & available_papers)
        return correct_citations / len(cited_papers)
    
    # ========================================================================
    # EVALUATION SUITE
    # ========================================================================
    
    def evaluate_retrieval(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance.
        
        Args:
            test_cases: List of dicts with 'query', 'retrieved', 'relevant', 'scores'
            
        Returns:
            Dict with all retrieval metrics
        """
        logger.info("Evaluating retrieval...")
        
        metrics = defaultdict(list)
        
        for case in test_cases:
            retrieved = case['retrieved']
            relevant = case['relevant']
            scores = case.get('scores', {})
            
            # Precision/Recall at various K
            for k in [1, 3, 5, 10, 20]:
                metrics[f'precision@{k}'].append(
                    self.precision_at_k(retrieved, relevant, k)
                )
                metrics[f'recall@{k}'].append(
                    self.recall_at_k(retrieved, relevant, k)
                )
            
            # NDCG
            for k in [5, 10, 20]:
                metrics[f'ndcg@{k}'].append(
                    self.ndcg_at_k(retrieved, relevant, scores, k)
                )
        
        # MRR and MAP
        retrieved_list = [case['retrieved'] for case in test_cases]
        relevant_list = [case['relevant'] for case in test_cases]
        
        metrics['mrr'] = self.mean_reciprocal_rank(retrieved_list, relevant_list)
        metrics['map'] = self.mean_average_precision(retrieved_list, relevant_list)
        
        # Average metrics
        results = {'mrr': metrics['mrr'], 'map': metrics['map']}
        for key, values in metrics.items():
            if isinstance(values, list):
                results[key] = np.mean(values)
        
        return results
    
    def evaluate_generation(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality.
        
        Args:
            test_cases: List of dicts with 'query', 'answer', 'reference', 'contexts'
            
        Returns:
            Dict with all generation metrics
        """
        logger.info("Evaluating generation...")
        
        metrics = defaultdict(list)
        
        for case in test_cases:
            answer = case['answer']
            reference = case.get('reference', '')
            contexts = case.get('contexts', [])
            query = case['query']
            
            # Faithfulness and relevance
            if contexts:
                context_texts = [c['text'] if isinstance(c, dict) else c for c in contexts]
                metrics['faithfulness'].append(self.faithfulness(answer, context_texts))
                metrics['context_relevance'].append(self.context_relevance(context_texts, query))
                metrics['hallucination_rate'].append(self.hallucination_rate(answer, context_texts))
            
            metrics['answer_relevance'].append(self.answer_relevance(answer, query))
            
            # Lexical metrics (if reference available)
            if reference:
                metrics['bleu'].append(self.bleu_score(answer, reference))
                rouge = self.rouge_scores(answer, reference)
                for k, v in rouge.items():
                    metrics[k].append(v)
                
                bert = self.bert_score(answer, reference)
                for k, v in bert.items():
                    metrics[f'bertscore_{k}'].append(v)
                
                metrics['answer_correctness'].append(
                    self.answer_correctness(answer, reference)
                )
            
            # Citation accuracy
            if contexts:
                metrics['citation_accuracy'].append(
                    self.citation_accuracy(answer, contexts)
                )
        
        # Average metrics
        results = {}
        for key, values in metrics.items():
            if values:
                results[key] = np.mean(values)
        
        return results
    
    def evaluate_performance(
        self,
        latencies: List[float],
        token_counts: List[Dict[str, int]]
    ) -> Dict[str, Any]:
        """
        Evaluate system performance.
        
        Args:
            latencies: List of query latencies (seconds)
            token_counts: List of dicts with 'input', 'output' token counts
            
        Returns:
            Dict with performance metrics
        """
        logger.info("Evaluating performance...")
        
        results = {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'throughput': 1.0 / np.mean(latencies) if latencies else 0.0
        }
        
        if token_counts:
            input_tokens = [tc['input'] for tc in token_counts]
            output_tokens = [tc['output'] for tc in token_counts]
            
            results['mean_input_tokens'] = np.mean(input_tokens)
            results['mean_output_tokens'] = np.mean(output_tokens)
            results['mean_total_tokens'] = np.mean(input_tokens) + np.mean(output_tokens)
        
        return results
    
    def run_full_evaluation(
        self,
        test_cases: List[Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Run complete evaluation suite.
        
        Args:
            test_cases: List of test cases with all required fields
            output_path: Path to save results
            
        Returns:
            Dict with all evaluation results
        """
        logger.info("=" * 80)
        logger.info("RUNNING FULL EVALUATION")
        logger.info("=" * 80)
        
        results = {}
        
        # Retrieval metrics
        retrieval_cases = [
            {
                'query': case['query'],
                'retrieved': case['retrieved_ids'],
                'relevant': case['relevant_ids'],
                'scores': case.get('relevance_scores', {})
            }
            for case in test_cases
        ]
        results['retrieval'] = self.evaluate_retrieval(retrieval_cases)
        
        # Generation metrics
        generation_cases = [
            {
                'query': case['query'],
                'answer': case['answer'],
                'reference': case.get('reference_answer', ''),
                'contexts': case['retrieved_contexts']
            }
            for case in test_cases
        ]
        results['generation'] = self.evaluate_generation(generation_cases)
        
        # Performance metrics
        latencies = [case['latency'] for case in test_cases]
        token_counts = [case.get('tokens', {}) for case in test_cases]
        results['performance'] = self.evaluate_performance(latencies, token_counts)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Evaluation complete. Results saved to {output_path}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print("\n📊 RETRIEVAL METRICS:")
        for k, v in results['retrieval'].items():
            print(f"  {k:20s}: {v:.4f}")
        
        print("\n💬 GENERATION METRICS:")
        for k, v in results['generation'].items():
            print(f"  {k:20s}: {v:.4f}")
        
        print("\n⚡ PERFORMANCE METRICS:")
        for k, v in results['performance'].items():
            if 'latency' in k:
                print(f"  {k:20s}: {v:.3f}s")
            elif 'throughput' in k:
                print(f"  {k:20s}: {v:.2f} queries/s")
            else:
                print(f"  {k:20s}: {v:.1f}")
        
        print("=" * 80)
