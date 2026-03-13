"""
Run comprehensive RAG evaluation.
"""
import logging
import sys
import asyncio
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.core.pipeline import AdvancedRAG
from rag.evaluation.metrics import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_evaluation():
    """Run full RAG evaluation."""
    
    # Load RAG system
    logger.info("Loading RAG system...")
    rag = AdvancedRAG(config_path="rag/config/rag_config.yaml")
    rag.load_index(Path("rag/index"))
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_path = Path("rag/evaluation/test_dataset.json")
    
    if not test_path.exists():
        logger.error(f"Test dataset not found: {test_path}")
        logger.error("Run: python rag/evaluation/generate_test_data.py")
        return
    
    with open(test_path) as f:
        test_questions = json.load(f)
    
    logger.info(f"Loaded {len(test_questions)} test questions")
    
    # Run queries and collect results
    logger.info("Running queries...")
    test_cases = []
    
    for i, test_q in enumerate(test_questions, 1):
        query = test_q['question']
        reference = test_q['answer']
        relevant_paper = test_q['paper_id']
        
        logger.info(f"\n[{i}/{len(test_questions)}] Query: {query}")
        
        # Time query
        start_time = time.time()
        result = await rag.query(query)
        latency = time.time() - start_time
        
        # Extract retrieved document IDs
        retrieved_ids = [ctx['paper_id'] for ctx in result.contexts]
        
        # Build relevance scores (1.0 for relevant paper, 0.0 for others)
        relevance_scores = {
            ctx['paper_id']: 1.0 if ctx['paper_id'] == relevant_paper else 0.0
            for ctx in result.contexts
        }
        
        test_case = {
            'query': query,
            'answer': result.answer,
            'reference_answer': reference,
            'retrieved_ids': retrieved_ids,
            'relevant_ids': [relevant_paper],
            'relevance_scores': relevance_scores,
            'retrieved_contexts': result.contexts,
            'latency': latency,
            'metadata': result.metadata
        }
        
        test_cases.append(test_case)
        
        logger.info(f"  Latency: {latency:.2f}s")
        logger.info(f"  Retrieved: {len(retrieved_ids)} docs")
        logger.info(f"  Answer: {result.answer[:100]}...")
    
    # Run evaluation
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING RAG SYSTEM")
    logger.info("=" * 80)
    
    evaluator = RAGEvaluator()
    results = evaluator.run_full_evaluation(
        test_cases,
        Path("rag/evaluation/results.json")
    )
    
    # Save detailed results
    with open("rag/evaluation/detailed_results.json", 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    logger.info(f"\n✅ Evaluation complete!")
    logger.info(f"   Summary: rag/evaluation/results.json")
    logger.info(f"   Details: rag/evaluation/detailed_results.json")


def main():
    """Main entry point."""
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
