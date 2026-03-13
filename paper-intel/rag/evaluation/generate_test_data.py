"""
Generate ground truth test dataset for evaluation.
"""
import logging
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import httpx
import os
from dotenv import load_dotenv

load_dotenv("config/.env")

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate test questions and answers from papers."""
    
    def __init__(
        self,
        model: str = "meta-llama/llama-3.2-3b-instruct",
        api_key: str = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    async def generate_questions(
        self,
        text: str,
        paper_id: str,
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate questions from a paper.
        
        Args:
            text: Paper text
            paper_id: Paper identifier
            num_questions: Number of questions to generate
            
        Returns:
            List of question dicts with question, answer, and metadata
        """
        prompt = f"""Based on this research paper excerpt:

{text[:3000]}

Generate {num_questions} diverse questions that could be answered using this paper.
For each question, provide:
1. The question
2. The answer (2-3 sentences)
3. Question type (factual/conceptual/analytical/comparison)
4. Difficulty (easy/medium/hard)

Format as JSON array:
[
  {{
    "question": "...",
    "answer": "...",
    "type": "...",
    "difficulty": "..."
  }},
  ...
]"""
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    
                    # Extract JSON
                    import re
                    json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
                    if json_match:
                        questions = json.loads(json_match.group())
                        
                        # Add metadata
                        for q in questions:
                            q['paper_id'] = paper_id
                            q['relevant_docs'] = [paper_id]
                        
                        return questions
                
        except Exception as e:
            logger.warning(f"Question generation failed for {paper_id}: {e}")
        
        return []
    
    async def generate_dataset(
        self,
        text_dir: Path,
        output_path: Path,
        num_papers: int = 10,
        questions_per_paper: int = 3
    ):
        """
        Generate test dataset from papers.
        
        Args:
            text_dir: Directory with text files
            output_path: Output path for test dataset
            num_papers: Number of papers to use
            questions_per_paper: Questions per paper
        """
        logger.info(f"Generating test dataset from {num_papers} papers...")
        
        text_files = list(text_dir.glob("*.txt"))[:num_papers]
        
        all_questions = []
        
        for text_file in text_files:
            paper_id = text_file.stem
            text = text_file.read_text(encoding='utf-8', errors='ignore')
            
            questions = await self.generate_questions(
                text,
                paper_id,
                questions_per_paper
            )
            
            all_questions.extend(questions)
            logger.info(f"Generated {len(questions)} questions from {paper_id}")
        
        # Save dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_questions, f, indent=2)
        
        logger.info(f"✅ Generated {len(all_questions)} questions")
        logger.info(f"   Saved to {output_path}")


async def main():
    """Generate test dataset."""
    generator = TestDataGenerator()
    
    await generator.generate_dataset(
        text_dir=Path("ingestion/raw_text"),
        output_path=Path("rag/evaluation/test_dataset.json"),
        num_papers=10,
        questions_per_paper=3
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
