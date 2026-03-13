"""
LLM Fallback Extractor - Uses OpenRouter when Graffiti fails.

Structured prompt-based extraction using Qwen-2.5-32B-Instruct.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple
import httpx
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

# Load environment variables from config/.env
config_env = Path(__file__).parent.parent / "config" / ".env"
if config_env.exists():
    load_dotenv(config_env)

logger = logging.getLogger(__name__)


class LLMFallbackExtractor:
    """
    Fallback extractor using OpenRouter LLM API.
    
    Uses structured prompts to extract entities, relations, and claims
    when Graffiti is unavailable or fails.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.2-3b-instruct",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_chunk_size: int = 4000,
        timeout: int = 120,
        temperature: float = 0.1
    ):
        """
        Initialize LLM fallback extractor.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier
            base_url: OpenRouter API endpoint
            max_chunk_size: Max characters per request
            timeout: Request timeout
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        self.base_url = base_url
        self.max_chunk_size = max_chunk_size
        self.timeout = timeout
        self.temperature = temperature
        
        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY found, fallback will fail")
        
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
        )
        
        logger.info(f"LLM fallback initialized: {model}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    def _build_extraction_prompt(self, text: str) -> str:
        """
        Build extraction prompt for LLM.
        
        Args:
            text: Scientific text to extract from
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert at extracting structured knowledge from scientific papers.

Extract the following from the text:
1. **Entities** (Models, Methods, Tasks, Datasets, Metrics, Results, Papers, Authors)
2. **Relations** between entities
3. **Claims** (metric results, comparisons)

**Entity Types:**
- Model: Neural network architectures, AI models (e.g., BERT, GPT, ResNet)
- Method: Algorithms, techniques, approaches (e.g., attention mechanism, backpropagation)
- Task: Problems being solved (e.g., machine translation, image classification)
- Dataset: Benchmarks, corpora (e.g., ImageNet, GLUE)
- Metric: Evaluation measures (e.g., accuracy, F1-score, BLEU)
- Result: Numerical outcomes, findings
- Paper: Referenced publications
- Author: Researchers, scientists

**Relation Types:**
- USES_MODEL: Entity uses a model
- USES_METHOD: Entity uses a method/technique
- USES_DATASET: Evaluates on a dataset
- EVALUATES_ON: Tests on benchmark
- REPORTS_METRIC: Reports a metric value
- ACHIEVES_RESULT: Achieves performance result
- COMPARES_WITH: Compares against baseline
- CITES: References another paper
- AUTHORED_BY: Paper written by author
- STUDIES_TASK: Addresses a task/problem

**Output Format (JSON):**
```json
{{
  "entities": [
    {{"type": "Model", "name": "BERT"}},
    {{"type": "Dataset", "name": "ImageNet"}},
    {{"type": "Metric", "name": "Accuracy"}}
  ],
  "relations": [
    {{"source": "BERT", "type": "EVALUATES_ON", "target": "ImageNet"}},
    {{"source": "BERT", "type": "REPORTS_METRIC", "target": "Accuracy"}}
  ],
  "claims": [
    {{
      "metric_name": "Accuracy",
      "metric_value": "92.4%",
      "dataset": "ImageNet",
      "experiment_description": "Top-1 accuracy on ImageNet validation set"
    }}
  ]
}}
```

**Text to analyze:**
{text}

**Output (JSON only, no explanation):**"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response to extract JSON.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed JSON dict or None
        """
        try:
            # Try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object anywhere in text
        json_match = re.search(r'\{.*"entities".*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Could not parse LLM response: {response_text[:200]}")
        return None
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for LLM processing.
        
        Args:
            text: Full text
            
        Returns:
            List of chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Split by double newline (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too long, truncate
                if len(para) > self.max_chunk_size:
                    current_chunk = para[:self.max_chunk_size] + '\n\n'
                else:
                    current_chunk = para + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks for LLM")
        return chunks
    
    async def _extract_chunk(self, text: str) -> Optional[Dict]:
        """
        Extract from single chunk using LLM.
        
        Args:
            text: Text chunk
            
        Returns:
            Extraction result or None
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        prompt = self._build_extraction_prompt(text)
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/sanshodhak/paper-intel',
            'X-Title': 'Sanshodhak Research Intelligence'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': self.temperature,
            'max_tokens': 2000
        }
        
        try:
            response = await self.client.post(
                self.base_url,
                json=payload,
                headers=headers
            )
            
            # Log response for debugging
            if response.status_code != 200:
                logger.error(f"OpenRouter API error {response.status_code}: {response.text[:500]}")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from content
            parsed = self._parse_llm_response(content)
            
            if not parsed:
                logger.error("Failed to parse LLM response")
                return None
            
            # Ensure required keys exist
            if 'entities' not in parsed:
                parsed['entities'] = []
            if 'relations' not in parsed:
                parsed['relations'] = []
            if 'claims' not in parsed:
                parsed['claims'] = []
            
            return parsed
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None
    
    def _merge_extractions(self, extractions: List[Dict]) -> Dict:
        """
        Merge extractions from multiple chunks.
        
        Args:
            extractions: List of extraction dicts
            
        Returns:
            Merged extraction dict
        """
        merged = {
            'entities': [],
            'relations': [],
            'claims': []
        }
        
        # Deduplicate entities
        seen_entities = set()
        
        for extraction in extractions:
            # Merge entities
            for entity in extraction.get('entities', []):
                key = (entity.get('name', '').lower(), entity.get('type', ''))
                if key not in seen_entities and entity.get('name'):
                    seen_entities.add(key)
                    merged['entities'].append(entity)
            
            # Merge relations
            merged['relations'].extend(extraction.get('relations', []))
            
            # Merge claims
            merged['claims'].extend(extraction.get('claims', []))
        
        logger.info(
            f"LLM merged: {len(merged['entities'])} entities, "
            f"{len(merged['relations'])} relations, "
            f"{len(merged['claims'])} claims"
        )
        
        return merged
    
    async def extract(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract using LLM fallback.
        
        Args:
            text: Full text to extract from
            
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        if not text or not text.strip():
            return False, None, "Empty text"
        
        try:
            # Chunk text
            chunks = self._chunk_text(text)
            
            logger.info(f"LLM extracting from {len(chunks)} chunks")
            
            # Process chunks sequentially (to avoid rate limits)
            extractions = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                result = await self._extract_chunk(chunk)
                
                if result:
                    extractions.append(result)
                
                # Small delay to avoid rate limits
                if i < len(chunks) - 1:
                    await asyncio.sleep(1)
            
            if not extractions:
                return False, None, "All chunks failed LLM extraction"
            
            # Merge results
            merged = self._merge_extractions(extractions)
            
            return True, merged, None
            
        except Exception as e:
            error_msg = f"LLM fallback failed: {e}"
            logger.error(error_msg)
            return False, None, error_msg
    
    @classmethod
    def from_config(cls, config_path: Path = None):
        """
        Create extractor from config file.
        
        Args:
            config_path: Path to settings.yaml
            
        Returns:
            LLMFallbackExtractor instance
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return cls()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        llm_config = config.get('llm', {})
        
        return cls(
            model=llm_config.get('model', 'meta-llama/llama-3.2-3b-instruct'),
            base_url=llm_config.get('base_url', 'https://openrouter.ai/api/v1/chat/completions'),
            max_chunk_size=llm_config.get('max_chunk_size', 4000),
            timeout=llm_config.get('timeout', 120),
            temperature=llm_config.get('temperature', 0.1)
        )


# Standalone test
async def test_llm_extraction():
    """Test LLM fallback extraction."""
    sample_text = """
    We propose BERT, a bidirectional transformer model for natural language processing.
    BERT is evaluated on the GLUE benchmark dataset and achieves 92.4% accuracy.
    Our method outperforms GPT-2 by 3.5 points on question answering tasks.
    The model uses 110M parameters and is trained on BookCorpus.
    """
    
    extractor = LLMFallbackExtractor.from_config()
    
    async with extractor:
        success, result, error = await extractor.extract(sample_text)
        
        if success:
            print("✅ LLM extraction successful")
            print(f"Entities: {len(result['entities'])}")
            print(f"Relations: {len(result['relations'])}")
            print(f"Claims: {len(result['claims'])}")
            
            for entity in result['entities']:
                print(f"  - {entity.get('type')}: {entity.get('name')}")
        else:
            print(f"❌ LLM extraction failed: {error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_llm_extraction())
