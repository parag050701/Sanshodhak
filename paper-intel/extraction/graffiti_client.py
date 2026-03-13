"""
Graffiti API Client - Primary extraction method using Meta's Graffiti.

Structured extraction of entities, relations, and claims from scientific text.
"""

import asyncio
import logging
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


class GraffitiClient:
    """
    Client for Meta's Graffiti API for structured knowledge extraction.
    
    Extracts entities, relations, and claims from scientific text.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        max_chunk_size: int = 3000,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize Graffiti client.
        
        Args:
            api_key: Graffiti API key (or from env)
            base_url: Graffiti API endpoint
            max_chunk_size: Max characters per API call
            timeout: Request timeout in seconds
            max_retries: Max retry attempts
        """
        self.api_key = api_key or os.getenv('GRAFFITI_API_KEY')
        self.base_url = base_url
        self.max_chunk_size = max_chunk_size
        self.timeout = timeout
        self.max_retries = max_retries
        
        if not self.api_key:
            logger.warning("No GRAFFITI_API_KEY found, extraction will fail")
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        logger.info(f"Graffiti client initialized: {base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for API calls.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If single paragraph is too long, split by sentences
            if len(para) > self.max_chunk_size:
                sentences = para.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= self.max_chunk_size:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
            else:
                # Add paragraph to current chunk
                if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + '\n\n'
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    async def _extract_chunk(
        self,
        text: str,
        retry_count: int = 0
    ) -> Dict:
        """
        Extract from single text chunk.
        
        Args:
            text: Text chunk to extract from
            retry_count: Current retry attempt
            
        Returns:
            Extraction result dict
        """
        if not self.api_key:
            raise ValueError("GRAFFITI_API_KEY not configured")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'text': text,
            'extract': ['entities', 'relations', 'claims'],
            'domain': 'scientific',
            'schema': {
                'entities': [
                    'Model', 'Method', 'Task', 'Dataset', 
                    'Metric', 'Result', 'Paper', 'Author'
                ],
                'relations': [
                    'USES_MODEL', 'USES_METHOD', 'USES_DATASET',
                    'EVALUATES_ON', 'REPORTS_METRIC', 'ACHIEVES_RESULT',
                    'COMPARES_WITH', 'CITES', 'AUTHORED_BY', 'STUDIES_TASK'
                ]
            }
        }
        
        try:
            response = await self.client.post(
                self.base_url,
                json=payload,
                headers=headers
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                
                if retry_count < self.max_retries:
                    return await self._extract_chunk(text, retry_count + 1)
                else:
                    raise Exception("Max retries exceeded for rate limiting")
            
            response.raise_for_status()
            result = response.json()
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            
            # Retry on server errors
            if e.response.status_code >= 500 and retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self._extract_chunk(text, retry_count + 1)
            
            raise
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    def _merge_extractions(self, extractions: List[Dict]) -> Dict:
        """
        Merge extractions from multiple chunks.
        
        Args:
            extractions: List of extraction results
            
        Returns:
            Merged extraction dict
        """
        merged = {
            'entities': [],
            'relations': [],
            'claims': []
        }
        
        # Deduplicate entities by name+type
        seen_entities = set()
        
        for extraction in extractions:
            # Merge entities
            for entity in extraction.get('entities', []):
                key = (entity.get('name', '').lower(), entity.get('type', ''))
                if key not in seen_entities and entity.get('name'):
                    seen_entities.add(key)
                    merged['entities'].append(entity)
            
            # Merge relations (may have duplicates, handled later)
            merged['relations'].extend(extraction.get('relations', []))
            
            # Merge claims
            merged['claims'].extend(extraction.get('claims', []))
        
        logger.info(
            f"Merged: {len(merged['entities'])} entities, "
            f"{len(merged['relations'])} relations, "
            f"{len(merged['claims'])} claims"
        )
        
        return merged
    
    async def extract(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract entities, relations, and claims from text.
        
        Args:
            text: Full text to extract from
            
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        if not text or not text.strip():
            return False, None, "Empty text"
        
        try:
            # Chunk text if needed
            chunks = self._chunk_text(text)
            
            logger.info(f"Extracting from {len(chunks)} chunks")
            
            # Extract from all chunks in parallel
            tasks = [self._extract_chunk(chunk) for chunk in chunks]
            extractions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            valid_extractions = []
            for i, result in enumerate(extractions):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i} failed: {result}")
                else:
                    valid_extractions.append(result)
            
            if not valid_extractions:
                return False, None, "All chunks failed extraction"
            
            # Merge results
            merged = self._merge_extractions(valid_extractions)
            
            return True, merged, None
            
        except Exception as e:
            error_msg = f"Graffiti extraction failed: {e}"
            logger.error(error_msg)
            return False, None, error_msg
    
    @classmethod
    def from_config(cls, config_path: Path = None):
        """
        Create client from config file.
        
        Args:
            config_path: Path to settings.yaml
            
        Returns:
            GraffitiClient instance
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return cls()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        graffiti_config = config.get('graffiti', {})
        
        return cls(
            base_url=graffiti_config.get('base_url', 'https://api.graffiti.ai/extract'),
            max_chunk_size=graffiti_config.get('max_chunk_size', 3000),
            timeout=graffiti_config.get('timeout', 60),
            max_retries=graffiti_config.get('max_retries', 3)
        )


# Standalone test
async def test_extraction():
    """Test Graffiti extraction."""
    sample_text = """
    We propose BERT, a bidirectional transformer model for natural language processing.
    BERT is evaluated on the GLUE benchmark dataset and achieves 92.4% accuracy.
    Our method outperforms GPT-2 by 3.5 points on question answering tasks.
    The model uses 110M parameters and is trained on BookCorpus.
    """
    
    client = GraffitiClient.from_config()
    
    async with client:
        success, result, error = await client.extract(sample_text)
        
        if success:
            print("✅ Extraction successful")
            print(f"Entities: {len(result['entities'])}")
            print(f"Relations: {len(result['relations'])}")
            print(f"Claims: {len(result['claims'])}")
            
            for entity in result['entities']:
                print(f"  - {entity.get('type')}: {entity.get('name')}")
        else:
            print(f"❌ Extraction failed: {error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_extraction())
