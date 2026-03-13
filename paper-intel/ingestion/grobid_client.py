"""
GROBID Client for PDF → TEI XML conversion.

Provides async and sync interfaces to GROBID REST API.
Handles retries, timeouts, and connection pooling.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


class GrobidClient:
    """
    Client for GROBID API.
    
    Endpoints:
        /api/processFulltextDocument - Full paper processing
        /api/processHeaderDocument   - Header-only processing
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout: int = 60,
        max_retries: int = 3,
        max_connections: int = 10
    ):
        """
        Initialize GROBID client.
        
        Args:
            base_url: GROBID server URL
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            max_connections: Max concurrent connections
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Create async client with connection pooling
        self.async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections // 2
            )
        )
        
        # Sync client for non-async usage
        self.sync_client = httpx.Client(
            timeout=httpx.Timeout(timeout)
        )
        
        logger.info(f"GROBID client initialized: {base_url}")
    
    async def health_check(self) -> bool:
        """Check if GROBID server is running."""
        try:
            response = await self.async_client.get(f"{self.base_url}/api/isalive")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GROBID health check failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def process_fulltext(
        self,
        pdf_path: Path,
        consolidate_header: bool = True,
        consolidate_citations: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process full PDF with GROBID.
        
        Args:
            pdf_path: Path to PDF file
            consolidate_header: Consolidate header metadata
            consolidate_citations: Consolidate citation metadata
            
        Returns:
            Tuple of (success, tei_xml, error_message)
        """
        if not pdf_path.exists():
            return False, None, f"PDF not found: {pdf_path}"
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (pdf_path.name, f, 'application/pdf')}
                
                data = {
                    'consolidateHeader': '1' if consolidate_header else '0',
                    'consolidateCitations': '1' if consolidate_citations else '0',
                    'includeRawAffiliations': '1',
                    'includeRawCitations': '1',
                    'teiCoordinates': ['ref', 'biblStruct', 'formula', 'figure']
                }
                
                response = await self.async_client.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    tei_xml = response.text
                    logger.debug(f"Successfully processed: {pdf_path.name}")
                    return True, tei_xml, None
                else:
                    error_msg = f"GROBID returned status {response.status_code}"
                    logger.error(f"{error_msg} for {pdf_path.name}")
                    return False, None, error_msg
                    
        except httpx.TimeoutException:
            error_msg = f"Timeout processing {pdf_path.name}"
            logger.error(error_msg)
            raise  # Let retry handle it
            
        except httpx.ConnectError:
            error_msg = f"Cannot connect to GROBID at {self.base_url}"
            logger.error(error_msg)
            raise  # Let retry handle it
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def process_header(
        self,
        pdf_path: Path,
        consolidate: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process only PDF header (faster, metadata only).
        
        Args:
            pdf_path: Path to PDF file
            consolidate: Consolidate metadata
            
        Returns:
            Tuple of (success, tei_xml, error_message)
        """
        if not pdf_path.exists():
            return False, None, f"PDF not found: {pdf_path}"
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (pdf_path.name, f, 'application/pdf')}
                data = {'consolidateHeader': '1' if consolidate else '0'}
                
                response = await self.async_client.post(
                    f"{self.base_url}/api/processHeaderDocument",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    tei_xml = response.text
                    logger.debug(f"Successfully processed header: {pdf_path.name}")
                    return True, tei_xml, None
                else:
                    error_msg = f"GROBID returned status {response.status_code}"
                    logger.error(f"{error_msg} for {pdf_path.name}")
                    return False, None, error_msg
                    
        except httpx.TimeoutException:
            error_msg = f"Timeout processing header {pdf_path.name}"
            logger.error(error_msg)
            raise
            
        except httpx.ConnectError:
            error_msg = f"Cannot connect to GROBID at {self.base_url}"
            logger.error(error_msg)
            raise
            
        except Exception as e:
            error_msg = f"Error processing header {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    async def process_batch(
        self,
        pdf_paths: list[Path],
        max_concurrent: int = 4
    ) -> list[Tuple[Path, bool, Optional[str], Optional[str]]]:
        """
        Process multiple PDFs concurrently.
        
        Args:
            pdf_paths: List of PDF paths
            max_concurrent: Max concurrent requests
            
        Returns:
            List of (pdf_path, success, tei_xml, error_message)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_path: Path):
            async with semaphore:
                success, tei_xml, error = await self.process_fulltext(pdf_path)
                return (pdf_path, success, tei_xml, error)
        
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pdf_path = pdf_paths[i]
                processed_results.append(
                    (pdf_path, False, None, str(result))
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_fulltext_sync(
        self,
        pdf_path: Path,
        consolidate_header: bool = True,
        consolidate_citations: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Synchronous version of process_fulltext.
        
        For use in non-async contexts.
        """
        if not pdf_path.exists():
            return False, None, f"PDF not found: {pdf_path}"
        
        for attempt in range(self.max_retries):
            try:
                with open(pdf_path, 'rb') as f:
                    files = {'input': (pdf_path.name, f, 'application/pdf')}
                    
                    data = {
                        'consolidateHeader': '1' if consolidate_header else '0',
                        'consolidateCitations': '1' if consolidate_citations else '0',
                        'includeRawAffiliations': '1',
                        'includeRawCitations': '1'
                    }
                    
                    response = self.sync_client.post(
                        f"{self.base_url}/api/processFulltextDocument",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        return True, response.text, None
                    else:
                        error_msg = f"GROBID returned status {response.status_code}"
                        if attempt == self.max_retries - 1:
                            return False, None, error_msg
                        
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return False, None, str(e)
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False, None, "Max retries exceeded"
    
    async def close(self):
        """Close async client."""
        await self.async_client.aclose()
    
    def close_sync(self):
        """Close sync client."""
        self.sync_client.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.sync_client.close()
        except:
            pass


async def test_grobid_connection(base_url: str = "http://localhost:8070") -> bool:
    """
    Test GROBID server connection.
    
    Args:
        base_url: GROBID server URL
        
    Returns:
        True if server is reachable and healthy
    """
    client = GrobidClient(base_url=base_url)
    try:
        is_healthy = await client.health_check()
        if is_healthy:
            logger.info(f"✅ GROBID server is healthy at {base_url}")
        else:
            logger.error(f"❌ GROBID server not responding at {base_url}")
        return is_healthy
    finally:
        await client.close()


if __name__ == "__main__":
    # Test GROBID connection
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        result = await test_grobid_connection()
        if result:
            print("✅ GROBID is ready")
        else:
            print("❌ GROBID is not available")
            print("Start GROBID with: docker run -p 8070:8070 lfoppiano/grobid:0.8.0")
    
    asyncio.run(main())
