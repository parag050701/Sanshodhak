"""CORE API client for open-access papers."""
import logging
from typing import List, Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class COREClient(BaseAPIClient):
    """
    Client for CORE API (https://core.ac.uk/services/api).
    
    World's largest collection of open-access papers (200M+).
    Free: 100 requests/day. Registered: 1,000 requests/day.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        super().__init__(
            base_url="https://api.core.ac.uk/v3",
            rate_limit_delay=0.6 if api_key else 6.0,  # 100/min with key
            timeout=30,
            headers=headers
        )
        self.api_key = api_key
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """
        Search CORE for open-access papers using /v3/search/works endpoint.
        
        Args:
            query: Search query
            limit: Maximum results
            min_year: Minimum publication year
            
        Returns:
            SearchResult with papers
        """
        start_time = time.time()
        
        try:
            # Use POST request as per CORE v3 API docs
            headers = {
                "Content-Type": "application/json"
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Build query with year filter if needed
            search_query = query
            if min_year:
                search_query = f'{query} AND yearPublished:>={min_year}'
            
            payload = {
                "q": search_query,
                "limit": min(limit, 100)
            }
            
            response = self._make_request(
                'POST',
                '/search/works',
                json_data=payload,
                headers=headers
            )
            
            if not response or response.status_code != 200:
                error_msg = "Rate limited or auth error" if response and response.status_code in [429, 401] else "Request failed"
                return SearchResult(
                    source='core',
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=error_msg,
                    search_time_ms=(time.time() - start_time) * 1000
                )
            
            data = response.json()
            results = data.get('results', [])
            
            papers = []
            for item in results:
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            
            return SearchResult(
                source='core',
                query=query,
                papers=papers,
                total_found=data.get('totalHits', 0),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"CORE search error: {e}")
            return SearchResult(
                source='core',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def _parse_work(self, item: dict) -> Optional[PaperMetadata]:
        """Parse CORE work JSON to PaperMetadata."""
        try:
            # Extract DOI
            doi = item.get('doi')
            if not doi and item.get('identifiers'):
                for identifier in item['identifiers']:
                    if isinstance(identifier, dict):
                        id_val = identifier.get('identifier', '')
                        if id_val.startswith('10.'):
                            doi = id_val
                            break
            
            # Extract PDF URL
            pdf_url = item.get('downloadUrl')
            is_oa = bool(pdf_url)
            
            if not pdf_url and item.get('links'):
                for link in item['links']:
                    if isinstance(link, dict) and link.get('type') == 'download':
                        pdf_url = link.get('url')
                        is_oa = True
                        break
            
            # Extract authors
            authors = []
            for author in item.get('authors', []):
                if isinstance(author, dict):
                    name = author.get('name', '')
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    authors.append(author)
            
            paper = PaperMetadata(
                doi=doi,
                title=item.get('title', ''),
                authors=authors,
                abstract=item.get('abstract'),
                year=item.get('yearPublished'),
                venue=item.get('publisher', {}).get('name') if isinstance(item.get('publisher'), dict) else item.get('publisher'),
                publisher=item.get('publisher', {}).get('name') if isinstance(item.get('publisher'), dict) else None,
                is_open_access=is_oa,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url if is_oa else None,
                source='core',
                score=0.0  # CORE doesn't provide relevance scores
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse CORE work: {e}")
            return None
