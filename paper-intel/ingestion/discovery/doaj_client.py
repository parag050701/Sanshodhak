"""DOAJ (Directory of Open Access Journals) API client."""
import logging
from typing import List, Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class DOAJClient(BaseAPIClient):
    """
    Client for DOAJ API (https://doaj.org/api/v3/docs).
    
    Directory of Open Access Journals - fully OA papers from verified journals.
    No API key required. Rate limit: ~100 req/min.
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://doaj.org/api/v3",
            rate_limit_delay=0.6,  # ~100/min
            timeout=30
        )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """
        Search DOAJ for open-access articles.
        
        Args:
            query: Search query
            limit: Maximum results (max 100 per request)
            min_year: Minimum publication year
            
        Returns:
            SearchResult with papers
        """
        start_time = time.time()
        
        try:
            # DOAJ uses Lucene query syntax
            search_query = query
            if min_year:
                search_query = f'{query} AND bibjson.year:>={min_year}'
            
            params = {
                'ref': 'title,abstract,keywords',  # Search in these fields
                'pageSize': min(limit, 100),
                'page': 1,
                'sort': 'bibjson.year:desc'  # Sort by year descending
            }
            
            # DOAJ API v3 expects the query in the URL path, not as 'q' param
            from urllib.parse import quote
            safe_query = quote(search_query)
            response = self.get(f'/search/articles/{safe_query}', params=params)
            
            if not response or response.status_code != 200:
                return SearchResult(
                    source='doaj',
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start_time) * 1000
                )
            
            data = response.json()
            results = data.get('results', [])
            
            papers = []
            for item in results:
                paper = self._parse_article(item)
                if paper:
                    papers.append(paper)
            
            return SearchResult(
                source='doaj',
                query=query,
                papers=papers,
                total_found=data.get('total', 0),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"DOAJ search error: {e}")
            return SearchResult(
                source='doaj',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def _parse_article(self, item: dict) -> Optional[PaperMetadata]:
        """Parse DOAJ article JSON to PaperMetadata."""
        try:
            bibjson = item.get('bibjson', {})
            
            # Extract DOI
            doi = None
            for identifier in bibjson.get('identifier', []):
                if identifier.get('type') == 'doi':
                    doi = identifier.get('id')
                    break
            
            # Extract authors
            authors = []
            for author in bibjson.get('author', []):
                if author.get('name'):
                    authors.append(author['name'])
            
            # Extract keywords
            keywords = []
            for keyword in bibjson.get('keywords', []):
                if isinstance(keyword, str):
                    keywords.append(keyword.lower())
                elif isinstance(keyword, dict) and keyword.get('term'):
                    keywords.append(keyword['term'].lower())
            
            # Extract PDF link
            pdf_url = None
            for link in bibjson.get('link', []):
                if link.get('type') == 'fulltext' and link.get('content_type') == 'PDF':
                    pdf_url = link.get('url')
                    break
            
            # Extract journal info
            journal = bibjson.get('journal', {}).get('title')
            
            paper = PaperMetadata(
                doi=doi,
                title=bibjson.get('title', ''),
                authors=authors,
                abstract=bibjson.get('abstract'),
                year=int(bibjson['year']) if bibjson.get('year') else None,
                venue=journal,
                publisher=bibjson.get('journal', {}).get('publisher'),
                keywords=keywords,
                is_open_access=True,  # Everything in DOAJ is OA
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source='doaj',
                score=0.0
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse DOAJ article: {e}")
            return None
