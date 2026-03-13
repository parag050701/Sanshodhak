"""Semantic Scholar API client."""
import logging
from typing import List, Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class SemanticScholarClient(BaseAPIClient):
    """
    Client for Semantic Scholar API (https://api.semanticscholar.org/).
    
    AI-powered paper search with 200M+ papers.
    Free tier: 100 requests/5min. Registered key: 1 req/sec.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        headers = {}
        if api_key:
            headers['x-api-key'] = api_key
        
        super().__init__(
            base_url="https://api.semanticscholar.org/graph/v1",
            rate_limit_delay=1.0 if api_key else 3.0,  # 1/s with key, slower without
            timeout=30,
            headers=headers
        )
        self.api_key = api_key
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> SearchResult:
        """
        Search Semantic Scholar for papers.
        
        Args:
            query: Search query
            limit: Maximum results (max 100)
            min_year: Minimum publication year
            fields: Fields to return
            
        Returns:
            SearchResult with papers
        """
        start_time = time.time()
        
        try:
            if fields is None:
                fields = [
                    'paperId', 'externalIds', 'title', 'abstract', 'year',
                    'authors', 'venue', 'citationCount', 'openAccessPdf',
                    'fieldsOfStudy'
                ]
            
            params = {
                'query': query,
                'limit': min(limit, 100),
                'fields': ','.join(fields)
            }
            
            if min_year:
                params['year'] = f'{min_year}-'
            
            response = self.get('/paper/search', params=params)
            
            if not response or response.status_code != 200:
                return SearchResult(
                    source='semantic_scholar',
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start_time) * 1000
                )
            
            data = response.json()
            results = data.get('data', [])
            
            papers = []
            for item in results:
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            
            return SearchResult(
                source='semantic_scholar',
                query=query,
                papers=papers,
                total_found=data.get('total', 0),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return SearchResult(
                source='semantic_scholar',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def _parse_paper(self, item: dict) -> Optional[PaperMetadata]:
        """Parse S2 paper JSON to PaperMetadata."""
        try:
            # Extract DOI
            doi = None
            external_ids = item.get('externalIds', {})
            if external_ids:
                doi = external_ids.get('DOI')
            
            # Extract authors
            authors = []
            for author in item.get('authors', []):
                if author.get('name'):
                    authors.append(author['name'])
            
            # Extract PDF URL
            pdf_url = None
            is_oa = False
            oa_pdf = item.get('openAccessPdf')
            if oa_pdf and oa_pdf.get('url'):
                pdf_url = oa_pdf['url']
                is_oa = True
            
            # Extract keywords from fields of study
            keywords = []
            fields = item.get('fieldsOfStudy', [])
            if fields:
                keywords = [f.lower() for f in fields if f]
            
            paper = PaperMetadata(
                doi=doi,
                arxiv_id=external_ids.get('ArXiv') if external_ids else None,
                pmid=external_ids.get('PubMed') if external_ids else None,
                s2_id=item.get('paperId'),
                title=item.get('title', ''),
                authors=authors,
                abstract=item.get('abstract'),
                year=item.get('year'),
                venue=item.get('venue'),
                keywords=keywords,
                citations=item.get('citationCount', 0),
                is_open_access=is_oa,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url if is_oa else None,
                source='semantic_scholar',
                score=item.get('citationCount', 0) / 100.0
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse S2 paper: {e}")
            return None
    
    def get_recommendations(self, paper_id: str, limit: int = 10) -> List[PaperMetadata]:
        """
        Get recommended/related papers for a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Max recommendations
            
        Returns:
            List of recommended papers
        """
        try:
            params = {
                'fields': 'paperId,externalIds,title,authors,year,citationCount',
                'limit': limit
            }
            
            response = self.get(f'/paper/{paper_id}/recommendations', params=params)
            
            if not response or response.status_code != 200:
                return []
            
            data = response.json()
            results = data.get('recommendedPapers', [])
            
            papers = []
            for item in results:
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"S2 recommendations error: {e}")
            return []
