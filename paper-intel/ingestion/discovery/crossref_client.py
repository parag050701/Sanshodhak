"""CrossRef API client for metadata discovery."""
import logging
import math
from typing import List, Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class CrossRefClient(BaseAPIClient):
    """
    Client for CrossRef API (https://api.crossref.org/).
    
    Metadata for 130M+ scholarly works with DOIs.
    No API key required. Polite pool: 50 req/s.
    """
    
    def __init__(self, email: Optional[str] = None):
        headers = {}
        if email:
            headers['User-Agent'] = f'SanshodhakBot/1.0 (mailto:{email})'
        
        super().__init__(
            base_url="https://api.crossref.org",
            rate_limit_delay=0.02,  # 50/s in polite pool
            timeout=30,
            headers=headers
        )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        enrich_abstracts: bool = False,
        **kwargs
    ) -> SearchResult:
        """
        Search CrossRef for papers.
        
        Args:
            query: Search query
            limit: Maximum results
            min_year: Minimum publication year
            max_year: Maximum publication year
            
        Returns:
            SearchResult with papers
        """
        start_time = time.time()
        
        try:
            # CrossRef supports pagination
            rows = min(100, limit)  # Max 1000 per request, but 100 is reasonable
            pages_needed = math.ceil(limit / rows)
            
            all_papers = []
            
            for page in range(pages_needed):
                params = {
                    'query': query,
                    'rows': rows,
                    'offset': page * rows,
                    'sort': 'relevance',
                    'filter': 'type:journal-article',
                }
                
                # Add year filters
                filters = ['type:journal-article']
                if min_year:
                    filters.append(f'from-pub-date:{min_year}')
                if max_year:
                    filters.append(f'until-pub-date:{max_year}')
                
                params['filter'] = ','.join(filters)
                
                response = self.get('/works', params=params)
                
                if not response or response.status_code != 200:
                    break
                
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                for item in items:
                    paper = self._parse_work(item)
                    if paper:
                        all_papers.append(paper)
                        if len(all_papers) >= limit:
                            break
                
                if len(all_papers) >= limit:
                    break
            
            final_papers = all_papers[:limit]

            # Optional abstract enrichment via DOI lookup.
            # CrossRef's search endpoint rarely returns abstracts; the /works/{doi}
            # endpoint is more reliable. Off by default to keep search latency low.
            if enrich_abstracts:
                for paper in final_papers:
                    if paper.doi and not paper.abstract:
                        enriched = self.lookup_doi(paper.doi)
                        if enriched and enriched.abstract:
                            paper.abstract = enriched.abstract
                        time.sleep(self.rate_limit_delay)

            return SearchResult(
                source='crossref',
                query=query,
                papers=final_papers,
                total_found=data.get('message', {}).get('total-results', 0),
                fetched_count=len(final_papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return SearchResult(
                source='crossref',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def lookup_doi(self, doi: str) -> Optional[PaperMetadata]:
        """
        Lookup metadata for a specific DOI.
        
        Args:
            doi: DOI to lookup
            
        Returns:
            PaperMetadata or None
        """
        try:
            response = self.get(f'/works/{doi}')
            
            if not response or response.status_code != 200:
                return None
            
            data = response.json()
            item = data.get('message')
            
            if item:
                return self._parse_work(item)
            
            return None
            
        except Exception as e:
            logger.error(f"CrossRef DOI lookup error: {e}")
            return None
    
    def _parse_work(self, item: dict) -> Optional[PaperMetadata]:
        """Parse CrossRef work JSON to PaperMetadata."""
        try:
            # Extract DOI
            doi = item.get('DOI')
            if not doi:
                return None
            
            # Extract title
            title_list = item.get('title', ['Untitled'])
            title = title_list[0] if title_list else 'Untitled'
            
            # Extract authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    authors.append(f"{given} {family}")
                elif 'name' in author:
                    authors.append(author['name'])
            
            # Extract year
            year = None
            date_parts = item.get('published-print', {}).get('date-parts', [])
            if not date_parts:
                date_parts = item.get('published-online', {}).get('date-parts', [])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
            
            # Extract journal
            journal = ''
            container = item.get('container-title', [''])
            if container:
                journal = container[0]
            
            # Extract abstract
            abstract = item.get('abstract', '')
            
            # Extract keywords/subjects
            keywords = []
            for subject_list in item.get('subject', []):
                if isinstance(subject_list, str):
                    keywords.extend([k.lower().strip() for k in subject_list.split(',')])
            
            # Extract publisher
            publisher = item.get('publisher')
            
            paper = PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract if abstract else None,
                year=year,
                venue=journal,
                publisher=publisher,
                keywords=keywords,
                citations=item.get('is-referenced-by-count', 0),
                source='crossref',
                score=item.get('score', 0.0)
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse CrossRef work: {e}")
            return None
