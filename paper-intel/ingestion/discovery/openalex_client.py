"""OpenAlex API client for paper discovery."""
import logging
import os
from typing import List, Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class OpenAlexClient(BaseAPIClient):
    """
    Client for OpenAlex API (https://docs.openalex.org/).
    
    Open-source academic graph with 250M+ works.
    Free, no API key required. Polite pool: 100k requests/day.
    """
    
    def __init__(self, email: Optional[str] = None):
        super().__init__(
            base_url="https://api.openalex.org",
            rate_limit_delay=0.1,  # 10 req/s in polite pool
            timeout=30
        )
        _email = email or os.getenv("OPENALEX_EMAIL", "sanshodhak@research.local")
        self.session.headers['mailto'] = _email
    
    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        open_access_only: bool = False,
        **kwargs
    ) -> SearchResult:
        """
        Search OpenAlex for papers.
        
        Args:
            query: Search query (searches title, abstract, fulltext)
            limit: Maximum results
            min_year: Minimum publication year
            open_access_only: Only return OA papers
            
        Returns:
            SearchResult with papers
        """
        start_time = time.time()
        
        try:
            # Build filter
            filters = [f'title.search:{query}']
            
            if min_year:
                filters.append(f'from_publication_date:{min_year}-01-01')
            
            if open_access_only:
                filters.append('is_oa:true')
            
            params = {
                'filter': ','.join(filters),
                'per-page': min(limit, 200),  # OpenAlex max is 200
                'sort': 'cited_by_count:desc',
                'select': (
                    'id,doi,title,authorships,publication_year,primary_location,'
                    'open_access,best_oa_location,cited_by_count,concepts,'
                    'abstract_inverted_index'
                ),
            }
            
            response = self.get('/works', params=params)
            
            if not response or response.status_code != 200:
                return SearchResult(
                    source='openalex',
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
            for item in results[:limit]:
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            
            return SearchResult(
                source='openalex',
                query=query,
                papers=papers,
                total_found=data.get('meta', {}).get('count', 0),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"OpenAlex search error: {e}")
            return SearchResult(
                source='openalex',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def _parse_work(self, item: dict) -> Optional[PaperMetadata]:
        """Parse OpenAlex work JSON to PaperMetadata."""
        try:
            # Validate item is not None and has required fields
            if not item or not isinstance(item, dict):
                return None
            
            # Extract DOI
            doi = None
            if item.get('doi'):
                doi = item['doi'].replace('https://doi.org/', '')
            
            # Extract PDF URL from best OA location
            pdf_url = None
            is_oa = False
            oa_status_dict = item.get('open_access')
            if oa_status_dict and isinstance(oa_status_dict, dict):
                oa_status = oa_status_dict.get('oa_status')
                if oa_status and oa_status != 'closed':
                    is_oa = True
                    best_oa = item.get('best_oa_location')
                    if best_oa and isinstance(best_oa, dict) and best_oa.get('pdf_url'):
                        pdf_url = best_oa['pdf_url']
            
            # Extract authors
            authors = []
            authorships = item.get('authorships', [])
            if authorships and isinstance(authorships, list):
                for authorship in authorships:
                    if authorship and isinstance(authorship, dict):
                        author = authorship.get('author', {})
                        if author and isinstance(author, dict) and author.get('display_name'):
                            authors.append(author['display_name'])
            
            # Extract venue
            venue = None
            primary_location = item.get('primary_location')
            if primary_location and isinstance(primary_location, dict):
                source = primary_location.get('source')
                if source and isinstance(source, dict):
                    venue = source.get('display_name')
            
            # Extract concepts/keywords
            keywords = []
            concepts = item.get('concepts', [])
            if concepts and isinstance(concepts, list):
                for concept in concepts[:5]:  # Top 5
                    if concept and isinstance(concept, dict) and concept.get('display_name'):
                        keywords.append(concept['display_name'].lower())
            
            # Reconstruct abstract from inverted index
            abstract: Optional[str] = None
            aii = item.get('abstract_inverted_index')
            if aii and isinstance(aii, dict):
                try:
                    max_pos = max(pos for positions in aii.values() for pos in positions)
                    tokens = [''] * (max_pos + 1)
                    for word, positions in aii.items():
                        for pos in positions:
                            tokens[pos] = word
                    abstract = ' '.join(t for t in tokens if t)
                except Exception:
                    abstract = None

            paper = PaperMetadata(
                doi=doi,
                openalex_id=item.get('id'),
                title=item.get('title', ''),
                authors=authors,
                abstract=abstract,
                year=item.get('publication_year'),
                venue=venue,
                publisher=(primary_location.get('source') or {}).get('host_organization_name') if primary_location else None,
                keywords=keywords,
                citations=item.get('cited_by_count', 0),
                is_open_access=is_oa,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url if is_oa else None,
                source='openalex',
                score=item.get('cited_by_count', 0) / 100.0  # Normalize citations as score
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse OpenAlex work: {e}")
            return None
