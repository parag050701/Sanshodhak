"""Europe PMC API client for open-access papers."""
import logging
from typing import Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class EuropePMCClient(BaseAPIClient):
    """
    Client for Europe PMC API.
    
    Access to life sciences literature.
    Free, no API key required.
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://www.ebi.ac.uk/europepmc/webservices/rest",
            rate_limit_delay=0.1,  # 10 req/s
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
        Search Europe PMC for papers.
        """
        start_time = time.time()
        
        try:
            search_query = query
            if min_year:
                search_query += f" AND FIRST_PDATE:[{min_year}-01-01 TO 3000-12-31]"
            
            params = {
                'query': search_query,
                'format': 'json',
                'resultType': 'core',
                'pageSize': min(limit, 1000)
            }
            
            response = self.get('/search', params=params)
            
            if not response or response.status_code != 200:
                return SearchResult(
                    source='europepmc',
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start_time) * 1000
                )
            
            data = response.json()
            
            result_list = data.get('resultList', {})
            results = result_list.get('result', [])
            
            papers = []
            for item in results[:limit]:
                paper = self._parse_article(item)
                if paper:
                    papers.append(paper)
            
            return SearchResult(
                source='europepmc',
                query=query,
                papers=papers,
                total_found=data.get('hitCount', 0),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Europe PMC search error: {e}")
            return SearchResult(
                source='europepmc',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
            
    def _parse_article(self, item: dict) -> Optional[PaperMetadata]:
        """Parse Europe PMC article JSON to PaperMetadata."""
        try:
            # Extract basic metadata
            doi = item.get('doi')
            title = item.get('title', '')
            abstract = item.get('abstractText')
            year = int(item.get('pubYear')) if item.get('pubYear') else None
            journal = item.get('journalTitle')
            citations = item.get('citedByCount', 0)
            
            # Extract authors
            authors = []
            author_list = item.get('authorList', {})
            for author in author_list.get('author', []):
                name = author.get('fullName')
                if name:
                    authors.append(name)
                    
            # Extract keywords
            keywords = []
            keyword_list = item.get('keywordList', {})
            for keyword in keyword_list.get('keyword', []):
                if isinstance(keyword, str):
                    keywords.append(keyword.lower())
            
            # Extract URLs
            pdf_url = None
            is_oa = item.get('isOpenAccess', 'N') == 'Y'
            
            fullTextUrlList = item.get('fullTextUrlList', {})
            for url_entry in fullTextUrlList.get('fullTextUrl', []):
                if url_entry.get('documentStyle') == 'pdf':
                    pdf_url = url_entry.get('url')
                    break
            
            paper = PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                venue=journal,
                keywords=keywords,
                is_open_access=is_oa,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url if is_oa else None,
                source='europepmc',
                score=citations / 100.0 if citations else 0.0
            )
            
            return paper
            
        except Exception as e:
            logger.warning(f"Failed to parse Europe PMC article: {e}")
            return None
