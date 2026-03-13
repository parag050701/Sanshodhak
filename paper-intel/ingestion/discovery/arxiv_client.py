"""arXiv API client for preprints."""
import logging
import re
from typing import Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ArxivClient(BaseAPIClient):
    """
    Client for arXiv API (https://arxiv.org/help/api).
    
    Preprint repository for physics, math, CS, etc.
    Free, no API key. Rate limit: 3 seconds between requests.
    """
    
    def __init__(self):
        super().__init__(
            base_url="http://export.arxiv.org/api",
            rate_limit_delay=3.0,  # 3s between requests per arXiv guidelines
            timeout=30
        )
    
    def search(
        self,
        query: str,
        limit: int = 10,
        sort_by: str = 'relevance',
        **kwargs
    ) -> SearchResult:
        """
        Search arXiv for preprints.
        
        Args:
            query: Search query (uses arXiv query syntax)
            limit: Maximum results
            sort_by: Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')
            
        Returns:
            SearchResult with papers
        """
        start_time = time.time()
        
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': limit,
                'sortBy': sort_by,
                'sortOrder': 'descending'
            }
            
            # arXiv API uses URL encoding
            url = f"{self.base_url}/query?{urllib.parse.urlencode(params)}"
            
            # Use urllib instead of requests for arXiv
            self._rate_limit()
            
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                content = response.read().decode('utf-8')
            
            # Parse Atom XML
            papers = self._parse_atom_feed(content)
            
            return SearchResult(
                source='arxiv',
                query=query,
                papers=papers[:limit],
                total_found=len(papers),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return SearchResult(
                source='arxiv',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_pdf_url(self, arxiv_id: str) -> str:
        """
        Get PDF URL for an arXiv ID.
        
        Args:
            arxiv_id: arXiv ID (e.g., '1706.03762' or 'arxiv:1706.03762')
            
        Returns:
            PDF URL
        """
        # Clean arxiv_id
        arxiv_id = arxiv_id.replace('arxiv:', '').replace('arXiv:', '')
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    def _parse_atom_feed(self, xml_content: str) -> list:
        """Parse arXiv Atom XML feed to PaperMetadata list."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', ns):
                try:
                    # Extract ID
                    id_url = entry.find('atom:id', ns).text
                    arxiv_id = id_url.split('/abs/')[-1]
                    
                    # Extract DOI if present
                    doi = None
                    doi_elem = entry.find('arxiv:doi', ns)
                    if doi_elem is not None:
                        doi = doi_elem.text
                    
                    # Title
                    title = entry.find('atom:title', ns).text.strip()
                    
                    # Abstract
                    abstract = entry.find('atom:summary', ns).text.strip()
                    
                    # Authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name = author.find('atom:name', ns).text
                        authors.append(name)
                    
                    # Published date
                    published = entry.find('atom:published', ns).text
                    year = int(published[:4])
                    
                    # Categories (as keywords)
                    keywords = []
                    primary_cat = entry.find('arxiv:primary_category', ns)
                    if primary_cat is not None:
                        keywords.append(primary_cat.get('term'))
                    
                    for category in entry.findall('atom:category', ns):
                        cat_term = category.get('term')
                        if cat_term not in keywords:
                            keywords.append(cat_term)
                    
                    # PDF URL
                    pdf_url = self.get_pdf_url(arxiv_id)
                    
                    paper = PaperMetadata(
                        doi=doi,
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        year=year,
                        keywords=keywords,
                        is_open_access=True,
                        pdf_url=pdf_url,
                        open_access_pdf_url=pdf_url,
                        source='arxiv',
                        score=1.0
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse arXiv entry: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to parse arXiv feed: {e}")
        
        return papers
