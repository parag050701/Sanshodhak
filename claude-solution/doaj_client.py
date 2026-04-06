"""DOAJ (Directory of Open Access Journals) API client — fixed v3 query format."""
import logging
import time
from typing import Optional
from urllib.parse import quote

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class DOAJClient(BaseAPIClient):
    """
    Client for DOAJ API v3.

    Fix summary vs. original:
    - Query must be URL-encoded and placed in the path: /search/articles/<query>
    - The `ref` parameter does NOT exist in v3; removed.
    - Sort param changed to `sort=bibjson.year:desc` (kept).
    - pageSize → correct param name retained (works in v3).
    - Year filter via Lucene: bibjson.year:[YYYY TO *]  (not >=YYYY).
    - 403/400 were caused by the old `ref` param and malformed year syntax.
    """

    def __init__(self):
        super().__init__(
            base_url="https://doaj.org/api/v3",
            rate_limit_delay=0.6,   # ~100 req/min
            timeout=30,
            source_name="doaj",
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> SearchResult:
        start_time = time.time()

        try:
            # Build Lucene query
            lucene_query = query
            if min_year:
                # Correct Lucene range syntax for DOAJ
                lucene_query = f'({query}) AND bibjson.year:[{min_year} TO *]'

            # v3: query goes in the path, NOT as a `q` parameter
            encoded_query = quote(lucene_query, safe='')
            endpoint = f'/search/articles/{encoded_query}'

            params = {
                'pageSize': min(limit, 100),
                'page': 1,
                'sort': 'bibjson.year:desc',
            }

            response = self.get(endpoint, params=params)

            if not response:
                # health already recorded by base_client
                return SearchResult(
                    source='doaj', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            papers = [
                p for p in (self._parse_article(r) for r in data.get('results', []))
                if p is not None
            ]

            return SearchResult(
                source='doaj', query=query,
                papers=papers,
                total_found=data.get('total', 0),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"DOAJ search error: {e}")
            return SearchResult(
                source='doaj', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_article(self, item: dict) -> Optional[PaperMetadata]:
        try:
            bibjson = item.get('bibjson', {})

            doi = None
            for identifier in bibjson.get('identifier', []):
                if identifier.get('type') == 'doi':
                    doi = identifier.get('id')
                    break

            authors = [
                a['name'] for a in bibjson.get('author', []) if a.get('name')
            ]

            keywords = []
            for kw in bibjson.get('keywords', []):
                if isinstance(kw, str):
                    keywords.append(kw.lower())
                elif isinstance(kw, dict) and kw.get('term'):
                    keywords.append(kw['term'].lower())

            pdf_url = None
            for link in bibjson.get('link', []):
                if link.get('type') == 'fulltext' and link.get('content_type') == 'PDF':
                    pdf_url = link.get('url')
                    break

            journal = bibjson.get('journal', {}).get('title')
            year_raw = bibjson.get('year')
            year = int(year_raw) if year_raw else None

            return PaperMetadata(
                doi=doi,
                title=bibjson.get('title', ''),
                authors=authors,
                abstract=bibjson.get('abstract'),
                year=year,
                venue=journal,
                publisher=bibjson.get('journal', {}).get('publisher'),
                keywords=keywords,
                is_open_access=True,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source='doaj',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse DOAJ article: {e}")
            return None
