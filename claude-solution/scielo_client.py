"""SciELO API client — fixed to use stable Solr-based search endpoint."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class SCiELOClient(BaseAPIClient):
    """
    Client for SciELO (Scientific Electronic Library Online).
    https://search.scielo.org

    Fix summary vs. original:
    - The previous client called SciELO's internal WSGI/article API directly,
      which returns 403 from most runtime contexts.
    - SciELO exposes a stable public *Solr-backed* search endpoint at
      https://search.scielo.org/?output=json  that does NOT require auth.
    - Params: q (query), count (page size), from (offset), format=json,
      lang=en, filter[in_SciELO_CI][]=(*) is optional.
    - This endpoint is what the scielo.org search UI uses internally.
    """

    SEARCH_URL = "https://search.scielo.org/"

    def __init__(self):
        # We hit only one URL, base_url is just a placeholder
        super().__init__(
            base_url="https://search.scielo.org",
            rate_limit_delay=1.0,
            timeout=30,
            source_name="scielo",
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
            params: dict = {
                'q': query,
                'count': min(limit, 100),
                'from': 1,
                'output': 'json',
                'format': 'json',
                'lang': 'en',
                'document_type': 'ART',
            }

            if min_year:
                # SciELO Solr supports year range filter
                params['filter[year_cluster][]'] = str(min_year)

            response = self.get('/', params=params)

            if not response:
                return SearchResult(
                    source='scielo', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            hits = data.get('hits', {})
            raw_items = hits.get('hits', [])

            papers = [
                p for p in (self._parse_hit(h) for h in raw_items)
                if p is not None
            ]

            return SearchResult(
                source='scielo', query=query,
                papers=papers,
                total_found=hits.get('total', len(papers)),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"SciELO search error: {e}")
            return SearchResult(
                source='scielo', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_hit(self, hit: dict) -> Optional[PaperMetadata]:
        try:
            src = hit.get('_source', {})

            title_list = src.get('ti', src.get('title', []))
            if isinstance(title_list, list):
                title = title_list[0] if title_list else ''
            else:
                title = str(title_list)
            title = title.strip()
            if not title:
                return None

            doi = src.get('doi')

            # Authors: may be list of strings or dicts
            raw_authors = src.get('au', [])
            authors = []
            for a in raw_authors:
                if isinstance(a, str):
                    authors.append(a)
                elif isinstance(a, dict):
                    authors.append(a.get('name', ''))

            # Year
            year_raw = src.get('year', src.get('dp', ''))
            year = None
            if year_raw:
                try:
                    year = int(str(year_raw)[:4])
                except ValueError:
                    pass

            # Abstract
            abstract_data = src.get('ab', src.get('abstract', {}))
            if isinstance(abstract_data, dict):
                abstract = next(iter(abstract_data.values()), None)
            elif isinstance(abstract_data, str):
                abstract = abstract_data
            else:
                abstract = None

            # Journal
            journal = src.get('ta', src.get('journal', ''))
            if isinstance(journal, list):
                journal = journal[0] if journal else ''

            # PDF link
            pdf_url = src.get('pdf_url')
            if not pdf_url:
                pid = src.get('id') or src.get('pid')
                if pid:
                    pdf_url = f"https://www.scielo.br/j/{pid}/pdf"

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                venue=journal,
                is_open_access=True,   # SciELO is fully OA
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source='scielo',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse SciELO hit: {e}")
            return None
