"""OpenAIRE Graph REST API client — replaces broken OAI-PMH implementation."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class OpenAIREClient(BaseAPIClient):
    """
    Client for OpenAIRE Graph REST API v1.
    https://graph.openaire.eu/develop/api.html

    Fix summary vs. original (OAI-PMH):
    - OAI-PMH endpoint requires a valid `set` or `metadataPrefix`, and many
      institutional feeds block external crawlers → 404/403.
    - OpenAIRE's stable *Graph* REST API is public, JSON-native, and requires
      no authentication for basic searches.
    - Endpoint: https://api.openaire.eu/search/researchProducts
    - Supports: keywords, publication year range, type filter, page size.
    """

    def __init__(self):
        super().__init__(
            base_url="https://api.openaire.eu",
            rate_limit_delay=0.3,   # ~3–4 req/s; generous public quota
            timeout=30,
            source_name="openaire",
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        **kwargs,
    ) -> SearchResult:
        start_time = time.time()

        try:
            params = {
                'keywords': query,
                'type': 'publications',
                'format': 'json',
                'size': min(limit, 100),
                'page': 1,
                'sortBy': 'resultdateofacceptance,descending',
            }

            if min_year:
                params['fromDateAccepted'] = f'{min_year}-01-01'
            if max_year:
                params['toDateAccepted'] = f'{max_year}-12-31'

            response = self.get('/search/researchProducts', params=params)

            if not response:
                return SearchResult(
                    source='openaire', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            response_body = data.get('response', {})
            results_wrap = response_body.get('results', {}) or {}
            raw_results = results_wrap.get('result', []) or []

            # API may return a single dict instead of list when size=1
            if isinstance(raw_results, dict):
                raw_results = [raw_results]

            papers = [
                p for p in (self._parse_result(r) for r in raw_results)
                if p is not None
            ]

            total_str = (response_body.get('header', {}) or {}).get('total', {})
            try:
                total = int(total_str.get('$', 0)) if isinstance(total_str, dict) else int(total_str)
            except (TypeError, ValueError):
                total = len(papers)

            return SearchResult(
                source='openaire', query=query,
                papers=papers[:limit],
                total_found=total,
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"OpenAIRE search error: {e}")
            return SearchResult(
                source='openaire', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_result(self, result: dict) -> Optional[PaperMetadata]:
        try:
            meta = result.get('metadata', {})
            entity = meta.get('oaf:entity', {})
            pub = entity.get('oaf:result', {})

            # Title
            titles = pub.get('title', [])
            if isinstance(titles, dict):
                titles = [titles]
            title_obj = next((t for t in titles if t.get('@classid') == 'main title'), titles[0] if titles else {})
            title = title_obj.get('$', '').strip() if isinstance(title_obj, dict) else str(title_obj).strip()
            if not title:
                return None

            # DOI
            doi = None
            pids = pub.get('pid', [])
            if isinstance(pids, dict):
                pids = [pids]
            for pid in pids:
                if isinstance(pid, dict) and pid.get('@classid') == 'doi':
                    doi = pid.get('$')
                    break

            # Authors
            creators = pub.get('creator', [])
            if isinstance(creators, dict):
                creators = [creators]
            authors = [c.get('$', '') for c in creators if isinstance(c, dict) and c.get('$')]

            # Year
            date_str = pub.get('dateofacceptance', {})
            if isinstance(date_str, dict):
                date_str = date_str.get('$', '')
            year = int(date_str[:4]) if date_str and len(str(date_str)) >= 4 else None

            # Abstract
            desc = pub.get('description', {})
            abstract = desc.get('$') if isinstance(desc, dict) else (desc or None)

            # Open-access
            best_oa = pub.get('bestaccessright', {})
            is_oa = isinstance(best_oa, dict) and best_oa.get('@classid', '') in ('OPEN', 'OPEN SOURCE')

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                is_open_access=is_oa,
                source='openaire',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse OpenAIRE result: {e}")
            return None
