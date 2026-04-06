"""SpringerOpen / Springer Nature Open Access API client — fixed query format."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class SpringerOpenClient(BaseAPIClient):
    """
    Client for Springer Nature's Open Access API.
    https://dev.springernature.com/

    Fix summary vs. original:
    - Original built a `filter` param by combining Crossref-style filters
      (e.g. `type:journal-article,from-pub-date:YYYY`) with Springer's API,
      which caused 400 — Springer uses a completely different filter syntax.
    - Springer's API uses `q` for full-text and separate params:
        openaccess:true  → in the `q` string or via dedicated constraint
        subject:<field>  → subject area filter
        date:<YYYY-YYYY> → publication year range
    - No API key = 10 req/min (free tier); with key = 5000 req/day.
    - Set SPRINGER_API_KEY env var for authenticated access.
    """

    import os as _os
    _API_KEY = _os.getenv("SPRINGER_API_KEY", "")

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("SPRINGER_API_KEY", "")

        super().__init__(
            base_url="https://api.springer.com",
            rate_limit_delay=6.0 if not self.api_key else 0.5,  # 10/min free
            timeout=30,
            source_name="springeropen",
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
            # Build Springer query string — use parentheses to avoid precedence issues
            q_parts = [f'({query})', 'openaccess:true']
            if min_year and max_year:
                q_parts.append(f'date:{min_year}-{max_year}')
            elif min_year:
                q_parts.append(f'date:{min_year}-{time.strftime("%Y")}')

            params: dict = {
                'q': ' AND '.join(q_parts),
                's': 1,                         # start record
                'p': min(limit, 25),            # page size (max 25 on free tier)
                'api_key': self.api_key or 'test',  # 'test' gives limited access
            }

            response = self.get('/metadata/json', params=params)

            if not response:
                return SearchResult(
                    source='springeropen', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            records = data.get('records', [])

            papers = [
                p for p in (self._parse_record(r) for r in records)
                if p is not None
            ]

            total = 0
            try:
                total = int(data.get('result', [{}])[0].get('total', 0))
            except (IndexError, TypeError, ValueError):
                total = len(papers)

            return SearchResult(
                source='springeropen', query=query,
                papers=papers,
                total_found=total,
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"SpringerOpen search error: {e}")
            return SearchResult(
                source='springeropen', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_record(self, rec: dict) -> Optional[PaperMetadata]:
        try:
            title = (rec.get('title') or '').strip()
            if not title:
                return None

            doi = rec.get('doi')

            # Authors: list of dicts with 'creator' key
            creators = rec.get('creators', [])
            authors = [c.get('creator', '') for c in creators if c.get('creator')]

            # Year
            pub_date = rec.get('publicationDate', rec.get('onlineDate', ''))
            year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

            # Abstract
            abstract = rec.get('abstract', '')

            # Journal
            journal = rec.get('publicationName', '')

            # OA flag
            is_oa = rec.get('openaccess', 'false').lower() == 'true'

            # PDF URL
            pdf_url = rec.get('url', [{}])[0].get('value') if rec.get('url') else None

            # Keywords
            kw_raw = rec.get('keyword', '')
            keywords = [k.strip().lower() for k in kw_raw.split(';') if k.strip()] if kw_raw else []

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract or None,
                year=year,
                venue=journal,
                publisher='Springer Nature',
                keywords=keywords,
                is_open_access=is_oa,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url if is_oa else None,
                source='springeropen',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Springer record: {e}")
            return None
