"""Zenodo REST API client — fixed query syntax."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class ZenodoClient(BaseAPIClient):
    """
    Client for Zenodo REST API (https://zenodo.org/api).

    Fix summary vs. original:
    - Do NOT mix Lucene field-prefixed queries with the simple `q` param.
      Passing `q=title:foo AND description:bar` causes 400.
    - Use plain keyword query in `q`; use dedicated params (type, subtype,
      access_right, communities) for filtering.
    - Year range: use `publication_date` param, not inline query syntax.
    - Removed invalid field-prefixed query construction that caused 400.
    """

    BASE = "https://zenodo.org/api"

    def __init__(self, access_token: Optional[str] = None):
        headers = {}
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'

        super().__init__(
            base_url=self.BASE,
            rate_limit_delay=0.5,
            timeout=30,
            headers=headers,
            source_name="zenodo",
        )
        self.access_token = access_token

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
            # Simple keyword query — NO field-prefixed Lucene syntax here.
            params: dict = {
                'q': query,                    # plain text, no `title:` prefix
                'size': min(limit, 1000),
                'sort': 'mostrecent',
                'type': 'publication',         # dedicated filter param
                'subtype': 'article',
                'status': 'published',
            }

            # Year range via dedicated publication_date filter (avoids 400)
            if min_year or max_year:
                lo = f'{min_year}-01-01' if min_year else '1900-01-01'
                hi = f'{max_year}-12-31' if max_year else '2099-12-31'
                params['publication_date'] = f'[{lo} TO {hi}]'

            if self.access_token:
                params['access_token'] = self.access_token

            response = self.get('/records', params=params)

            if not response:
                return SearchResult(
                    source='zenodo', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            hits = data.get('hits', {})
            papers = [
                p for p in (self._parse_record(r) for r in hits.get('hits', []))
                if p is not None
            ]

            return SearchResult(
                source='zenodo', query=query,
                papers=papers[:limit],
                total_found=hits.get('total', {}).get('value', len(papers)),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Zenodo search error: {e}")
            return SearchResult(
                source='zenodo', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_record(self, record: dict) -> Optional[PaperMetadata]:
        try:
            meta = record.get('metadata', {})

            title = meta.get('title', '').strip()
            if not title:
                return None

            doi = meta.get('doi') or record.get('doi')

            # Authors
            creators = meta.get('creators', [])
            authors = []
            for c in creators:
                name = c.get('name') or f"{c.get('given_name', '')} {c.get('family_name', '')}".strip()
                if name:
                    authors.append(name)

            # Year
            pub_date = meta.get('publication_date', '')
            year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

            # Keywords
            keywords = [kw.lower() for kw in meta.get('keywords', []) if isinstance(kw, str)]

            # PDF/file links
            pdf_url = None
            for f in record.get('files', []):
                if f.get('type') == 'pdf' or str(f.get('key', '')).endswith('.pdf'):
                    pdf_url = f.get('links', {}).get('self')
                    break

            is_oa = meta.get('access_right') in ('open', 'embargoed')

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=meta.get('description'),
                year=year,
                venue=meta.get('journal', {}).get('title') if isinstance(meta.get('journal'), dict) else None,
                publisher=meta.get('publisher'),
                keywords=keywords,
                is_open_access=is_oa,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url if is_oa else None,
                source='zenodo',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Zenodo record: {e}")
            return None
