"""ERIC API client."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class ERICClient(BaseAPIClient):
    """Client for ERIC (Institute of Education Sciences) API."""

    def __init__(self):
        super().__init__(
            base_url="https://api.ies.ed.gov/eric/",
            rate_limit_delay=0.25,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            where_parts = []
            if min_year:
                where_parts.append(f"pubyear>={min_year}")
            where = " AND ".join(where_parts) if where_parts else None

            params = {
                "search": query,
                "rows": min(limit, 200),
                "format": "json",
            }
            if where:
                params["where"] = where

            response = self.get("/", params=params)
            if not response or response.status_code != 200:
                return SearchResult(
                    source="eric",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            data = response.json()
            records = data.get("response", {}).get("docs", []) if isinstance(data.get("response"), dict) else data.get("docs", [])

            papers = []
            for rec in records:
                paper = self._parse_record(rec)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="eric",
                query=query,
                papers=papers,
                total_found=len(records),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"ERIC search error: {e}")
            return SearchResult(
                source="eric",
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _parse_record(self, rec: dict) -> Optional[PaperMetadata]:
        try:
            title = (rec.get("title") or "").strip()
            if not title:
                return None

            authors = rec.get("author", []) if isinstance(rec.get("author"), list) else []
            year = None
            pubyear = rec.get("pubyear")
            if pubyear:
                try:
                    year = int(pubyear)
                except ValueError:
                    year = None

            doi = rec.get("doi")
            if isinstance(doi, list):
                doi = doi[0] if doi else None

            url = rec.get("fulltext") or rec.get("url")
            if isinstance(url, list):
                url = url[0] if url else None

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=rec.get("abstract"),
                year=year,
                venue=rec.get("source"),
                publisher="ERIC",
                keywords=rec.get("descriptor", []) if isinstance(rec.get("descriptor"), list) else [],
                is_open_access=bool(url),
                pdf_url=url,
                open_access_pdf_url=url,
                source="eric",
                score=0.0,
            )
        except Exception:
            return None
