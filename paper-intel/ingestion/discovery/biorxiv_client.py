"""bioRxiv / medRxiv metadata API client."""
import logging
import time
from datetime import datetime
from typing import Optional, List

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class BioRxivMedRxivClient(BaseAPIClient):
    """Client for Cold Spring Harbor bioRxiv/medRxiv JSON endpoints."""

    def __init__(self):
        super().__init__(
            base_url="https://api.biorxiv.org",
            rate_limit_delay=0.2,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            candidates = []
            candidates.extend(self._fetch_server("biorxiv", limit=min(limit * 3, 200), min_year=min_year))
            candidates.extend(self._fetch_server("medrxiv", limit=min(limit * 3, 200), min_year=min_year))

            query_l = query.lower()
            matched = []
            for item in candidates:
                text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
                if query_l in text:
                    matched.append(item)
                if len(matched) >= limit:
                    break

            papers = []
            for item in matched:
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="biorxiv_medrxiv",
                query=query,
                papers=papers,
                total_found=len(candidates),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"bioRxiv/medRxiv search error: {e}")
            return SearchResult(
                source="biorxiv_medrxiv",
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _fetch_server(self, server: str, limit: int, min_year: Optional[int]) -> List[dict]:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        start_date = f"{min_year}-01-01" if min_year else "2015-01-01"
        endpoint = f"/details/{server}/{start_date}/{end_date}/0"

        response = self.get(endpoint)
        if not response or response.status_code != 200:
            return []

        data = response.json()
        collection = data.get("collection", [])
        return collection[:limit]

    def _parse_item(self, item: dict) -> Optional[PaperMetadata]:
        try:
            title = (item.get("title") or "").strip()
            if not title:
                return None

            doi = item.get("doi")
            date = item.get("date", "")
            year = None
            if date and len(date) >= 4:
                try:
                    year = int(date[:4])
                except ValueError:
                    year = None

            author_str = item.get("authors", "")
            authors = [a.strip() for a in author_str.split(";") if a.strip()] if author_str else []

            pdf_url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf" if doi else None

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=year,
                venue=item.get("server"),
                publisher="Cold Spring Harbor Laboratory",
                is_open_access=True,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source="biorxiv_medrxiv",
                score=0.0,
            )
        except Exception as e:
            logger.debug(f"bioRxiv/medRxiv parse error: {e}")
            return None
