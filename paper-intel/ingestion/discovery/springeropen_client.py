"""SpringerOpen metadata client via CrossRef filter."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class SpringerOpenClient(BaseAPIClient):
    """Fetch SpringerOpen/BMC journal papers from CrossRef by publisher filter."""

    def __init__(self):
        super().__init__(
            base_url="https://api.crossref.org",
            rate_limit_delay=0.15,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            filters = ["from-pub-date:2010-01-01", "publisher-name:Springer Science and Business Media LLC"]
            if min_year:
                filters.append(f"from-pub-date:{min_year}-01-01")

            params = {
                "query.title": query,
                "filter": ",".join(filters),
                "rows": min(limit, 100),
                "sort": "is-referenced-by-count",
                "order": "desc",
            }
            response = self.get("/works", params=params)

            if not response or response.status_code != 200:
                return SearchResult(
                    source="springeropen",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            items = response.json().get("message", {}).get("items", [])
            papers = []
            for item in items:
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="springeropen",
                query=query,
                papers=papers,
                total_found=len(items),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"SpringerOpen search error: {e}")
            return SearchResult(
                source="springeropen",
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _parse_item(self, item: dict) -> Optional[PaperMetadata]:
        try:
            title_arr = item.get("title") or []
            title = title_arr[0].strip() if title_arr else ""
            if not title:
                return None

            doi = item.get("DOI")
            authors = []
            for a in item.get("author", []):
                given = a.get("given", "").strip()
                family = a.get("family", "").strip()
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)

            year = None
            issued = item.get("issued", {}).get("date-parts", [])
            if issued and issued[0]:
                try:
                    year = int(issued[0][0])
                except Exception:
                    year = None

            citations = int(item.get("is-referenced-by-count", 0) or 0)
            journal = (item.get("container-title") or [None])[0]

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=item.get("abstract"),
                year=year,
                venue=journal,
                publisher=item.get("publisher"),
                citations=citations,
                is_open_access=item.get("license") is not None,
                source="springeropen",
                score=citations / 100.0 if citations else 0.0,
            )
        except Exception as e:
            logger.debug(f"SpringerOpen parse error: {e}")
            return None
