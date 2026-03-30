"""SciELO API client via Europe PMC-style endpoint fallback."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class SciELOClient(BaseAPIClient):
    """Client for SciELO search endpoint."""

    def __init__(self):
        super().__init__(
            base_url="https://search.scielo.org",
            rate_limit_delay=0.3,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            params = {
                "q": query,
                "lang": "en",
                "count": min(limit, 100),
                "output": "site",
                "format": "json",
            }
            if min_year:
                params["from"] = str(min_year)

            response = self.get("/", params=params)
            if not response or response.status_code != 200:
                return SearchResult(
                    source="scielo",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            data = response.json() if response.text.strip().startswith("{") else {}
            docs = data.get("results", []) if isinstance(data, dict) else []

            papers = []
            for doc in docs[:limit]:
                paper = self._parse_doc(doc)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="scielo",
                query=query,
                papers=papers,
                total_found=len(docs),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"SciELO search error: {e}")
            return SearchResult(
                source="scielo",
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _parse_doc(self, doc: dict) -> Optional[PaperMetadata]:
        try:
            title = (doc.get("title") or "").strip()
            if not title:
                return None

            authors = doc.get("authors") or []
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(";") if a.strip()]

            year = None
            yr = doc.get("year") or doc.get("publication_year")
            if yr:
                try:
                    year = int(str(yr)[:4])
                except ValueError:
                    year = None

            doi = doc.get("doi")
            pdf_url = doc.get("pdf_url") or doc.get("url")

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=doc.get("abstract"),
                year=year,
                venue=doc.get("journal"),
                is_open_access=bool(pdf_url),
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source="scielo",
                score=0.0,
            )
        except Exception:
            return None
