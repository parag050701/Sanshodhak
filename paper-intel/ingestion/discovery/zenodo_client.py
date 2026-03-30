"""Zenodo API client for paper and preprint discovery."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class ZenodoClient(BaseAPIClient):
    """Client for Zenodo Records API."""

    def __init__(self):
        super().__init__(
            base_url="https://zenodo.org/api",
            rate_limit_delay=0.2,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            q = f"metadata.title:{query} OR metadata.description:{query}"
            if min_year:
                q += f" AND metadata.publication_date:[{min_year}-01-01 TO *]"

            params = {
                "q": q,
                "size": min(limit, 100),
                "sort": "mostrecent",
            }
            response = self.get("/records", params=params)

            if not response or response.status_code != 200:
                return SearchResult(
                    source="zenodo",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            data = response.json()
            hits = data.get("hits", {})
            items = hits.get("hits", [])

            papers = []
            for item in items:
                paper = self._parse_record(item)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="zenodo",
                query=query,
                papers=papers,
                total_found=hits.get("total", len(items)) if isinstance(hits.get("total"), int) else len(items),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Zenodo search error: {e}")
            return SearchResult(
                source="zenodo",
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _parse_record(self, item: dict) -> Optional[PaperMetadata]:
        try:
            metadata = item.get("metadata", {})
            title = metadata.get("title", "").strip()
            if not title:
                return None

            creators = metadata.get("creators", [])
            authors = [c.get("name", "").strip() for c in creators if c.get("name")]

            year = None
            date = metadata.get("publication_date")
            if date and len(date) >= 4:
                try:
                    year = int(date[:4])
                except ValueError:
                    year = None

            doi = item.get("doi") or metadata.get("doi")
            journal = metadata.get("journal_title") or metadata.get("publication_title")

            files = item.get("files", [])
            pdf_url = None
            for f in files:
                key = (f.get("key") or "").lower()
                if key.endswith(".pdf"):
                    links = f.get("links", {})
                    pdf_url = links.get("self")
                    break

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=metadata.get("description"),
                year=year,
                venue=journal,
                publisher="Zenodo",
                keywords=[k.get("term", "").lower() for k in metadata.get("keywords", []) if isinstance(k, dict) and k.get("term")],
                is_open_access=bool(pdf_url),
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source="zenodo",
                score=0.0,
            )
        except Exception as e:
            logger.debug(f"Zenodo parse error: {e}")
            return None
