"""HAL (archives ouvertes) API client."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class HALClient(BaseAPIClient):
    """Client for HAL public Solr API."""

    def __init__(self):
        super().__init__(
            base_url="https://api.archives-ouvertes.fr",
            rate_limit_delay=0.2,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            q = f"(title_t:{query} OR abstract_t:{query})"
            if min_year:
                q += f" AND submittedDateY_i:[{min_year} TO *]"

            params = {
                "q": q,
                "rows": min(limit, 100),
                "fl": "docid,doiId_s,title_s,authFullName_s,abstract_s,submittedDateY_i,journalTitle_s,fileMain_s",
                "wt": "json",
                "sort": "submittedDateY_i desc",
            }

            response = self.get("/search/", params=params)
            if not response or response.status_code != 200:
                return SearchResult(
                    source="hal",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            papers = []
            for doc in docs:
                paper = self._parse_doc(doc)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="hal",
                query=query,
                papers=papers,
                total_found=data.get("response", {}).get("numFound", len(docs)),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"HAL search error: {e}")
            return SearchResult(
                source="hal",
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
            title = doc.get("title_s")
            if isinstance(title, list):
                title = title[0] if title else ""
            title = (title or "").strip()
            if not title:
                return None

            authors = doc.get("authFullName_s") or []
            if not isinstance(authors, list):
                authors = [str(authors)]

            doi = doc.get("doiId_s")
            if isinstance(doi, list):
                doi = doi[0] if doi else None

            pdf_url = doc.get("fileMain_s")
            if isinstance(pdf_url, list):
                pdf_url = pdf_url[0] if pdf_url else None

            year = doc.get("submittedDateY_i")
            try:
                year = int(year) if year is not None else None
            except Exception:
                year = None

            abstract = doc.get("abstract_s")
            if isinstance(abstract, list):
                abstract = abstract[0] if abstract else None

            venue = doc.get("journalTitle_s")
            if isinstance(venue, list):
                venue = venue[0] if venue else None

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                venue=venue,
                is_open_access=bool(pdf_url),
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source="hal",
                score=0.0,
            )
        except Exception:
            return None
