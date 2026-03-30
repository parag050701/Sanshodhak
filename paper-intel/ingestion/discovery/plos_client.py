"""PLOS Search API client."""
import logging
import time
from typing import Optional

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class PLOSClient(BaseAPIClient):
    """Client for PLOS Search API (Solr-backed)."""

    def __init__(self):
        super().__init__(
            base_url="https://api.plos.org",
            rate_limit_delay=0.2,
            timeout=30,
        )

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            fq = []
            if min_year:
                fq.append(f"publication_date:[{min_year}-01-01T00:00:00Z TO *]")

            params = {
                "q": f"title:{query} OR abstract:{query}",
                "rows": min(limit, 100),
                "wt": "json",
                "fl": "id,title,author,abstract,publication_date,journal,article_type,counter_total_all",
                "sort": "counter_total_all desc",
            }
            if fq:
                params["fq"] = fq

            response = self.get("/search", params=params)
            if not response or response.status_code != 200:
                return SearchResult(
                    source="plos",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            data = response.json()
            response_obj = data.get("response", {})
            docs = response_obj.get("docs", [])

            papers = []
            for doc in docs:
                paper = self._parse_doc(doc)
                if paper:
                    papers.append(paper)

            return SearchResult(
                source="plos",
                query=query,
                papers=papers,
                total_found=response_obj.get("numFound", len(docs)),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"PLOS search error: {e}")
            return SearchResult(
                source="plos",
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
            title_raw = doc.get("title")
            title = title_raw[0] if isinstance(title_raw, list) and title_raw else (title_raw or "")
            title = title.strip()
            if not title:
                return None

            doi = (doc.get("id") or "").replace("info:doi/", "") or None
            authors = doc.get("author") or []
            abstract = None
            abs_raw = doc.get("abstract")
            if isinstance(abs_raw, list) and abs_raw:
                abstract = " ".join(abs_raw)
            elif isinstance(abs_raw, str):
                abstract = abs_raw

            year = None
            pub_date = doc.get("publication_date")
            if pub_date and len(pub_date) >= 4:
                try:
                    year = int(pub_date[:4])
                except ValueError:
                    year = None

            citations = int(doc.get("counter_total_all", 0) or 0)
            pdf_url = f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable" if doi else None

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                venue=doc.get("journal"),
                citations=citations,
                is_open_access=True,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source="plos",
                score=citations / 100.0 if citations else 0.0,
            )
        except Exception as e:
            logger.debug(f"PLOS parse error: {e}")
            return None
