"""OpenAIRE API client for publication discovery."""
import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote_plus

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class OpenAIREClient(BaseAPIClient):
    """Client for OpenAIRE Search API (XML)."""

    def __init__(self):
        super().__init__(
            base_url="https://api.openaire.eu",
            rate_limit_delay=0.4,
            timeout=40,
            headers={"Accept": "application/xml"},
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> SearchResult:
        start = time.time()
        try:
            final_query = query
            endpoint = f"/search/publications?keywords={quote_plus(final_query)}&size={min(limit, 100)}"
            response = self.get(endpoint)

            if not response or response.status_code != 200:
                return SearchResult(
                    source="openaire",
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            root = ET.fromstring(response.text)
            papers = []
            for result in root.findall(".//{*}result")[:limit]:
                paper = self._parse_result(result)
                if paper:
                    papers.append(paper)

            total_found = len(root.findall(".//{*}result"))
            return SearchResult(
                source="openaire",
                query=query,
                papers=papers,
                total_found=total_found,
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"OpenAIRE search error: {e}")
            return SearchResult(
                source="openaire",
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _parse_result(self, result: ET.Element) -> Optional[PaperMetadata]:
        try:
            title = self._extract_text(result, ["title", "maintitle"])
            if not title:
                return None

            doi = self._extract_text(result, ["doi", "pid"]) 
            year = None
            year_raw = self._extract_text(result, ["dateofacceptance", "publicationdate"])
            if year_raw and len(year_raw) >= 4:
                try:
                    year = int(year_raw[:4])
                except ValueError:
                    year = None

            authors = [
                node.text.strip()
                for node in result.findall(".//{*}creator")
                if node.text and node.text.strip()
            ]

            venue = self._extract_text(result, ["journal", "collectedfrom"])
            abstract = self._extract_text(result, ["description", "abstract"])
            pdf_url = self._extract_pdf(result)

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
                source="openaire",
                score=0.0,
            )
        except Exception as e:
            logger.debug(f"OpenAIRE parse error: {e}")
            return None

    def _extract_text(self, node: ET.Element, tags: list[str]) -> Optional[str]:
        for tag in tags:
            found = node.find(f".//{{*}}{tag}")
            if found is not None and found.text and found.text.strip():
                return found.text.strip()
        return None

    def _extract_pdf(self, node: ET.Element) -> Optional[str]:
        for tag in ["bestaccessright", "webresource", "url", "fulltext"]:
            for found in node.findall(f".//{{*}}{tag}"):
                text = (found.text or "").strip()
                if text.startswith("http") and ("pdf" in text.lower() or "download" in text.lower()):
                    return text
        return None
