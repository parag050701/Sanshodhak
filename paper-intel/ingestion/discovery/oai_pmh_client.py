"""Generic OAI-PMH client for OpenAIRE-compatible repositories."""
import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote_plus

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)


class OAIPMHClient(BaseAPIClient):
    """Minimal OAI-PMH ListRecords client with Dublin Core parsing."""

    def __init__(self, base_url: str, source_name: str = "oai_pmh"):
        super().__init__(
            base_url=base_url.rstrip("/"),
            rate_limit_delay=0.4,
            timeout=40,
            headers={"Accept": "application/xml"},
        )
        self.source_name = source_name

    def search(self, query: str, limit: int = 10, min_year: Optional[int] = None, **kwargs) -> SearchResult:
        start = time.time()
        try:
            endpoint = f"?verb=ListRecords&metadataPrefix=oai_dc"
            response = self.get(endpoint)

            if not response or response.status_code != 200:
                return SearchResult(
                    source=self.source_name,
                    query=query,
                    papers=[],
                    total_found=0,
                    fetched_count=0,
                    success=False,
                    error=f"HTTP {response.status_code if response else 'timeout'}",
                    search_time_ms=(time.time() - start) * 1000,
                )

            root = ET.fromstring(response.text)
            query_l = query.lower()
            papers = []

            for record in root.findall(".//{*}record"):
                paper = self._parse_record(record)
                if not paper:
                    continue

                if min_year and paper.year and paper.year < min_year:
                    continue

                text = f"{paper.title} {paper.abstract or ''}".lower()
                if query_l in text:
                    papers.append(paper)

                if len(papers) >= limit:
                    break

            return SearchResult(
                source=self.source_name,
                query=query,
                papers=papers,
                total_found=len(papers),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"{self.source_name} OAI-PMH search error: {e}")
            return SearchResult(
                source=self.source_name,
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

    def _parse_record(self, record: ET.Element) -> Optional[PaperMetadata]:
        try:
            title_node = record.find(".//{*}title")
            title = (title_node.text or "").strip() if title_node is not None else ""
            if not title:
                return None

            creators = [
                c.text.strip()
                for c in record.findall(".//{*}creator")
                if c.text and c.text.strip()
            ]

            description_node = record.find(".//{*}description")
            abstract = (description_node.text or "").strip() if description_node is not None and description_node.text else None

            date_node = record.find(".//{*}date")
            year = None
            if date_node is not None and date_node.text and len(date_node.text) >= 4:
                try:
                    year = int(date_node.text[:4])
                except ValueError:
                    year = None

            identifier_nodes = record.findall(".//{*}identifier")
            doi = None
            pdf_url = None
            for i in identifier_nodes:
                val = (i.text or "").strip()
                if not val:
                    continue
                low = val.lower()
                if "doi.org/" in low and not doi:
                    doi = val.split("doi.org/")[-1]
                elif low.startswith("10.") and not doi:
                    doi = val
                if val.startswith("http") and ("pdf" in low or "download" in low) and not pdf_url:
                    pdf_url = val

            return PaperMetadata(
                doi=doi,
                title=title,
                authors=creators,
                abstract=abstract,
                year=year,
                is_open_access=bool(pdf_url),
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source=self.source_name,
                score=0.0,
            )
        except Exception:
            return None
