"""
PubMed / NCBI E-utilities Client
=================================

Provides access to 35M+ biomedical and life-science papers via the
NCBI E-utilities REST API (free, no API key required for ≤3 req/s;
register an API key for 10 req/s).

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/

Endpoints used:
  esearch  → paper ID list for a query
  esummary → title, authors, year, DOI, journal per ID
  elink    → related/cited papers (optional)
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
from urllib.parse import quote

from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient(BaseAPIClient):
    """
    NCBI PubMed client — 35M+ biomedical papers.

    Free tier: 3 requests/second without API key.
    With NCBI_API_KEY env var: 10 requests/second.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        rate_delay = 0.1 if self.api_key else 0.34  # 10 or 3 req/s

        super().__init__(
            base_url=_NCBI_BASE,
            rate_limit_delay=rate_delay,
            timeout=30,
        )
        if self.api_key:
            self.session.headers["api_key"] = self.api_key

    # ------------------------------------------------------------------
    # Public interface (same signature as other discovery clients)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 20,
        min_year: Optional[int] = None,
        **kwargs,
    ) -> SearchResult:
        """
        Search PubMed and return structured PaperMetadata list.

        Args:
            query    : Free-text query (PubMed query syntax supported)
            limit    : Max results to return
            min_year : Optional publication year lower bound

        Returns:
            SearchResult with papers list
        """
        start = time.time()

        try:
            pmids = self._esearch(query, limit=limit, min_year=min_year)
            if not pmids:
                return SearchResult(
                    source="pubmed", query=query, papers=[],
                    total_found=0, fetched_count=0, success=True,
                    search_time_ms=(time.time() - start) * 1000,
                )

            papers = self._esummary(pmids)

            return SearchResult(
                source="pubmed", query=query, papers=papers,
                total_found=len(pmids), fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            logger.error(f"PubMed search error: {exc}")
            return SearchResult(
                source="pubmed", query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(exc),
                search_time_ms=(time.time() - start) * 1000,
            )

    # ------------------------------------------------------------------
    # E-utilities helpers
    # ------------------------------------------------------------------

    def _esearch(
        self,
        query: str,
        limit: int = 20,
        min_year: Optional[int] = None,
    ) -> List[str]:
        """Run esearch and return a list of PubMed IDs."""
        # Build date filter
        date_range = ""
        if min_year:
            date_range = f" AND {min_year}:3000[pdat]"

        params: Dict = {
            "db": "pubmed",
            "term": query + date_range,
            "retmax": min(limit, 200),
            "retmode": "json",
            "sort": "relevance",
            "usehistory": "n",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = self.get("/esearch.fcgi", params=params)
            if resp and resp.status_code == 200:
                data = resp.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                logger.info(f"PubMed esearch '{query}': {len(ids)} IDs")
                return ids
        except Exception as e:
            logger.warning(f"PubMed esearch failed: {e}")

        return []

    def _esummary(self, pmids: List[str]) -> List[PaperMetadata]:
        """Fetch document summaries for a list of PMIDs."""
        if not pmids:
            return []

        # Batch in groups of 20 to stay within rate limits
        papers: List[PaperMetadata] = []
        batch_size = 20

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i: i + batch_size]
            params: Dict = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            try:
                resp = self.get("/esummary.fcgi", params=params)
                if not resp or resp.status_code != 200:
                    continue

                data = resp.json()
                result_dict = data.get("result", {})
                uids = result_dict.get("uids", batch)

                for uid in uids:
                    item = result_dict.get(str(uid))
                    if not item or isinstance(item, str):
                        continue
                    paper = self._parse_summary(uid, item)
                    if paper:
                        papers.append(paper)

                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.warning(f"PubMed esummary batch error: {e}")

        return papers

    def _parse_summary(self, pmid: str, item: dict) -> Optional[PaperMetadata]:
        """Convert an esummary result dict to PaperMetadata."""
        try:
            title = item.get("title", "").strip()
            if not title:
                return None

            # Authors — list of dicts with 'name' key
            raw_authors = item.get("authors", [])
            authors = [a.get("name", "") for a in raw_authors if a.get("name")]

            # Year from pubdate e.g. "2024 Jan" or "2024"
            pubdate = item.get("pubdate", "")
            year: Optional[int] = None
            if pubdate:
                try:
                    year = int(pubdate.split()[0])
                except (ValueError, IndexError):
                    pass

            # DOI from articleids list
            doi: Optional[str] = None
            pdf_url: Optional[str] = None
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "").strip()
                if aid.get("idtype") == "pmc":
                    pmc_id = aid.get("value", "").strip()
                    if pmc_id:
                        pdf_url = (
                            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
                        )

            # Venue / journal
            venue = item.get("source", "") or item.get("fulljournalname", "")

            return PaperMetadata(
                doi=doi,
                pmid=pmid,
                title=title,
                authors=authors,
                abstract=None,  # esummary doesn't return full abstract; use efetch if needed
                year=year,
                venue=venue,
                citations=0,  # not provided by esummary
                is_open_access=pdf_url is not None,
                pdf_url=pdf_url,
                open_access_pdf_url=pdf_url,
                source="pubmed",
                score=0.0,
            )

        except Exception as e:
            logger.debug(f"PubMed parse error for PMID {pmid}: {e}")
            return None

    def fetch_abstract(self, pmid: str) -> Optional[str]:
        """
        Fetch full abstract for a single PMID via efetch (XML).
        Used optionally after esummary to enrich papers with abstracts.
        """
        params: Dict = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = self.get("/efetch.fcgi", params=params)
            if resp and resp.status_code == 200:
                root = ET.fromstring(resp.text)
                abstract_texts = root.findall(".//AbstractText")
                if abstract_texts:
                    return " ".join(
                        (el.text or "") for el in abstract_texts if el.text
                    )
        except Exception as e:
            logger.debug(f"PubMed efetch abstract failed for {pmid}: {e}")

        return None


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = PubMedClient()
    result = client.search(
        "retrieval augmented generation biomedical", limit=5, min_year=2022
    )
    print(f"\n✅ PubMed: {result.fetched_count} papers (total={result.total_found})")
    for p in result.papers:
        print(f"  {p.year} | {p.title[:70]} | OA={p.is_open_access}")
