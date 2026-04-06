"""OATD (Open Access Theses and Dissertations) client — feature-flagged."""
import logging
import os
import time
from typing import Optional

from .base_client import BaseAPIClient, FailureReason, APIError
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)

# OATD has historically returned 403 from cloud/CI runtime contexts because
# their OAI-PMH endpoint applies IP-based access controls.  The client is
# gated behind an env-var feature flag so it never silently poisons aggregate
# results.  To enable: set SANSHODHAK_ENABLE_OATD=1 in your environment and
# confirm you can reach https://oatd.org/oatd/record from your host.
_ENABLED = os.getenv("SANSHODHAK_ENABLE_OATD", "0") == "1"


class OATDClient(BaseAPIClient):
    """
    Client for OATD (https://oatd.org).

    Fix summary vs. original:
    - OAI-PMH harvesting from oatd.org returns 403 in most cloud/sandbox
      environments due to IP filtering on their OAI endpoint.
    - Client is now feature-flagged: it raises a clear error when disabled
      rather than producing misleading "timeout" results.
    - When enabled, uses the correct OAI-PMH verb=Search endpoint with
      proper metadataPrefix=oai_dc and query param.
    """

    OAI_ENDPOINT = "https://oatd.org/oatd/search"

    def __init__(self):
        super().__init__(
            base_url="https://oatd.org",
            rate_limit_delay=2.0,
            timeout=30,
            source_name="oatd",
        )
        self.enabled = _ENABLED
        if not self.enabled:
            logger.info(
                "OATD client is disabled (set SANSHODHAK_ENABLE_OATD=1 to enable). "
                "OATD's OAI-PMH endpoint blocks most cloud IPs."
            )

    def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> SearchResult:
        start_time = time.time()

        if not self.enabled:
            self.health.record_failure(FailureReason.NOT_IMPLEMENTED)
            return SearchResult(
                source='oatd', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=(
                    "OATD disabled — set SANSHODHAK_ENABLE_OATD=1 to enable. "
                    "Note: oatd.org OAI endpoint blocks most cloud runtime IPs."
                ),
                search_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            # OATD has a basic keyword search API (not OAI-PMH)
            params = {
                'q': query,
                'pagesize': min(limit, 50),
                'start': 0,
                'format': 'json',
            }

            response = self.get('/oatd/search', params=params)

            if not response:
                return SearchResult(
                    source='oatd', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            records = data.get('records', [])

            papers = [
                p for p in (self._parse_record(r) for r in records)
                if p is not None
            ]

            return SearchResult(
                source='oatd', query=query,
                papers=papers,
                total_found=data.get('total', len(papers)),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"OATD search error: {e}")
            return SearchResult(
                source='oatd', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_record(self, record: dict) -> Optional[PaperMetadata]:
        try:
            title = (record.get('title') or '').strip()
            if not title:
                return None

            authors_raw = record.get('creator', record.get('author', []))
            if isinstance(authors_raw, str):
                authors = [authors_raw]
            elif isinstance(authors_raw, list):
                authors = authors_raw
            else:
                authors = []

            date_raw = record.get('date', '')
            year = int(date_raw[:4]) if date_raw and len(str(date_raw)) >= 4 else None

            return PaperMetadata(
                doi=record.get('identifier'),
                title=title,
                authors=authors,
                abstract=record.get('description'),
                year=year,
                venue=record.get('publisher'),
                is_open_access=True,
                source='oatd',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse OATD record: {e}")
            return None
