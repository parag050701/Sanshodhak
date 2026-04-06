"""ArcSieve client — gated placeholder, never silently fails."""
import logging
import os
import time
from typing import Optional

from .base_client import BaseAPIClient, FailureReason
from ..models import PaperMetadata, SearchResult

logger = logging.getLogger(__name__)

# ArcSieve is a placeholder until the real endpoint is known.
# Gated behind env-var so it never pollutes health dashboards.
# To enable once endpoint is known: set SANSHODHAK_ENABLE_ARCSIEVE=1
_ENABLED = os.getenv("SANSHODHAK_ENABLE_ARCSIEVE", "0") == "1"
_BASE_URL = os.getenv("SANSHODHAK_ARCSIEVE_URL", "https://arc-sieve.example.com/api")


class ArcSieveClient(BaseAPIClient):
    """
    Client for ARC-SIEVE.

    Fix summary vs. original:
    - Original always raised NotImplementedError (or silently returned empty
      SearchResult with success=False), which showed up as mysterious failures.
    - Now gated behind SANSHODHAK_ENABLE_ARCSIEVE=1 and returns a clear
      human-readable error in the SearchResult when disabled.
    - Base URL is also configurable via SANSHODHAK_ARCSIEVE_URL so the client
      works once the real endpoint is available without a code change.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url=_BASE_URL,
            rate_limit_delay=1.0,
            timeout=30,
            source_name="arc_sieve",
        )
        self.api_key = api_key
        self.enabled = _ENABLED

        if not self.enabled:
            logger.info(
                "ArcSieve client is disabled (placeholder — set "
                "SANSHODHAK_ENABLE_ARCSIEVE=1 and SANSHODHAK_ARCSIEVE_URL "
                "once the endpoint is known)."
            )

    def search(self, query: str, limit: int = 10, **kwargs) -> SearchResult:
        start_time = time.time()

        if not self.enabled:
            self.health.record_failure(FailureReason.NOT_IMPLEMENTED)
            return SearchResult(
                source='arc_sieve', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=(
                    "ArcSieve is a placeholder client. "
                    "Set SANSHODHAK_ENABLE_ARCSIEVE=1 and SANSHODHAK_ARCSIEVE_URL "
                    "once the real endpoint is available."
                ),
                search_time_ms=(time.time() - start_time) * 1000,
            )

        # --- Real implementation when endpoint is known ---
        try:
            params = {'q': query, 'limit': min(limit, 100)}
            if self.api_key:
                params['api_key'] = self.api_key

            response = self.get('/search', params=params)

            if not response:
                return SearchResult(
                    source='arc_sieve', query=query, papers=[],
                    total_found=0, fetched_count=0, success=False,
                    error="Request failed (see base_client logs for reason)",
                    search_time_ms=(time.time() - start_time) * 1000,
                )

            data = response.json()
            papers = [
                p for p in (self._parse_item(i) for i in data.get('results', []))
                if p is not None
            ]

            return SearchResult(
                source='arc_sieve', query=query,
                papers=papers,
                total_found=data.get('total', len(papers)),
                fetched_count=len(papers),
                success=True,
                search_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"ArcSieve search error: {e}")
            return SearchResult(
                source='arc_sieve', query=query, papers=[],
                total_found=0, fetched_count=0, success=False,
                error=str(e),
                search_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_item(self, item: dict) -> Optional[PaperMetadata]:
        try:
            title = (item.get('title') or '').strip()
            if not title:
                return None
            return PaperMetadata(
                doi=item.get('doi'),
                title=title,
                authors=item.get('authors', []),
                abstract=item.get('abstract'),
                year=item.get('year'),
                source='arc_sieve',
                score=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse ArcSieve item: {e}")
            return None

    def get_metadata(self, identifier: str) -> Optional[PaperMetadata]:
        if not self.enabled:
            logger.warning("ArcSieve disabled — metadata lookup skipped.")
            return None
        response = self.get(f'/metadata/{identifier}')
        if not response:
            return None
        return self._parse_item(response.json())

    def get_pdf_url(self, identifier: str) -> Optional[str]:
        if not self.enabled:
            return None
        response = self.get(f'/pdf/{identifier}')
        if not response:
            return None
        return response.json().get('url')
