"""ARC-SIEVE client for academic papers."""
import logging
import os
from typing import Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata, SearchResult
import time

logger = logging.getLogger(__name__)


class ArcSieveClient(BaseAPIClient):
    """
    Client for ARC-SIEVE (PLACEHOLDER - DISABLED BY DEFAULT).

    This is a placeholder client that returns no results.
    It is disabled by default and only runs if ENABLE_PLACEHOLDER_SOURCES=true.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Placeholder URL - update with actual ARC-SIEVE endpoint
        super().__init__(
            base_url="https://arc-sieve.example.com/api",
            rate_limit_delay=1.0,
            timeout=30
        )
        self.api_key = api_key
        self._enabled = os.getenv("ENABLE_PLACEHOLDER_SOURCES", "").lower() == "true"

    def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search ARC-SIEVE for papers.

        Note: This is a placeholder. Set ENABLE_PLACEHOLDER_SOURCES=true to enable.
        """
        start_time = time.time()

        if not self._enabled:
            return SearchResult(
                source='arc_sieve',
                query=query,
                papers=[],
                total_found=0,
                fetched_count=0,
                success=False,
                error="DISABLED - Set ENABLE_PLACEHOLDER_SOURCES=true to enable",
                search_time_ms=(time.time() - start_time) * 1000
            )

        logger.warning("ARC-SIEVE client is a placeholder - no real implementation")

        return SearchResult(
            source='arc_sieve',
            query=query,
            papers=[],
            total_found=0,
            fetched_count=0,
            success=False,
            error="Not implemented - placeholder client",
            search_time_ms=(time.time() - start_time) * 1000
        )

    def get_metadata(self, identifier: str) -> Optional[PaperMetadata]:
        """Get metadata for a specific paper."""
        if not self._enabled:
            return None
        logger.warning("ARC-SIEVE metadata lookup not implemented")
        return None

    def get_pdf_url(self, identifier: str) -> Optional[str]:
        """Get PDF URL for a paper."""
        if not self._enabled:
            return None
        logger.warning("ARC-SIEVE PDF URL lookup not implemented")
        return None
