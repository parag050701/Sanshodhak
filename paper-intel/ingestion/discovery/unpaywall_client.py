"""Unpaywall API client for open-access PDF discovery."""
import logging
from typing import Optional
from .base_client import BaseAPIClient
from ..models import PaperMetadata
import time

logger = logging.getLogger(__name__)


class UnpaywallClient(BaseAPIClient):
    """
    Client for Unpaywall API (https://unpaywall.org/products/api).
    
    Legal open-access PDF lookup by DOI.
    Free with email. Rate limit: 100k requests/day.
    """
    
    def __init__(self, email: str):
        if not email:
            raise ValueError("Unpaywall requires an email address")
        
        super().__init__(
            base_url="https://api.unpaywall.org/v2",
            rate_limit_delay=0.01,  # Very generous rate limit
            timeout=10
        )
        self.email = email
    
    def search(self, query: str, limit: int = 10, **kwargs):
        """Unpaywall doesn't support search, only DOI lookup."""
        raise NotImplementedError("Unpaywall only supports DOI lookup, not search")
    
    def get_pdf_url(self, doi: str) -> Optional[dict]:
        """
        Get open-access PDF URL for a DOI.
        
        Args:
            doi: DOI to lookup
            
        Returns:
            Dict with {pdf_url, is_oa, oa_status} or None
        """
        try:
            # Clean DOI
            doi = doi.strip().replace('https://doi.org/', '')
            
            response = self.get(f'/{doi}', params={'email': self.email})
            
            if not response or response.status_code != 200:
                return None
            
            data = response.json()
            
            is_oa = data.get('is_oa', False)
            if not is_oa:
                return None
            
            # Get best OA location
            best_oa = data.get('best_oa_location')
            if not best_oa:
                return None
            
            pdf_url = best_oa.get('url_for_pdf') or best_oa.get('url')
            if not pdf_url:
                return None
            
            return {
                'pdf_url': pdf_url,
                'is_oa': True,
                'oa_status': data.get('oa_status'),
                'host_type': best_oa.get('host_type'),
                'version': best_oa.get('version')
            }
            
        except Exception as e:
            logger.debug(f"Unpaywall lookup failed for {doi}: {e}")
            return None
    
    def enrich_paper(self, paper: PaperMetadata) -> PaperMetadata:
        """
        Enrich paper metadata with Unpaywall OA info.
        
        Args:
            paper: Paper with DOI
            
        Returns:
            Enriched paper (modifies in-place and returns)
        """
        if not paper.doi:
            return paper
        
        oa_info = self.get_pdf_url(paper.doi)
        if oa_info:
            paper.is_open_access = True
            paper.pdf_url = oa_info['pdf_url']
            paper.open_access_pdf_url = oa_info['pdf_url']
            
            # Add OA status to keywords if not present
            oa_status = oa_info.get('oa_status', 'open')
            if oa_status not in paper.keywords:
                paper.keywords.append(oa_status)
        
        return paper
