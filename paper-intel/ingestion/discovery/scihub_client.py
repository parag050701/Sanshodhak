"""Sci-Hub client for PDF downloading (use responsibly)."""
import logging
import re
import random
from typing import Optional, Tuple
from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class SciHubClient(BaseAPIClient):
    """
    Client for Sci-Hub PDF resolution.
    
    WARNING: Use responsibly and legally. Sci-Hub operates in legal gray areas.
    Always try legal sources first (Unpaywall, OpenAlex, arXiv, etc.).
    """
    
    def __init__(self):
        # Rotating list of Sci-Hub mirrors
        self.mirrors = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru",
            "https://sci-hub.ren",
            "https://sci-hub.wf",
            "https://sci-hub.ee",
        ]
        
        super().__init__(
            base_url=self.mirrors[0],
            rate_limit_delay=3.0,  # Be polite
            timeout=30
        )
    
    def search(self, query: str, limit: int = 10, **kwargs):
        """Sci-Hub doesn't support search, only DOI resolution."""
        raise NotImplementedError("Sci-Hub only supports DOI/URL resolution")
    
    def get_pdf(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """
        Attempt to download PDF from Sci-Hub.
        
        Args:
            doi: DOI or paper URL
            
        Returns:
            Tuple of (pdf_content, mirror_used) or None
        """
        # Clean DOI
        doi = doi.strip().replace('https://doi.org/', '')
        
        # Try each mirror
        for mirror in random.sample(self.mirrors, len(self.mirrors)):
            try:
                self.base_url = mirror
                self._rate_limit()
                
                # Direct PDF URL attempt
                pdf_url = f"{mirror}/{doi}"
                response = self.session.get(pdf_url, timeout=self.timeout, allow_redirects=True)
                
                # Check if it's a PDF
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/pdf' in content_type and len(response.content) > 10000:
                    logger.info(f"[ok] PDF obtained from Sci-Hub: {mirror}")
                    return response.content, mirror
                
                # Parse HTML page for PDF link
                if response.status_code == 200 and 'text/html' in content_type:
                    pdf_link = self._extract_pdf_link(response.text, mirror)
                    if pdf_link:
                        pdf_response = self.session.get(pdf_link, timeout=self.timeout)
                        if 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                            if len(pdf_response.content) > 10000:
                                logger.info(f"[ok] PDF from embedded link: {mirror}")
                                return pdf_response.content, mirror
                
            except Exception as e:
                logger.debug(f"Sci-Hub mirror {mirror} failed: {e}")
                continue
        
        logger.warning(f"Sci-Hub: No PDF found for {doi}")
        return None
    
    def _extract_pdf_link(self, html: str, base_url: str) -> Optional[str]:
        """Extract PDF link from Sci-Hub HTML page."""
        patterns = [
            r'<iframe[^>]+src=["\']([^"\']+)["\']',
            r'<embed[^>]+src=["\']([^"\']+)["\']',
            r'location\.href=["\']([^"\']+\.pdf)["\']',
            r'<button[^>]+onclick=["\']location\.href=["\']([^"\']+)["\']',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                if match.endswith('.pdf') or '//sci-hub' in match or 'moscow.sci-hub' in match:
                    # Make absolute URL
                    if match.startswith('//'):
                        return f'https:{match}'
                    elif match.startswith('/'):
                        return f'{base_url}{match}'
                    elif match.startswith('http'):
                        return match
        
        return None
