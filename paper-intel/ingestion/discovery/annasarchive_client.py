"""Anna's Archive client for academic papers."""
import logging
from typing import Optional, Tuple
from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class AnnasArchiveClient(BaseAPIClient):
    """
    Client for Anna's Archive (https://annas-archive.org).
    
    Large shadow library aggregating LibGen, Z-Library, etc.
    Use responsibly and legally. Always try legal sources first.
    """
    
    def __init__(self):
        super().__init__(
            base_url="https://annas-archive.org",
            rate_limit_delay=2.0,
            timeout=30
        )
        
        # Known LibGen mirrors for fallback
        self.libgen_mirrors = [
            "https://libgen.is",
            "https://libgen.rs",
            "https://libgen.st",
            "https://libgen.li",
        ]
    
    def search(self, query: str, limit: int = 10, **kwargs):
        """Anna's Archive search is complex and requires rendering JS."""
        raise NotImplementedError("Anna's Archive search requires browser automation")
    
    def get_pdf_by_md5(self, md5: str) -> Optional[Tuple[bytes, str]]:
        """
        Download PDF from Anna's Archive using MD5 hash.
        
        Args:
            md5: MD5 hash of the file
            
        Returns:
            Tuple of (pdf_content, source) or None
        """
        try:
            # Try LibGen mirrors directly
            for mirror in self.libgen_mirrors:
                try:
                    self._rate_limit()
                    
                    # LibGen download URL pattern
                    url = f"{mirror}/get?md5={md5}"
                    
                    response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'application/pdf' in content_type and len(response.content) > 10000:
                            logger.info(f"✓ PDF from LibGen: {mirror}")
                            return response.content, f"libgen_{mirror}"
                
                except Exception as e:
                    logger.debug(f"LibGen mirror {mirror} failed: {e}")
                    continue
            
            logger.warning(f"Anna's Archive: No PDF found for MD5 {md5}")
            return None
            
        except Exception as e:
            logger.error(f"Anna's Archive error: {e}")
            return None
    
    def get_pdf_by_doi(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """
        Attempt to find and download PDF by DOI.
        
        Note: This is a simplified implementation. Full implementation
        would require searching Anna's Archive database first.
        """
        logger.info("Anna's Archive DOI lookup requires advanced search - not implemented")
        return None
