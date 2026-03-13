"""
PDF downloader with multi-source fallback strategy.

Download order:
1. Direct open-access URL (from OpenAlex, Unpaywall, etc.)
2. arXiv (if arXiv ID present)
3. CORE (if accessible)
4. Sci-Hub (legal gray area - use as last resort)
5. Anna's Archive / LibGen (if MD5 hash available)
"""
import logging
import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .scihub_client import SciHubClient
from .annasarchive_client import AnnasArchiveClient
from .arxiv_client import ArxivClient
from .doi_utils import generate_paper_filename, sanitize_filename
from ..models import PaperMetadata, DownloadResult

logger = logging.getLogger(__name__)


class PDFDownloader:
    """
    Multi-source PDF downloader with intelligent fallback.
    
    Tries legal/open sources first, falls back to gray-area sources if needed.
    """
    
    def __init__(
        self,
        output_dir: str = "papers",
        enable_scihub: bool = False,  # Disabled by default for legal reasons
        enable_libgen: bool = False,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize PDF downloader.
        
        Args:
            output_dir: Directory to save PDFs
            enable_scihub: Enable Sci-Hub fallback (use responsibly)
            enable_libgen: Enable LibGen/Anna's Archive fallback
            max_retries: Max retry attempts per source
            timeout: Download timeout in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_scihub = enable_scihub
        self.enable_libgen = enable_libgen
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize clients
        self.arxiv = ArxivClient()
        self.scihub = SciHubClient() if enable_scihub else None
        self.annas = AnnasArchiveClient() if enable_libgen else None
        
        # Session for direct downloads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_paper(
        self,
        paper: PaperMetadata,
        filename: Optional[str] = None
    ) -> DownloadResult:
        """
        Download PDF for a single paper using all available sources.
        
        Args:
            paper: Paper metadata
            filename: Optional custom filename (otherwise auto-generated)
            
        Returns:
            DownloadResult with success status and path
        """
        # Generate filename
        if not filename:
            filename = generate_paper_filename(paper)
        
        output_path = self.output_dir / filename
        paper_id = paper.get_primary_id() or paper.title[:50]
        
        # Check if already downloaded
        if output_path.exists() and self._validate_pdf(output_path):
            logger.info(f"✓ Already have: {filename}")
            return DownloadResult(
                paper_id=paper_id,
                success=True,
                pdf_path=str(output_path),
                source='cache',
                file_size_bytes=output_path.stat().st_size
            )
        
        logger.info(f"Downloading: {paper.title[:60]}...")
        
        attempts = []
        
        # Source 1: Direct open-access URL
        if paper.pdf_url or paper.open_access_pdf_url:
            url = paper.open_access_pdf_url or paper.pdf_url
            result = self._download_from_url(url, output_path, 'direct_oa')
            attempts.append({'source': 'direct_oa', 'success': result[0], 'url': url})
            
            if result[0]:
                return DownloadResult(
                    paper_id=paper_id,
                    success=True,
                    pdf_path=str(output_path),
                    source='direct_oa',
                    file_size_bytes=result[1],
                    attempts=attempts
                )
        
        # Source 2: arXiv
        if paper.arxiv_id:
            arxiv_url = self.arxiv.get_pdf_url(paper.arxiv_id)
            result = self._download_from_url(arxiv_url, output_path, 'arxiv')
            attempts.append({'source': 'arxiv', 'success': result[0], 'url': arxiv_url})
            
            if result[0]:
                return DownloadResult(
                    paper_id=paper_id,
                    success=True,
                    pdf_path=str(output_path),
                    source='arxiv',
                    file_size_bytes=result[1],
                    attempts=attempts
                )
        
        # Source 3: Sci-Hub (if enabled and DOI available)
        if self.enable_scihub and self.scihub and paper.doi:
            try:
                result = self.scihub.get_pdf(paper.doi)
                if result:
                    pdf_content, mirror = result
                    with open(output_path, 'wb') as f:
                        f.write(pdf_content)
                    
                    if self._validate_pdf(output_path):
                        logger.info(f"✓ Downloaded via Sci-Hub: {filename}")
                        attempts.append({'source': f'scihub_{mirror}', 'success': True})
                        return DownloadResult(
                            paper_id=paper_id,
                            success=True,
                            pdf_path=str(output_path),
                            source=f'scihub_{mirror}',
                            file_size_bytes=len(pdf_content),
                            attempts=attempts
                        )
                    else:
                        output_path.unlink()  # Remove invalid PDF
                
                attempts.append({'source': 'scihub', 'success': False})
            except Exception as e:
                logger.debug(f"Sci-Hub failed: {e}")
                attempts.append({'source': 'scihub', 'success': False, 'error': str(e)})
        
        # Source 4: Anna's Archive / LibGen (if enabled)
        if self.enable_libgen and self.annas and paper.doi:
            try:
                # Note: This requires knowing the MD5 hash, which we don't have
                # In a full implementation, you'd search first, then download
                logger.debug("LibGen download requires MD5 hash (not implemented)")
                attempts.append({'source': 'libgen', 'success': False, 'error': 'requires_md5'})
            except Exception as e:
                logger.debug(f"LibGen failed: {e}")
                attempts.append({'source': 'libgen', 'success': False, 'error': str(e)})
        
        # All sources failed
        logger.warning(f"✗ Failed to download: {paper.title[:60]}")
        return DownloadResult(
            paper_id=paper_id,
            success=False,
            error="All sources failed",
            attempts=attempts
        )
    
    def _download_from_url(
        self,
        url: str,
        output_path: Path,
        source_name: str
    ) -> Tuple[bool, int]:
        """
        Download PDF from a direct URL.
        
        Returns:
            (success: bool, file_size: int)
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True
                )
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    logger.debug(f"{source_name}: URL returned HTML, not PDF")
                    return False, 0
                
                response.raise_for_status()
                
                # Download
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Validate
                if self._validate_pdf(output_path):
                    file_size = output_path.stat().st_size
                    logger.info(f"✓ Downloaded via {source_name}: {output_path.name}")
                    return True, file_size
                else:
                    output_path.unlink()  # Remove invalid file
                    logger.debug(f"{source_name}: Downloaded file is not a valid PDF")
                    return False, 0
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.debug(f"{source_name} attempt {attempt+1} failed: {e}")
                    continue
                logger.debug(f"{source_name} failed after {self.max_retries} attempts: {e}")
                return False, 0
        
        return False, 0
    
    def _validate_pdf(self, path: Path) -> bool:
        """
        Validate that file is a PDF.
        
        Checks:
        - File exists
        - Size > 1 KB
        - Starts with %PDF magic bytes
        """
        if not path.exists():
            return False
        
        if path.stat().st_size < 1024:
            return False
        
        try:
            with open(path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except:
            return False
    
    def download_batch(
        self,
        papers: List[PaperMetadata],
        parallel: bool = True,
        max_workers: int = 5
    ) -> List[DownloadResult]:
        """
        Download PDFs for multiple papers.
        
        Args:
            papers: List of papers to download
            parallel: Download in parallel
            max_workers: Max parallel downloads
            
        Returns:
            List of DownloadResults
        """
        logger.info(f"Starting batch download of {len(papers)} papers...")
        
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_paper = {
                    executor.submit(self.download_paper, paper): paper
                    for paper in papers
                }
                
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    try:
                        result = future.result(timeout=self.timeout + 10)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error downloading {paper.title[:50]}: {e}")
                        results.append(DownloadResult(
                            paper_id=paper.get_primary_id() or paper.title,
                            success=False,
                            error=str(e)
                        ))
        else:
            for paper in papers:
                result = self.download_paper(paper)
                results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch download complete: {successful}/{len(papers)} successful")
        
        return results
    
    def close(self):
        """Close session."""
        self.session.close()
        if self.arxiv:
            self.arxiv.close()
        if self.scihub:
            self.scihub.close()
        if self.annas:
            self.annas.close()
