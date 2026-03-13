import requests
import time
import re
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DOIPDFFetcher:
    """Fetch PDFs from DOIs using multiple Sci-Hub mirrors and fallback methods"""
    
    def __init__(self, output_dir: str = "papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Multiple Sci-Hub mirrors (these change frequently, update as needed)
        self.scihub_mirrors = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru",
            "https://sci-hub.ren",
            "https://sci-hub.wf",
        ]
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def clean_doi(self, doi: str) -> str:
        """Clean and normalize DOI"""
        doi = doi.strip()
        # Remove common prefixes
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        return doi
    
    def get_filename_from_doi(self, doi: str) -> str:
        """Generate a safe filename from DOI"""
        safe_name = re.sub(r'[^\w\-.]', '_', doi)
        return f"{safe_name}.pdf"
    
    def fetch_from_scihub(self, doi: str, mirror: str) -> Optional[bytes]:
        """Fetch PDF from a specific Sci-Hub mirror"""
        try:
            # Try direct PDF URL first
            pdf_url = f"{mirror}/{doi}"
            logger.info(f"Trying {mirror} for DOI: {doi}")
            
            response = self.session.get(pdf_url, timeout=30, allow_redirects=True)
            
            # Check if we got a PDF
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type and len(response.content) > 1000:
                logger.info(f"✓ PDF found at {mirror}")
                return response.content
            
            # If not direct PDF, try parsing the page for PDF link
            if response.status_code == 200:
                # Look for iframe or embed with PDF
                pdf_match = re.search(r'(https?://[^\s<>"]+\.pdf[^\s<>"]*)', response.text)
                if pdf_match:
                    pdf_link = pdf_match.group(1)
                    logger.info(f"Found PDF link: {pdf_link}")
                    
                    pdf_response = self.session.get(pdf_link, timeout=30)
                    if 'application/pdf' in pdf_response.headers.get('content-type', ''):
                        logger.info(f"✓ PDF downloaded from embedded link")
                        return pdf_response.content
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch from {mirror}: {str(e)}")
            return None
    
    def fetch_pdf(self, doi: str, retry_delay: float = 2.0) -> Optional[Path]:
        """
        Fetch PDF for a given DOI, trying all mirrors
        
        Args:
            doi: The DOI to fetch
            retry_delay: Delay between mirror attempts (seconds)
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        doi = self.clean_doi(doi)
        filename = self.get_filename_from_doi(doi)
        output_path = self.output_dir / filename
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"PDF already exists: {output_path}")
            return output_path
        
        # Try each mirror
        for mirror in self.scihub_mirrors:
            pdf_content = self.fetch_from_scihub(doi, mirror)
            
            if pdf_content:
                # Save PDF
                with open(output_path, 'wb') as f:
                    f.write(pdf_content)
                logger.info(f"✓ Saved PDF: {output_path}")
                return output_path
            
            # Wait before trying next mirror
            time.sleep(retry_delay)
        
        logger.error(f"✗ Failed to fetch PDF for DOI: {doi}")
        return None
    
    def batch_fetch(self, dois: List[str], delay: float = 3.0) -> dict:
        """
        Fetch multiple PDFs
        
        Args:
            dois: List of DOIs to fetch
            delay: Delay between requests (seconds)
            
        Returns:
            Dictionary with results: {'success': [...], 'failed': [...]}
        """
        results = {'success': [], 'failed': []}
        
        for i, doi in enumerate(dois, 1):
            logger.info(f"\n[{i}/{len(dois)}] Processing: {doi}")
            
            try:
                pdf_path = self.fetch_pdf(doi)
                
                if pdf_path:
                    results['success'].append({'doi': doi, 'path': str(pdf_path)})
                else:
                    results['failed'].append(doi)
                    
            except Exception as e:
                logger.error(f"Error processing {doi}: {str(e)}")
                results['failed'].append(doi)
            
            # Delay between requests to be respectful
            if i < len(dois):
                time.sleep(delay)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = DOIPDFFetcher(output_dir="papers")
    
    # Example DOIs
    dois = [
        "10.1038/nature12373",
        "10.1126/science.1249098",
        "10.1016/j.cell.2013.05.039",
    ]
    
    # Fetch single PDF
    print("=== Fetching single PDF ===")
    pdf_path = fetcher.fetch_pdf(dois[0])
    if pdf_path:
        print(f"Downloaded: {pdf_path}")
    
    # Batch fetch
    print("\n=== Batch fetching ===")
    results = fetcher.batch_fetch(dois, delay=3.0)
    
    print(f"\n✓ Success: {len(results['success'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\nFailed DOIs:")
        for doi in results['failed']:
            print(f"  - {doi}")