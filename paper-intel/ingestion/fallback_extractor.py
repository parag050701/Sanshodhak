"""
Fast fallback PDF text extractors.

Provides multiple extraction methods with fallback chain:
1. PyMuPDF (fastest, most reliable)
2. pdfplumber (good for tables)
3. PyPDF2 (lightweight fallback)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FallbackExtractor:
    """
    Multi-method PDF text extraction with intelligent fallback.
    
    Chain: PyMuPDF → pdfplumber → PyPDF2
    """
    
    def __init__(self):
        """Initialize available extractors."""
        self.available_methods = []
        
        # Check PyMuPDF (fitz)
        try:
            import fitz
            self.available_methods.append('pymupdf')
            logger.debug("PyMuPDF available")
        except ImportError:
            logger.warning("PyMuPDF not available - install: pip install pymupdf")
        
        # Check pdfplumber
        try:
            import pdfplumber
            self.available_methods.append('pdfplumber')
            logger.debug("pdfplumber available")
        except ImportError:
            logger.warning("pdfplumber not available - install: pip install pdfplumber")
        
        # Check PyPDF2
        try:
            import PyPDF2
            self.available_methods.append('pypdf2')
            logger.debug("PyPDF2 available")
        except ImportError:
            logger.warning("PyPDF2 not available - install: pip install PyPDF2")
        
        if not self.available_methods:
            logger.error("No PDF extraction libraries available!")
    
    def extract(
        self,
        pdf_path: Path,
        method: str = 'auto'
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Extract text from PDF with fallback chain.
        
        Args:
            pdf_path: Path to PDF file
            method: 'auto', 'pymupdf', 'pdfplumber', or 'pypdf2'
            
        Returns:
            Tuple of (success, text, metadata)
        """
        if method == 'auto':
            # Try methods in order of preference
            for method_name in self.available_methods:
                success, text, metadata = self._extract_with_method(pdf_path, method_name)
                if success and text and len(text) > 100:
                    return True, text, metadata
            
            return False, None, None
        else:
            return self._extract_with_method(pdf_path, method)
    
    def _extract_with_method(
        self,
        pdf_path: Path,
        method: str
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Extract using specific method."""
        try:
            if method == 'pymupdf':
                return self._extract_pymupdf(pdf_path)
            elif method == 'pdfplumber':
                return self._extract_pdfplumber(pdf_path)
            elif method == 'pypdf2':
                return self._extract_pypdf2(pdf_path)
            else:
                logger.error(f"Unknown method: {method}")
                return False, None, None
        except Exception as e:
            logger.warning(f"{method} extraction failed: {e}")
            return False, None, None
    
    def _extract_pymupdf(self, pdf_path: Path) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Extract with PyMuPDF (fastest, best quality)."""
        import fitz
        
        doc = fitz.open(pdf_path)
        
        # Extract text
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_parts.append(text)
        
        full_text = '\n\n'.join(text_parts)
        
        # Extract metadata
        metadata = {
            'extractor': 'pymupdf',
            'pages': len(doc),
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
        }
        
        doc.close()
        
        logger.debug(f"PyMuPDF extracted {len(full_text)} chars from {pdf_path.name}")
        return True, full_text, metadata
    
    def _extract_pdfplumber(self, pdf_path: Path) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Extract with pdfplumber (good for tables)."""
        import pdfplumber
        
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            metadata = {
                'extractor': 'pdfplumber',
                'pages': len(pdf.pages),
            }
        
        full_text = '\n\n'.join(text_parts)
        
        logger.debug(f"pdfplumber extracted {len(full_text)} chars from {pdf_path.name}")
        return True, full_text, metadata
    
    def _extract_pypdf2(self, pdf_path: Path) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Extract with PyPDF2 (lightweight fallback)."""
        import PyPDF2
        
        text_parts = []
        
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            metadata = {
                'extractor': 'pypdf2',
                'pages': len(reader.pages),
            }
            
            # Try to get metadata
            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title', '')
                metadata['author'] = reader.metadata.get('/Author', '')
        
        full_text = '\n\n'.join(text_parts)
        
        logger.debug(f"PyPDF2 extracted {len(full_text)} chars from {pdf_path.name}")
        return True, full_text, metadata


def extract_text_fast(pdf_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Quick extraction using best available method.
    
    Args:
        pdf_path: Path to PDF
        
    Returns:
        Tuple of (success, text)
    """
    extractor = FallbackExtractor()
    success, text, _ = extractor.extract(pdf_path)
    return success, text


if __name__ == "__main__":
    # Test extraction
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fallback_extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    extractor = FallbackExtractor()
    print(f"\nAvailable methods: {extractor.available_methods}\n")
    
    success, text, metadata = extractor.extract(pdf_path)
    
    if success:
        print(f"✅ Success!")
        print(f"Extractor: {metadata.get('extractor')}")
        print(f"Pages: {metadata.get('pages')}")
        print(f"Text length: {len(text)} chars")
        print(f"\nFirst 500 chars:\n{text[:500]}")
    else:
        print("❌ Extraction failed")
