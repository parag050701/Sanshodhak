"""PDF to text extraction utilities."""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract raw text from PDF.
    
    Uses PyPDF2 or pdfplumber as fallback to GROBID.
    
    Args:
        pdf_path: Path to PDF
        
    Returns:
        Extracted text or None
    """
    try:
        # Try pdfplumber first (better quality)
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text.strip()
            
    except ImportError:
        logger.warning("pdfplumber not installed, trying PyPDF2...")
        
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                return text.strip()
                
        except ImportError:
            logger.error("Neither pdfplumber nor PyPDF2 installed")
            return None
            
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return None


def save_text(text: str, output_path: Path):
    """Save extracted text to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
