"""Text cleaning and normalization utilities."""
import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean extracted text.
    
    Fixes:
    - Hyphenation across line breaks
    - Extra whitespace
    - Common OCR errors
    - Headers/footers (basic)
    """
    if not text:
        return ""
    
    # Fix hyphenation
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common ligatures
    text = text.replace('ﬁ', 'fi')
    text = text.replace('ﬂ', 'fl')
    text = text.replace('ﬀ', 'ff')
    
    # Remove page numbers (basic heuristic)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def segment_sections(text: str) -> dict:
    """
    Segment text into sections (basic heuristic).
    
    Returns:
        Dict with keys: abstract, introduction, methods, results, conclusion, references
    """
    sections = {}
    
    # Define section patterns
    patterns = {
        'abstract': r'(?i)abstract[\s\n]+(.+?)(?=\n\s*(?:introduction|1\s+introduction))',
        'introduction': r'(?i)(?:introduction|1\s+introduction)[\s\n]+(.+?)(?=\n\s*(?:methods|methodology|2\s+))',
        'methods': r'(?i)(?:methods|methodology)[\s\n]+(.+?)(?=\n\s*(?:results|3\s+))',
        'results': r'(?i)(?:results)[\s\n]+(.+?)(?=\n\s*(?:discussion|conclusion|4\s+))',
        'conclusion': r'(?i)(?:conclusion|discussion)[\s\n]+(.+?)(?=\n\s*(?:references|bibliography))',
        'references': r'(?i)(?:references|bibliography)[\s\n]+(.+?)$',
    }
    
    for section_name, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section_name] = clean_text(match.group(1))
    
    return sections
