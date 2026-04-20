"""
Text cleaning utilities for academic papers.

Handles:
- Hyphenated line breaks
- Paragraph normalization
- Unicode normalization
- Header/footer removal
- Whitespace normalization
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize extracted text from academic papers."""
    
    # Common header/footer patterns
    HEADER_FOOTER_PATTERNS = [
        r'^\d+\s+[A-Z][\w\s]+et al\.',  # "123 Smith et al."
        r'^\d+\s*$',  # Page numbers
        r'^Page \d+ of \d+',
        r'^\d+\s*/\s*\d+',
        r'^[A-Z\s]+\d{4}',  # "CONFERENCE 2024"
        r'©.*?\d{4}',  # Copyright notices
        r'doi:.*',  # DOI lines
        r'https?://.*',  # URLs
        r'arXiv:\d+\.\d+',  # arXiv identifiers
    ]
    
    # Section header patterns (for removal if needed)
    SECTION_HEADERS = [
        r'^(?:Abstract|Introduction|Related Work|Methodology|Methods|'
        r'Experiments|Results|Discussion|Conclusion|References|Acknowledgments?)\s*$'
    ]
    
    def __init__(
        self,
        remove_headers_footers: bool = True,
        fix_hyphenation: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True
    ):
        """
        Initialize text cleaner.
        
        Args:
            remove_headers_footers: Remove detected headers/footers
            fix_hyphenation: Fix hyphenated line breaks
            normalize_unicode: Normalize unicode characters
            normalize_whitespace: Normalize whitespace
        """
        self.remove_headers_footers = remove_headers_footers
        self.fix_hyphenation = fix_hyphenation
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        
        # Compile patterns
        self.header_footer_regex = [
            re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            for pattern in self.HEADER_FOOTER_PATTERNS
        ]
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning operations.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Fix hyphenated line breaks
        if self.fix_hyphenation:
            text = self._fix_hyphenation(text)
        
        # Step 3: Remove headers/footers
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)
        
        # Step 4: Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Step 5: Fix paragraph breaks
        text = self._fix_paragraphs(text)
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters.
        
        - Convert to NFC form
        - Replace common ligatures
        - Fix quotation marks
        """
        # Normalize to NFC (canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Replace ligatures
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            'ﬅ': 'st',
            'ﬆ': 'st',
            # Smart quotes
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '-': '-',
            '-': '-',
            # Mathematical symbols
            'x': 'x',
            '−': '-',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """
        Fix hyphenated line breaks.
        
        Examples:
            "compu-\nter" -> "computer"
            "state-of-the-art" -> "state-of-the-art" (keep)
        """
        # Fix hyphenated line breaks (word- \n word)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix soft hyphens
        text = text.replace('\u00AD', '')
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Remove common header and footer patterns.
        
        This is heuristic and conservative.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check against header/footer patterns
            is_header_footer = False
            for pattern in self.header_footer_regex:
                if pattern.match(line_stripped):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace.
        
        - Replace multiple spaces with single space
        - Replace tabs with spaces
        - Normalize line breaks
        """
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks (max 2 consecutive)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _fix_paragraphs(self, text: str) -> str:
        """
        Fix paragraph breaks.
        
        Heuristics:
        - Single line break within sentence -> space
        - Double line break -> paragraph break
        """
        # Split into paragraphs (double line break)
        paragraphs = text.split('\n\n')
        
        fixed_paragraphs = []
        for para in paragraphs:
            # Within paragraph, replace single line breaks with space
            # unless line ends with punctuation
            lines = para.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # If line ends with sentence-ending punctuation, keep break
                if line and line[-1] in '.!?:':
                    fixed_lines.append(line)
                # If it's the last line, keep it
                elif i == len(lines) - 1:
                    fixed_lines.append(line)
                # Otherwise, join with space
                else:
                    if fixed_lines:
                        fixed_lines[-1] += ' ' + line
                    else:
                        fixed_lines.append(line)
            
            if fixed_lines:
                fixed_paragraphs.append(' '.join(fixed_lines))
        
        return '\n\n'.join(fixed_paragraphs)
    
    def clean_title(self, title: str) -> str:
        """Clean paper title."""
        if not title:
            return ""
        
        title = self._normalize_unicode(title)
        title = title.strip()
        
        # Remove trailing punctuation
        title = re.sub(r'[.,:;]+$', '', title)
        
        return title
    
    def clean_abstract(self, abstract: str) -> str:
        """Clean paper abstract."""
        if not abstract:
            return ""
        
        abstract = self._normalize_unicode(abstract)
        abstract = self._normalize_whitespace(abstract)
        abstract = abstract.strip()
        
        # Remove "Abstract" prefix if present
        abstract = re.sub(r'^Abstract[:\s]+', '', abstract, flags=re.IGNORECASE)
        
        return abstract
    
    def clean_author_name(self, name: str) -> str:
        """Clean author name."""
        if not name:
            return ""
        
        name = self._normalize_unicode(name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        return name


def clean_text(
    text: str,
    remove_headers_footers: bool = True,
    fix_hyphenation: bool = True
) -> str:
    """
    Convenience function for quick text cleaning.
    
    Args:
        text: Text to clean
        remove_headers_footers: Remove headers/footers
        fix_hyphenation: Fix hyphenated line breaks
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(
        remove_headers_footers=remove_headers_footers,
        fix_hyphenation=fix_hyphenation
    )
    return cleaner.clean(text)


if __name__ == "__main__":
    # Test text cleaning
    test_text = """
    123 Smith et al.
    
    This is a test sen-
    tence with hyphen-
    ation.
    
    This is another paragraph with   extra    spaces
    and\t\ttabs.
    
    Page 42 of 100
    © 2024 IEEE
    """
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean(test_text)
    
    print("Original:")
    print(test_text)
    print("\nCleaned:")
    print(cleaned)
