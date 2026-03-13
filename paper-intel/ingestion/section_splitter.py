"""
Section splitting for academic papers.

Splits cleaned paper text into canonical sections:
- introduction
- related_work
- methods
- experiments
- results
- discussion
- conclusion
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SectionMatch:
    """Represents a detected section heading."""
    name: str
    canonical_name: str
    start_pos: int
    confidence: float


class SectionSplitter:
    """
    Split academic paper into canonical sections.
    
    Uses heading-based heuristics with fallback strategies.
    """
    
    # Section name mappings to canonical names
    SECTION_MAPPINGS = {
        'introduction': 'introduction',
        'intro': 'introduction',
        'background': 'introduction',
        
        'related work': 'related_work',
        'relatedwork': 'related_work',
        'related works': 'related_work',
        'literature review': 'related_work',
        'previous work': 'related_work',
        'prior work': 'related_work',
        'state of the art': 'related_work',
        'research on': 'related_work',  # Added for patterns like "Research on X"
        
        'method': 'methods',
        'methods': 'methods',
        'methodology': 'methods',
        'approach': 'methods',
        'proposed method': 'methods',
        'our approach': 'methods',
        'system': 'methods',
        'model': 'methods',
        'algorithm': 'methods',
        'model architecture': 'methods',
        'architecture': 'methods',
        
        'experiment': 'experiments',
        'experiments': 'experiments',
        'experimental setup': 'experiments',
        'experimental results': 'experiments',
        'evaluation': 'experiments',
        'empirical evaluation': 'experiments',
        'technical details': 'experiments',
        'dataset': 'experiments',
        
        'result': 'results',
        'results': 'results',
        'findings': 'results',
        'experimental results': 'results',
        'results and discussion': 'results',
        
        'discussion': 'discussion',
        'analysis': 'discussion',
        'discussion and analysis': 'discussion',
        
        'conclusion': 'conclusion',
        'conclusions': 'conclusion',
        'concluding remarks': 'conclusion',
        'future work': 'conclusion',
        'conclusions and future work': 'conclusion',
    }
    
    # Canonical section order
    CANONICAL_SECTIONS = [
        'introduction',
        'related_work',
        'methods',
        'experiments',
        'results',
        'discussion',
        'conclusion'
    ]
    
    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize section splitter.
        
        Args:
            min_confidence: Minimum confidence for section detection
        """
        self.min_confidence = min_confidence
        
        # Compile heading patterns
        self.heading_pattern = self._compile_heading_pattern()
    
    def _compile_heading_pattern(self) -> re.Pattern:
        """
        Compile regex pattern for section headings.
        
        Matches patterns like:
            1. Introduction
            2.1 Related Work
            Introduction
            I. INTRODUCTION
        """
        # Get all possible section names
        section_names = '|'.join(
            re.escape(name)
            for name in self.SECTION_MAPPINGS.keys()
        )
        
        pattern = (
            r'^'  # Start of line
            r'(?:'
            r'(?:\d+\.)*\d+\.?\s+|'  # Numbered: "1.", "2.1", "3.2.1."
            r'[IVXLCDM]+\.?\s+|'  # Roman numerals: "I.", "II."
            r')'
            r'?'  # Optional numbering
            r'('  # Capture group
            + section_names +
            r')'
            r'(?:\s|$)'  # Followed by whitespace or end
        )
        
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    
    def split(self, text: str) -> Dict[str, str]:
        """
        Split text into canonical sections.
        
        Args:
            text: Cleaned paper text
            
        Returns:
            Dictionary mapping section names to content
        """
        if not text:
            return self._empty_sections()
        
        # Detect section boundaries
        section_matches = self._detect_sections(text)
        
        if not section_matches:
            logger.warning("No sections detected, using full text as introduction")
            return self._fallback_split(text)
        
        # Extract section content
        sections = self._extract_sections(text, section_matches)
        
        # Ensure all canonical sections present
        return self._normalize_sections(sections)
    
    def _detect_sections(self, text: str) -> List[SectionMatch]:
        """
        Detect section headings in text.
        
        Returns list of SectionMatch objects with positions.
        """
        matches = []
        
        for match in self.heading_pattern.finditer(text):
            section_name = match.group(1).lower().strip()
            canonical_name = self.SECTION_MAPPINGS.get(section_name)
            
            if canonical_name:
                # Calculate confidence based on context
                confidence = self._calculate_confidence(text, match)
                
                if confidence >= self.min_confidence:
                    matches.append(SectionMatch(
                        name=section_name,
                        canonical_name=canonical_name,
                        start_pos=match.start(),
                        confidence=confidence
                    ))
        
        # Sort by position
        matches.sort(key=lambda x: x.start_pos)
        
        # Remove duplicates (keep highest confidence)
        matches = self._deduplicate_matches(matches)
        
        logger.debug(f"Detected {len(matches)} sections: "
                    f"{[m.canonical_name for m in matches]}")
        
        return matches
    
    def _calculate_confidence(
        self,
        text: str,
        match: re.Match
    ) -> float:
        """
        Calculate confidence score for section heading.
        
        Factors:
        - Position in text (earlier = higher)
        - Line independence (own line = higher)
        - Capitalization
        - Numbering presence
        """
        confidence = 0.5  # Base confidence
        
        # Check if heading is on its own line
        start = match.start()
        end = match.end()
        
        # Check previous character
        if start == 0 or text[start-1] == '\n':
            confidence += 0.2
        
        # Check next few characters
        rest_of_line = text[end:end+50]
        if rest_of_line.startswith('\n') or rest_of_line.startswith('\r'):
            confidence += 0.2
        
        # Check for numbering
        if re.match(r'\d+\.', match.group(0)):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_matches(
        self,
        matches: List[SectionMatch]
    ) -> List[SectionMatch]:
        """Remove duplicate section matches, keeping highest confidence."""
        seen = {}
        
        for match in matches:
            canon = match.canonical_name
            if canon not in seen or match.confidence > seen[canon].confidence:
                seen[canon] = match
        
        # Return in positional order
        result = sorted(seen.values(), key=lambda x: x.start_pos)
        return result
    
    def _extract_sections(
        self,
        text: str,
        matches: List[SectionMatch]
    ) -> Dict[str, str]:
        """Extract section content based on detected boundaries."""
        sections = {}
        
        for i, match in enumerate(matches):
            # Find start of content (after heading)
            start = match.start_pos
            
            # Skip the heading line
            heading_end = text.find('\n', start)
            if heading_end == -1:
                heading_end = start
            content_start = heading_end + 1
            
            # Find end of section (start of next section or end of text)
            if i + 1 < len(matches):
                content_end = matches[i + 1].start_pos
            else:
                content_end = len(text)
            
            # Extract and clean content
            content = text[content_start:content_end].strip()
            
            sections[match.canonical_name] = content
        
        return sections
    
    def _normalize_sections(
        self,
        sections: Dict[str, str]
    ) -> Dict[str, str]:
        """Ensure all canonical sections are present."""
        normalized = {}
        
        for section_name in self.CANONICAL_SECTIONS:
            normalized[section_name] = sections.get(section_name, "")
        
        return normalized
    
    def _empty_sections(self) -> Dict[str, str]:
        """Return empty sections dictionary."""
        return {name: "" for name in self.CANONICAL_SECTIONS}
    
    def _fallback_split(self, text: str) -> Dict[str, str]:
        """
        Fallback: Use keyword-based heuristics to split text.
        
        Looks for keyword-rich sentences that might indicate section starts.
        """
        logger.warning("Using intelligent fallback splitting")
        
        sections = self._empty_sections()
        
        # Try case-insensitive keyword search for section boundaries
        text_lower = text.lower()
        
        # Find rough boundaries using keyword presence
        boundaries = []
        
        # Look for "Related work" variants - match with word boundary
        for keyword in ['related works', 'related work', 'literature review', 
                        'previous work', 'prior work']:
            # Try with line break OR capital letter (for concatenated sections)
            patterns = [
                r'(\n\s*|(?<=[.!?])\s+)' + re.escape(keyword),  # After sentence
                r'(?<!\w)' + re.escape(keyword) + r'(?=[A-Z])',  # Before capital (concatenated)
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    boundaries.append(('related_work', match.start()))
                    break
            if boundaries and boundaries[-1][0] == 'related_work':
                break
        
        # Look for "Methods" variants
        for keyword in ['methods ', 'methodology', ' approach', 'proposed method',
                        'model architecture', 'our approach']:
            patterns = [
                r'(\n\s*|(?<=[.!?])\s+)' + re.escape(keyword),
                r'(?<!\w)' + re.escape(keyword) + r'(?=[A-Z\s])',
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match and match.start() > 100:  # Not in abstract
                    boundaries.append(('methods', match.start()))
                    break
            if boundaries and boundaries[-1][0] == 'methods':
                break
        
        # Look for "Experiments" variants
        for keyword in ['experiments', 'experiment', 'evaluation', 'experimental setup',
                        'technical details', 'dataset']:
            patterns = [
                r'(\n\s*|(?<=[.!?])\s+)' + re.escape(keyword),
                r'(?<!\w)' + re.escape(keyword) + r'(?=[A-Z\s])',
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match and match.start() > 200:
                    boundaries.append(('experiments', match.start()))
                    break
            if boundaries and boundaries[-1][0] == 'experiments':
                break
        
        # Look for "Results" variants  
        for keyword in ['results and discussion', 'results', 'result', 'findings']:
            patterns = [
                r'(\n\s*|(?<=[.!?])\s+)' + re.escape(keyword),
                r'(?<!\w)' + re.escape(keyword) + r'(?=[A-Z\s])',
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match and match.start() > 300:
                    boundaries.append(('results', match.start()))
                    break
            if boundaries and boundaries[-1][0] == 'results':
                break
        
        # Look for "Conclusion" variants
        for keyword in ['conclusions', 'conclusion', 'concluding remarks', 'future work']:
            patterns = [
                r'(\n\s*|(?<=[.!?])\s+)' + re.escape(keyword),
                r'(?<!\w)' + re.escape(keyword) + r'(?=[A-Z\s])',
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match and match.start() > 400:
                    boundaries.append(('conclusion', match.start()))
                    break
            if boundaries and boundaries[-1][0] == 'conclusion':
                break
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[1])
        
        if not boundaries:
            # Really no sections found, use full text as introduction
            logger.warning("No boundaries found, using full text as introduction")
            sections['introduction'] = text
            return sections
        
        # Extract sections based on boundaries
        current_pos = 0
        
        for i, (section_name, boundary_pos) in enumerate(boundaries):
            # Before first boundary = introduction
            if i == 0 and boundary_pos > 100:
                sections['introduction'] = text[current_pos:boundary_pos].strip()
            
            # Find end of current section
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][1]
            else:
                end_pos = len(text)
            
            # Extract section content
            content = text[boundary_pos:end_pos].strip()
            
            # Append to section (may have multiple boundaries for same section)
            if sections[section_name]:
                sections[section_name] += "\n\n" + content
            else:
                sections[section_name] = content
            
            current_pos = end_pos
        
        # If no introduction extracted, use first 500 chars
        if not sections['introduction'] and boundaries:
            first_boundary = boundaries[0][1]
            if first_boundary > 50:
                sections['introduction'] = text[:first_boundary].strip()
        
        logger.info(f"Fallback extracted sections: {[k for k, v in sections.items() if v]}")
        
        return sections


def split_sections(
    text: str,
    min_confidence: float = 0.3
) -> Dict[str, str]:
    """
    Convenience function for section splitting.
    
    Args:
        text: Cleaned paper text
        min_confidence: Minimum confidence for section detection
        
    Returns:
        Dictionary mapping section names to content
    """
    splitter = SectionSplitter(min_confidence=min_confidence)
    return splitter.split(text)


if __name__ == "__main__":
    # Test section splitting
    test_text = """
    Abstract
    This is the abstract of the paper.
    
    1. Introduction
    This is the introduction section with some content.
    
    2. Related Work
    This section discusses related work in the field.
    
    3. Methodology
    Here we describe our approach and methods.
    
    4. Experiments
    We conducted several experiments to validate our approach.
    
    5. Results
    The results show significant improvements.
    
    6. Discussion
    We discuss the implications of our findings.
    
    7. Conclusion
    In conclusion, we have demonstrated...
    """
    
    splitter = SectionSplitter()
    sections = splitter.split(test_text)
    
    print("Detected sections:")
    for name, content in sections.items():
        if content:
            preview = content[:100].replace('\n', ' ')
            print(f"\n{name}:")
            print(f"  {preview}...")
