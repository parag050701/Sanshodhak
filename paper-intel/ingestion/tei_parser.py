"""
TEI XML Parser for GROBID output.

Parses TEI (Text Encoding Initiative) XML from GROBID
and extracts structured paper metadata and content.
"""

import re
import logging
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# TEI namespace
TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


class TEIParser:
    """Parse GROBID TEI XML output."""
    
    def __init__(self):
        """Initialize TEI parser."""
        pass
    
    def parse(self, tei_xml: str) -> Dict:
        """
        Parse TEI XML to structured dictionary.
        
        Args:
            tei_xml: TEI XML string from GROBID
            
        Returns:
            Dictionary with paper metadata and content
        """
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            logger.error(f"Failed to parse TEI XML: {e}")
            return self._empty_result()
        
        result = {
            'title': self._extract_title(root),
            'authors': self._extract_authors(root),
            'abstract': self._extract_abstract(root),
            'body_text': self._extract_body_text(root),
            'references': self._extract_references(root),
            'metadata': {
                'year': self._extract_year(root),
                'doi': self._extract_doi(root),
                'venue': self._extract_venue(root),
            }
        }
        
        return result
    
    def _extract_title(self, root: ET.Element) -> str:
        """Extract paper title."""
        title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', TEI_NS)
        
        if title_elem is not None:
            return self._get_text(title_elem)
        
        # Fallback: try any title
        title_elem = root.find('.//tei:titleStmt/tei:title', TEI_NS)
        if title_elem is not None:
            return self._get_text(title_elem)
        
        return ""
    
    def _extract_authors(self, root: ET.Element) -> List[str]:
        """Extract author names."""
        authors = []
        
        # Find all author elements
        author_elems = root.findall('.//tei:sourceDesc//tei:author', TEI_NS)
        
        for author in author_elems:
            # Try to get persName
            persname = author.find('.//tei:persName', TEI_NS)
            if persname is not None:
                # Build full name
                forename = persname.find('.//tei:forename[@type="first"]', TEI_NS)
                surname = persname.find('.//tei:surname', TEI_NS)
                
                name_parts = []
                if forename is not None:
                    name_parts.append(self._get_text(forename))
                if surname is not None:
                    name_parts.append(self._get_text(surname))
                
                if name_parts:
                    authors.append(' '.join(name_parts))
        
        return authors
    
    def _extract_abstract(self, root: ET.Element) -> str:
        """Extract abstract text."""
        abstract_elem = root.find('.//tei:profileDesc//tei:abstract', TEI_NS)
        
        if abstract_elem is not None:
            return self._get_text(abstract_elem)
        
        return ""
    
    def _extract_body_text(self, root: ET.Element) -> str:
        """Extract full body text."""
        body_elem = root.find('.//tei:text/tei:body', TEI_NS)
        
        if body_elem is not None:
            return self._get_text(body_elem)
        
        return ""
    
    def _extract_references(self, root: ET.Element) -> List[Dict]:
        """Extract bibliography references."""
        references = []
        
        # Find all biblStruct elements
        bibl_elems = root.findall('.//tei:listBibl/tei:biblStruct', TEI_NS)
        
        for i, bibl in enumerate(bibl_elems, 1):
            ref = {
                'id': bibl.get('{http://www.w3.org/XML/1998/namespace}id', f'ref_{i}'),
                'title': '',
                'authors': [],
                'year': '',
                'venue': ''
            }
            
            # Extract title
            title_elem = bibl.find('.//tei:title[@level="a"]', TEI_NS)
            if title_elem is not None:
                ref['title'] = self._get_text(title_elem)
            
            # Extract authors
            author_elems = bibl.findall('.//tei:author', TEI_NS)
            for author in author_elems:
                persname = author.find('.//tei:persName', TEI_NS)
                if persname is not None:
                    surname = persname.find('.//tei:surname', TEI_NS)
                    if surname is not None:
                        ref['authors'].append(self._get_text(surname))
            
            # Extract year
            date_elem = bibl.find('.//tei:date[@type="published"]', TEI_NS)
            if date_elem is not None:
                year = date_elem.get('when', '')
                if year:
                    ref['year'] = year[:4]  # Extract year
            
            # Extract venue
            venue_elem = bibl.find('.//tei:title[@level="j"]', TEI_NS)  # Journal
            if venue_elem is None:
                venue_elem = bibl.find('.//tei:title[@level="m"]', TEI_NS)  # Book/proceedings
            if venue_elem is not None:
                ref['venue'] = self._get_text(venue_elem)
            
            references.append(ref)
        
        return references
    
    def _extract_year(self, root: ET.Element) -> str:
        """Extract publication year."""
        # Try publication date
        date_elem = root.find('.//tei:publicationStmt//tei:date[@type="published"]', TEI_NS)
        if date_elem is not None:
            when = date_elem.get('when', '')
            if when:
                return when[:4]
        
        # Try any date
        date_elem = root.find('.//tei:publicationStmt//tei:date', TEI_NS)
        if date_elem is not None:
            when = date_elem.get('when', '')
            if when:
                return when[:4]
            text = self._get_text(date_elem)
            # Try to extract year from text
            year_match = re.search(r'\b(19|20)\d{2}\b', text)
            if year_match:
                return year_match.group(0)
        
        return ""
    
    def _extract_doi(self, root: ET.Element) -> str:
        """Extract DOI."""
        # Try idno with type="DOI"
        doi_elem = root.find('.//tei:idno[@type="DOI"]', TEI_NS)
        if doi_elem is not None:
            doi = self._get_text(doi_elem)
            # Normalize DOI
            doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
            return doi.strip()
        
        return ""
    
    def _extract_venue(self, root: ET.Element) -> str:
        """Extract publication venue (journal/conference)."""
        # Try journal title
        venue_elem = root.find('.//tei:sourceDesc//tei:title[@level="j"]', TEI_NS)
        if venue_elem is not None:
            return self._get_text(venue_elem)
        
        # Try meeting/conference
        venue_elem = root.find('.//tei:sourceDesc//tei:meeting', TEI_NS)
        if venue_elem is not None:
            return self._get_text(venue_elem)
        
        return ""
    
    def _get_text(self, elem: ET.Element) -> str:
        """
        Extract all text from element and descendants.
        
        Handles inline elements properly.
        """
        if elem is None:
            return ""
        
        # Get text and all tail text from descendants
        text_parts = []
        
        if elem.text:
            text_parts.append(elem.text)
        
        for child in elem:
            text_parts.append(self._get_text(child))
            if child.tail:
                text_parts.append(child.tail)
        
        text = ''.join(text_parts)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'title': '',
            'authors': [],
            'abstract': '',
            'body_text': '',
            'references': [],
            'metadata': {
                'year': '',
                'doi': '',
                'venue': ''
            }
        }


def parse_tei(tei_xml: str) -> Dict:
    """
    Convenience function for TEI parsing.
    
    Args:
        tei_xml: TEI XML string
        
    Returns:
        Parsed paper data
    """
    parser = TEIParser()
    return parser.parse(tei_xml)


if __name__ == "__main__":
    # Test with sample TEI
    sample_tei = """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title type="main">Deep Learning for Computer Vision</title>
                </titleStmt>
                <sourceDesc>
                    <biblStruct>
                        <analytic>
                            <author>
                                <persName>
                                    <forename type="first">John</forename>
                                    <surname>Smith</surname>
                                </persName>
                            </author>
                        </analytic>
                    </biblStruct>
                </sourceDesc>
            </fileDesc>
            <profileDesc>
                <abstract>
                    <p>This is a sample abstract.</p>
                </abstract>
            </profileDesc>
        </teiHeader>
        <text>
            <body>
                <div>
                    <head>1 Introduction</head>
                    <p>This is the introduction.</p>
                </div>
            </body>
        </text>
    </TEI>"""
    
    parser = TEIParser()
    result = parser.parse(sample_tei)
    
    print("Parsed TEI:")
    print(f"Title: {result['title']}")
    print(f"Authors: {result['authors']}")
    print(f"Abstract: {result['abstract']}")
