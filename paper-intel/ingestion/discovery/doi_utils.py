"""
Utility functions for DOI normalization, deduplication, and paper matching.
"""
import re
import hashlib
from typing import List, Optional, Set
from fuzzywuzzy import fuzz
from ..models import PaperMetadata


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """
    Normalize DOI to standard format (10.xxxx/yyyy).
    
    Removes URLs, whitespace, trailing punctuation.
    """
    if not doi:
        return None
    
    doi = str(doi).strip()
    
    # Remove URL prefixes
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
    
    # Remove whitespace and trailing punctuation
    doi = doi.strip().rstrip('.,;')
    
    # Validate format: must start with 10.
    if not doi.startswith('10.'):
        return None
    
    # Basic sanity check: 10.XXXX/YYYY
    if not re.match(r'^10\.\d{4,9}/\S+$', doi):
        return None
    
    return doi


def is_valid_doi(doi: Optional[str]) -> bool:
    """Check if DOI is valid (format and not predatory)."""
    normalized = normalize_doi(doi)
    if not normalized:
        return False
    
    # Predatory/invalid DOI prefixes to reject
    INVALID_PREFIXES = {
        '10.21275',  # IJSR (predatory)
        '10.2139',   # SSRN (unreliable preprints)
        '10.32388',  # Predatory
        '10.65215',  # Test DOIs
        '10.37074',  # Predatory
        '10.5281',   # Zenodo (not always peer-reviewed)
    }
    
    prefix = normalized.split('/')[0]
    return prefix not in INVALID_PREFIXES


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching (lowercase, alphanumeric only)."""
    if not title:
        return ""
    
    # Lowercase
    title = title.lower()
    
    # Keep only alphanumeric and spaces
    title = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in title)
    
    # Collapse whitespace
    title = ' '.join(title.split())
    
    return title


def compute_title_similarity(title1: str, title2: str) -> float:
    """
    Compute fuzzy similarity between two titles (0-1 scale).
    
    Uses token_set_ratio for better handling of word order differences.
    """
    if not title1 or not title2:
        return 0.0
    
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    
    # Use token set ratio (handles word order)
    similarity = fuzz.token_set_ratio(norm1, norm2) / 100.0
    
    return similarity


def deduplicate_papers(
    papers: List[PaperMetadata],
    doi_priority: bool = True,
    title_threshold: float = 0.90
) -> List[PaperMetadata]:
    """
    Remove duplicate papers using DOI and title similarity.
    
    Strategy:
    1. Exact DOI match (highest priority)
    2. Fuzzy title matching (>= threshold)
    
    Args:
        papers: List of papers to deduplicate
        doi_priority: If True, DOI match takes absolute priority
        title_threshold: Minimum title similarity (0-1) to consider duplicates
        
    Returns:
        Deduplicated list of papers
    """
    if not papers:
        return []
    
    unique_papers: List[PaperMetadata] = []
    seen_dois: Set[str] = set()
    seen_titles: List[str] = []
    
    for paper in papers:
        # Check DOI first
        if paper.doi:
            normalized_doi = normalize_doi(paper.doi)
            if normalized_doi:
                if normalized_doi in seen_dois:
                    continue  # Duplicate DOI
                seen_dois.add(normalized_doi)
        
        # Check title similarity
        title = paper.title
        if not title:
            continue
        
        normalized_title = normalize_title(title)
        if not normalized_title:
            continue
        
        is_duplicate = False
        for seen_title in seen_titles:
            similarity = compute_title_similarity(normalized_title, seen_title)
            if similarity >= title_threshold:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
        
        # Add to unique set
        unique_papers.append(paper)
        seen_titles.append(normalized_title)
    
    return unique_papers


def merge_paper_metadata(papers: List[PaperMetadata]) -> PaperMetadata:
    """
    Merge metadata from multiple sources for the same paper.
    
    Prefers: non-empty fields, higher citation counts, open access URLs.
    """
    if not papers:
        raise ValueError("No papers to merge")
    
    if len(papers) == 1:
        return papers[0]
    
    # Start with paper that has most complete metadata
    merged = max(papers, key=lambda p: sum([
        bool(p.doi),
        bool(p.abstract),
        len(p.authors),
        len(p.keywords),
        p.is_open_access,
    ]))
    
    # Merge fields from other papers
    for paper in papers:
        if paper is merged:
            continue
        
        # Prefer non-empty values
        if not merged.doi and paper.doi:
            merged.doi = paper.doi
        if not merged.abstract and paper.abstract:
            merged.abstract = paper.abstract
        if not merged.year and paper.year:
            merged.year = paper.year
        if not merged.venue and paper.venue:
            merged.venue = paper.venue
        
        # Merge authors (deduplicate)
        all_authors = set(merged.authors + paper.authors)
        merged.authors = list(all_authors)
        
        # Merge keywords (deduplicate)
        all_keywords = set(merged.keywords + paper.keywords)
        merged.keywords = list(all_keywords)
        
        # Take max citations
        merged.citations = max(merged.citations, paper.citations)
        
        # Prefer open access URLs
        if paper.is_open_access and not merged.is_open_access:
            merged.is_open_access = True
            merged.open_access_pdf_url = paper.open_access_pdf_url
        
        # Aggregate sources
        if paper.source and paper.source != merged.source:
            merged.source = f"{merged.source}, {paper.source}"
    
    return merged


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Create filesystem-safe filename.
    
    Removes unsafe characters, limits length.
    """
    # Remove unsafe characters
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    safe = safe.strip('. ')
    
    # Limit length
    if len(safe) > max_length:
        safe = safe[:max_length].strip('. ')
    
    # Ensure non-empty
    if not safe:
        safe = "paper"
    
    return safe


def generate_paper_filename(paper: PaperMetadata) -> str:
    """
    Generate safe filename for paper PDF.
    
    Format: {first_author}_{year}_{safe_title_snippet}.pdf
    Fallback: {doi_hash}.pdf if title missing
    """
    # Try DOI-based name first (most reliable)
    if paper.doi:
        doi_hash = hashlib.md5(paper.doi.encode()).hexdigest()[:12]
        return f"{doi_hash}.pdf"
    
    # Try title-based name
    if paper.title:
        title_snippet = paper.title[:50]
        safe_title = sanitize_filename(title_snippet)
        
        # Add first author if available
        if paper.authors:
            first_author = paper.authors[0].split()[-1]  # Last name
            safe_author = sanitize_filename(first_author)[:20]
            safe_title = f"{safe_author}_{paper.year or 'unknown'}_{safe_title}"
        
        return f"{safe_title}.pdf"
    
    # Fallback: use any available ID
    primary_id = paper.get_primary_id()
    if primary_id:
        id_hash = hashlib.md5(primary_id.encode()).hexdigest()[:12]
        return f"{id_hash}.pdf"
    
    # Last resort: random hash
    random_hash = hashlib.md5(str(paper).encode()).hexdigest()[:12]
    return f"paper_{random_hash}.pdf"
