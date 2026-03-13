"""
Data models and type definitions for the Sanshodhak ingestion pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class PaperMetadata:
    """Unified paper metadata model across all sources."""
    
    # Core identifiers
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    
    # Bibliographic info
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None  # Journal/conference
    publisher: Optional[str] = None
    
    # Additional metadata
    keywords: List[str] = field(default_factory=list)
    citations: int = 0
    references: List[str] = field(default_factory=list)
    
    # Access information
    is_open_access: bool = False
    open_access_pdf_url: Optional[str] = None
    pdf_url: Optional[str] = None
    
    # Provenance
    source: str = "unknown"  # Which API/service provided this
    score: float = 0.0  # Relevance/quality score
    fetched_at: datetime = field(default_factory=datetime.now)
    
    # Local storage
    pdf_path: Optional[str] = None
    text_path: Optional[str] = None
    json_path: Optional[str] = None
    
    def __post_init__(self):
        """Normalize fields after initialization."""
        if isinstance(self.authors, str):
            self.authors = [self.authors]
        if isinstance(self.keywords, str):
            self.keywords = [self.keywords]
        if self.title:
            self.title = self.title.strip()
    
    def get_primary_id(self) -> Optional[str]:
        """Return the first available identifier."""
        return self.doi or self.arxiv_id or self.pmid or self.openalex_id or self.s2_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['fetched_at'] = self.fetched_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperMetadata':
        """Create from dictionary."""
        if 'fetched_at' in data and isinstance(data['fetched_at'], str):
            data['fetched_at'] = datetime.fromisoformat(data['fetched_at'])
        return cls(**data)


@dataclass
class SearchResult:
    """Container for search results from a single source."""
    source: str
    query: str
    papers: List[PaperMetadata]
    total_found: int
    fetched_count: int
    success: bool = True
    error: Optional[str] = None
    search_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'query': self.query,
            'papers': [p.to_dict() for p in self.papers],
            'total_found': self.total_found,
            'fetched_count': self.fetched_count,
            'success': self.success,
            'error': self.error,
            'search_time_ms': self.search_time_ms
        }


@dataclass
class DownloadResult:
    """Result of PDF download attempt."""
    paper_id: str  # DOI or other identifier
    success: bool
    pdf_path: Optional[str] = None
    source: Optional[str] = None  # Which source succeeded
    file_size_bytes: int = 0
    error: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
