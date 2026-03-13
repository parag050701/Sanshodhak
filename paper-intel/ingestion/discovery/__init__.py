"""Discovery module for paper search and PDF downloading."""
from .openalex_client import OpenAlexClient
from .core_client import COREClient
from .semanticscholar_client import SemanticScholarClient
from .arcsieve_client import ArcSieveClient
from .crossref_client import CrossRefClient
from .unpaywall_client import UnpaywallClient
from .arxiv_client import ArxivClient
from .scihub_client import SciHubClient
from .annasarchive_client import AnnasArchiveClient
from .search_engine import SearchEngine
from .pdf_downloader import PDFDownloader
from .doi_utils import (
    normalize_doi,
    is_valid_doi,
    deduplicate_papers,
    merge_paper_metadata,
    generate_paper_filename
)

__all__ = [
    'OpenAlexClient',
    'COREClient',
    'SemanticScholarClient',
    'ArcSieveClient',
    'CrossRefClient',
    'UnpaywallClient',
    'ArxivClient',
    'SciHubClient',
    'AnnasArchiveClient',
    'SearchEngine',
    'PDFDownloader',
    'normalize_doi',
    'is_valid_doi',
    'deduplicate_papers',
    'merge_paper_metadata',
    'generate_paper_filename',
]
