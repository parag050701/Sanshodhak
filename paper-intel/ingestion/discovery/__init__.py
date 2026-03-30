"""Discovery module for paper search and PDF downloading."""
from .openalex_client import OpenAlexClient
from .core_client import COREClient
from .semanticscholar_client import SemanticScholarClient
from .arcsieve_client import ArcSieveClient
from .crossref_client import CrossRefClient
from .unpaywall_client import UnpaywallClient
from .arxiv_client import ArxivClient
from .openaire_client import OpenAIREClient
from .zenodo_client import ZenodoClient
from .plos_client import PLOSClient
from .eric_client import ERICClient
from .oatd_client import OATDClient
from .hal_client import HALClient
from .scielo_client import SciELOClient
from .biorxiv_client import BioRxivMedRxivClient
from .springeropen_client import SpringerOpenClient
from .oai_pmh_client import OAIPMHClient
from .scihub_client import SciHubClient
from .annasarchive_client import AnnasArchiveClient
from .search_engine import SearchEngine
from .pdf_downloader import PDFDownloader
from .health_report import get_health_tracker, SourceHealthTracker, SourceHealth, CircuitState
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
    'OpenAIREClient',
    'ZenodoClient',
    'PLOSClient',
    'ERICClient',
    'OATDClient',
    'HALClient',
    'SciELOClient',
    'BioRxivMedRxivClient',
    'SpringerOpenClient',
    'OAIPMHClient',
    'SciHubClient',
    'AnnasArchiveClient',
    'SearchEngine',
    'PDFDownloader',
    'get_health_tracker',
    'SourceHealthTracker',
    'SourceHealth',
    'CircuitState',
    'normalize_doi',
    'is_valid_doi',
    'deduplicate_papers',
    'merge_paper_metadata',
    'generate_paper_filename',
]
