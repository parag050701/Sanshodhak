"""Ingestion module - paper discovery, download, and preprocessing."""
from .models import PaperMetadata, SearchResult, DownloadResult
from .discovery import SearchEngine, PDFDownloader
from .ingestion_engine import IngestionEngine

__all__ = [
    'PaperMetadata',
    'SearchResult',
    'DownloadResult',
    'SearchEngine',
    'PDFDownloader',
    'IngestionEngine',
]
