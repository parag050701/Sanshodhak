"""
ingest_papers.py — Download papers on RAG/IR topics into ingestion/raw_text/

Uses OpenAlex, CrossRef, and arXiv to discover papers, then Unpaywall to get
open-access PDF URLs. Downloads PDFs and extracts plain text.

Usage:
    D:\\Sanshodhak\\.venv\\Scripts\\python.exe ingest_papers.py

Output:
    ingestion/raw_text/<filename>.txt  (one file per paper)

Requires:
    UNPAYWALL_EMAIL in .env (or set directly in this script)
"""

import os
import sys
import re
import time
import hashlib
import logging
import requests
import unicodedata
from pathlib import Path
from typing import Optional

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Add paper-intel to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from ingestion.discovery import (
    OpenAlexClient,
    CrossRefClient,
    ArxivClient,
    UnpaywallClient,
    deduplicate_papers,
)
from ingestion.models import PaperMetadata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_TEXT_DIR = Path(__file__).parent / "ingestion" / "raw_text"
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")

# Search queries and per-query limits (total target ~200 unique papers)
QUERIES = [
    ("retrieval augmented generation", 60),
    ("hybrid BM25 dense vector retrieval", 50),
    ("knowledge graph question answering NLP", 45),
    ("cross encoder reranking information retrieval", 40),
    ("scholarly research assistant NLP pipeline", 35),
]

MIN_YEAR = 2019  # only recent papers


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF → pdfplumber → PyPDF2 cascade."""
    text = ""

    # Try PyMuPDF (fitz) — best quality
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        text = "\n".join(parts)
        if len(text.strip()) > 200:
            return text
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            text = "\n".join(parts)
            if len(text.strip()) > 200:
                return text
    except Exception:
        pass

    # Try PyPDF2
    try:
        import PyPDF2
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        text = "\n".join(parts)
    except Exception:
        pass

    return text


def download_pdf(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download a PDF from a URL and return raw bytes, or None on failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; SanshodhakBot/1.0; "
                "+https://github.com/parag050701/Sanshodhak)"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return None
        # Check Content-Type
        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct and not url.lower().endswith(".pdf"):
            # Read a bit to check magic bytes
            chunk = next(resp.iter_content(8), b"")
            if not chunk.startswith(b"%PDF"):
                return None
            return chunk + resp.content
        return resp.content
    except Exception as e:
        logger.debug("Download failed for %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _safe_token(s: str, max_len: int = 20) -> str:
    """Normalise and truncate a string for use in a filename."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return s[:max_len]


def make_filename(paper: PaperMetadata) -> str:
    """
    Deterministic filename from paper metadata.
    Format: <FirstAuthorLastname>_<Year>_<doi_hash8>.txt
    Matches convention of existing raw_text files.
    """
    # Author
    first_author = "Unknown"
    if paper.authors:
        name = paper.authors[0]
        parts = name.strip().split()
        first_author = _safe_token(parts[-1]) if parts else "Unknown"

    year = str(paper.year) if paper.year else "0000"

    # Short hash for uniqueness (DOI preferred, fallback title)
    key = paper.doi or paper.title or "nokey"
    h = hashlib.md5(key.encode()).hexdigest()[:8]

    return f"{first_author}_{year}_{h}.txt"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def existing_filenames() -> set:
    return {f.name for f in RAW_TEXT_DIR.glob("*.txt")}


def ingest_paper(paper: PaperMetadata, unpaywall: Optional[UnpaywallClient]) -> bool:
    """
    Try to get text for a paper and save it to raw_text/.
    Returns True if saved successfully.
    """
    filename = make_filename(paper)
    out_path = RAW_TEXT_DIR / filename

    if out_path.exists():
        logger.debug("Already exists: %s", filename)
        return False  # already ingested

    # --- Determine PDF URL ---
    pdf_url = paper.pdf_url or paper.open_access_pdf_url

    # Unpaywall lookup if we have a DOI and no URL yet
    if not pdf_url and paper.doi and unpaywall:
        oa = unpaywall.get_pdf_url(paper.doi)
        if oa:
            pdf_url = oa["pdf_url"]
            paper.is_open_access = True

    # arXiv fallback via arxiv_id
    if not pdf_url and paper.arxiv_id:
        arxiv_id = paper.arxiv_id.replace("arXiv:", "").strip()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    if not pdf_url:
        logger.debug("No PDF URL for: %s", paper.title[:60])
        return False

    # --- Download PDF ---
    logger.info("Downloading: %s", paper.title[:70])
    pdf_bytes = download_pdf(pdf_url)
    if not pdf_bytes:
        logger.debug("Download failed: %s", pdf_url)
        return False

    # --- Extract text ---
    text = extract_text_from_pdf(pdf_bytes)
    text = text.strip()
    if len(text) < 500:
        logger.debug("Too short after extraction (%d chars): %s", len(text), filename)
        return False

    # --- Save ---
    out_path.write_text(text, encoding="utf-8", errors="replace")
    logger.info("Saved: %s  (%d chars)", filename, len(text))
    return True


def run():
    # Setup clients
    unpaywall = None
    if UNPAYWALL_EMAIL and UNPAYWALL_EMAIL != "your@email.com":
        unpaywall = UnpaywallClient(email=UNPAYWALL_EMAIL)
        logger.info("Unpaywall enabled (%s)", UNPAYWALL_EMAIL)
    else:
        logger.warning(
            "UNPAYWALL_EMAIL not set — set it in .env to enable OA PDF lookup for CrossRef results"
        )

    openalex = OpenAlexClient(email=UNPAYWALL_EMAIL or None)
    crossref = CrossRefClient(email=UNPAYWALL_EMAIL or None)
    arxiv = ArxivClient()

    all_papers: list[PaperMetadata] = []

    for query, limit in QUERIES:
        logger.info("=== Query: %r (limit=%d) ===", query, limit)

        # OpenAlex — OA papers preferred, has direct PDF links
        try:
            res = openalex.search(query, limit=limit, min_year=MIN_YEAR, open_access_only=True)
            logger.info("  OpenAlex: %d papers", len(res.papers))
            all_papers.extend(res.papers)
        except Exception as e:
            logger.warning("  OpenAlex failed: %s", e)

        # CrossRef — broad metadata coverage, needs Unpaywall for PDFs
        try:
            res = crossref.search(query, limit=limit // 2, min_year=MIN_YEAR)
            logger.info("  CrossRef: %d papers", len(res.papers))
            all_papers.extend(res.papers)
        except Exception as e:
            logger.warning("  CrossRef failed: %s", e)

        # arXiv — preprints, always have PDFs
        try:
            res = arxiv.search(query, limit=limit // 3)
            logger.info("  arXiv: %d papers", len(res.papers))
            all_papers.extend(res.papers)
        except Exception as e:
            logger.warning("  arXiv failed: %s", e)

        time.sleep(0.5)  # brief pause between query groups

    logger.info("Total discovered (before dedup): %d", len(all_papers))

    # Deduplicate
    unique = deduplicate_papers(all_papers)
    logger.info("After deduplication: %d unique papers", len(unique))

    # Filter out already-ingested papers
    existing = existing_filenames()
    logger.info("Already in raw_text/: %d files", len(existing))

    # Download and save
    saved = 0
    skipped = 0
    failed = 0

    for paper in unique:
        # Skip if filename collision (DOI-based hash is stable)
        fn = make_filename(paper)
        if fn in existing:
            skipped += 1
            continue

        ok = ingest_paper(paper, unpaywall)
        if ok:
            saved += 1
            existing.add(make_filename(paper))
        else:
            failed += 1

        # Small delay between downloads to be polite
        time.sleep(0.3)

    print("\n" + "=" * 60)
    print(f"Ingestion complete")
    print(f"  New papers saved : {saved}")
    print(f"  Already existed  : {skipped}")
    print(f"  No PDF / failed  : {failed}")
    print(f"  Total in raw_text: {len(list(RAW_TEXT_DIR.glob('*.txt')))}")
    print("=" * 60)
    print("\nNext step: run rebuild_rag_index.py to update the FAISS index")


if __name__ == "__main__":
    run()
