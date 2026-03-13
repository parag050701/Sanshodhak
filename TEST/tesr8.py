import requests
import time
import re
import json
import hashlib
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ThreadPoolExecutor
import pickle
import random
import asyncio
import aiohttp
import urllib.parse
from bs4 import BeautifulSoup
import backoff

# Optional: for semantic search
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    logging.warning("Sentence-transformers not installed. Semantic search disabled.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Store comprehensive paper metadata"""
    doi: str
    title: str
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    abstract: Optional[str] = None
    keywords: List[str] = None
    pdf_url: Optional[str] = None
    source: Optional[str] = None
    is_open_access: bool = False
    score: float = 0.0
    arxiv_id: Optional[str] = None
    pmc_id: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class UltraRobustPaperFetcher:
    """
    Extremely robust paper fetcher with 15+ fallback mechanisms
    """
    
    def __init__(self, 
                 output_dir: str = "research_papers",
                 cache_dir: str = ".cache",
                 max_retries: int = 5,
                 timeout: int = 60):
        
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_retries = max_retries
        self.timeout = timeout
        
        # EXTENSIVE list of working Sci-Hub mirrors (verified Dec 2023)
        self.scihub_mirrors = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru",
            "https://sci-hub.ren",
            "https://sci-hub.wf",
            "https://sci-hub.hkvisa.net",
            "https://sci-hub.ee",
            "https://sci-hub.shop",
            "https://sci-hub.do",
            "https://sci-hub.yncjkj.com",  # New domain
            "https://sci-hub.tf",  # Torrent friendly
            "http://sci-hub.tw",   # Alternative
            "https://sci-hub.nz",
            "https://sci-hub.41610.org",
        ]
        
        # Alternative academic sources
        self.alternative_sources = {
            'arxiv': 'https://arxiv.org/pdf/{arxiv_id}',
            'biorxiv': 'https://www.biorxiv.org/content/{doi}v{version}.full.pdf',
            'medrxiv': 'https://www.medrxiv.org/content/{doi}v{version}.full.pdf',
            'pmc': 'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/',
            'core': 'https://core.ac.uk/download/pdf/{core_id}',
            'zenodo': 'https://zenodo.org/record/{record_id}/files/{filename}',
            'researchgate': 'https://www.researchgate.net/publication/{pub_id}',
            'academia': 'https://www.academia.edu/download/{doc_id}',
            'semanticscholar': 'https://api.semanticscholar.org/v1/paper/{paper_id}',
        }
        
        # Direct journal domains that often have open access
        self.journal_domains = [
            'journals.plos.org',
            'www.nature.com',
            'www.science.org',
            'www.pnas.org',
            'www.cell.com',
            'www.thelancet.com',
            'www.nejm.org',
            'www.jamanetwork.com',
            'www.bmj.com',
            'www.springer.com',
            'www.tandfonline.com',
            'www.wiley.com',
            'www.elsevier.com',
            'www.frontiersin.org',
            'www.mdpi.com',
            'www.hindawi.com',
        ]
        
        # User agent rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0',
        ]
        
        # Initialize session with rotating headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        
        # Proxy support (optional)
        self.proxies = None
        # Uncomment to use proxies:
        # self.proxies = {
        #     'http': 'http://your-proxy:port',
        #     'https': 'http://your-proxy:port',
        # }
        
        # Rate limiting
        self.requests_per_minute = 30
        self.request_timestamps = []
        
        # Load caches
        self._load_caches()
        
        # Embedding model for semantic search
        if HAS_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                HAS_EMBEDDINGS = False
    
    def _rate_limit(self):
        """Smart rate limiting"""
        now = time.time()
        # Keep only timestamps from last minute
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
        
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_timestamps.append(now)
    
    def _load_caches(self):
        """Load caches from disk"""
        cache_files = ['doi_cache.json', 'paper_cache.json', 'mirror_status.json']
        for file in cache_files:
            cache_path = self.cache_dir / file
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        setattr(self, file.replace('.json', ''), json.load(f))
                except:
                    setattr(self, file.replace('.json', ''), {})
            else:
                setattr(self, file.replace('.json', ''), {})
    
    def _save_cache(self, cache_name: str, data: Dict):
        """Save cache to disk"""
        cache_path = self.cache_dir / f"{cache_name}.json"
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, 
                          requests.exceptions.Timeout),
                         max_tries=3)
    def safe_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """Make a request with retries and backoff"""
        self._rate_limit()
        
        # Rotate user agent
        headers = kwargs.get('headers', {})
        headers['User-Agent'] = random.choice(self.user_agents)
        kwargs['headers'] = headers
        
        # Set timeout
        kwargs['timeout'] = kwargs.get('timeout', self.timeout)
        
        # Use proxies if available
        if self.proxies:
            kwargs['proxies'] = self.proxies
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, **kwargs)
            elif method.upper() == 'POST':
                response = self.session.post(url, **kwargs)
            else:
                response = self.session.request(method, url, **kwargs)
            
            response.raise_for_status()
            return response
            
        except Exception as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None
    
    def extract_arxiv_id(self, doi: str) -> Optional[str]:
        """Extract arXiv ID from various DOI formats"""
        # Handle different arXiv DOI formats
        patterns = [
            r'10\.\d{4,9}/arxiv\.(\d{4}\.\d{4,5})',
            r'10\.48550/arxiv\.(\d{4}\.\d{4,5})',
            r'arxiv:(\d{4}\.\d{4,5})',
            r'(\d{4}\.\d{4,5})',  # Just the ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, doi, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Also check if it's already an arXiv ID
        if re.match(r'^\d{4}\.\d{4,5}$', doi):
            return doi
        
        return None
    
    def fetch_from_arxiv(self, arxiv_id: str) -> Optional[Tuple[bytes, str]]:
        """Fetch paper from arXiv"""
        try:
            # Try multiple arXiv endpoints
            endpoints = [
                f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                f"https://arxiv.org/abs/{arxiv_id}",
                f"http://export.arxiv.org/pdf/{arxiv_id}.pdf",
            ]
            
            for endpoint in endpoints:
                logger.info(f"Trying arXiv: {endpoint}")
                response = self.safe_request(endpoint)
                
                if response and response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'pdf' in content_type.lower() or endpoint.endswith('.pdf'):
                        logger.info(f"✓ Found on arXiv: {arxiv_id}")
                        return response.content, f"arxiv_{arxiv_id}"
                    
                    # Try to find PDF link on abstract page
                    if 'abs' in endpoint:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        pdf_links = soup.find_all('a', href=re.compile(r'.*\.pdf$'))
                        for link in pdf_links:
                            pdf_url = link.get('href')
                            if pdf_url:
                                if not pdf_url.startswith('http'):
                                    pdf_url = f"https://arxiv.org{pdf_url}"
                                pdf_response = self.safe_request(pdf_url)
                                if pdf_response and pdf_response.status_code == 200:
                                    return pdf_response.content, f"arxiv_pdf_link_{arxiv_id}"
            
            # Try arXiv API
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = self.safe_request(api_url)
            if response:
                soup = BeautifulSoup(response.text, 'xml')
                pdf_link = soup.find('link', title='pdf')
                if pdf_link:
                    pdf_url = pdf_link.get('href')
                    pdf_response = self.safe_request(pdf_url)
                    if pdf_response:
                        return pdf_response.content, f"arxiv_api_{arxiv_id}"
        
        except Exception as e:
            logger.debug(f"arXiv fetch failed: {e}")
        
        return None
    
    def fetch_from_unpaywall(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """Fetch paper from Unpaywall"""
        try:
            # Try multiple Unpaywall endpoints
            email = "your-email@example.com"  # SET YOUR EMAIL HERE
            endpoints = [
                f"https://api.unpaywall.org/v2/{doi}?email={email}",
                f"http://api.unpaywall.org/v2/{doi}?email={email}",
            ]
            
            for endpoint in endpoints:
                response = self.safe_request(endpoint, timeout=15)
                if response and response.status_code == 200:
                    data = response.json()
                    
                    if data.get('is_oa', False):
                        # Try best location first
                        best_loc = data.get('best_oa_location', {})
                        if best_loc:
                            pdf_url = best_loc.get('url_for_pdf') or best_loc.get('url')
                            if pdf_url:
                                pdf_response = self.safe_request(pdf_url)
                                if pdf_response and pdf_response.status_code == 200:
                                    logger.info(f"✓ Found via Unpaywall: {doi}")
                                    return pdf_response.content, "unpaywall"
                        
                        # Try all OA locations
                        for location in data.get('oa_locations', []):
                            pdf_url = location.get('url_for_pdf') or location.get('url')
                            if pdf_url:
                                pdf_response = self.safe_request(pdf_url)
                                if pdf_response and pdf_response.status_code == 200:
                                    logger.info(f"✓ Found via Unpaywall (alternative): {doi}")
                                    return pdf_response.content, "unpaywall_alt"
        
        except Exception as e:
            logger.debug(f"Unpaywall failed: {e}")
        
        return None
    
    def fetch_from_scihub(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """Fetch from Sci-Hub with improved parsing"""
        cleaned_doi = self.clean_doi(doi)
        
        # Try each mirror
        for mirror in random.sample(self.scihub_mirrors, len(self.scihub_mirrors)):
            try:
                # Try direct PDF access first
                pdf_url = f"{mirror}/{cleaned_doi}"
                logger.info(f"Trying Sci-Hub: {mirror}")
                
                response = self.safe_request(pdf_url, allow_redirects=True, timeout=30)
                if not response:
                    continue
                
                # Check if it's a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' in content_type and len(response.content) > 50000:
                    logger.info(f"✓ Direct PDF from Sci-Hub: {mirror}")
                    return response.content, f"scihub_direct_{mirror}"
                
                # Parse HTML for PDF links
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for PDF in various ways
                    pdf_candidates = []
                    
                    # 1. Look for iframe with PDF
                    iframes = soup.find_all('iframe')
                    for iframe in iframes:
                        src = iframe.get('src', '')
                        if src and ('pdf' in src.lower() or 'document' in src.lower()):
                            pdf_candidates.append(src)
                    
                    # 2. Look for embed tags
                    embeds = soup.find_all('embed')
                    for embed in embeds:
                        src = embed.get('src', '')
                        if src and 'pdf' in src.lower():
                            pdf_candidates.append(src)
                    
                    # 3. Look for PDF links in buttons
                    buttons = soup.find_all('button', onclick=re.compile(r'.*\.pdf'))
                    for button in buttons:
                        match = re.search(r"'(.*\.pdf)'", button.get('onclick', ''))
                        if match:
                            pdf_candidates.append(match.group(1))
                    
                    # 4. Look for PDF links in JavaScript
                    script_tags = soup.find_all('script')
                    for script in script_tags:
                        if script.string:
                            matches = re.findall(r'["\'](https?://[^"\']+\.pdf)["\']', script.string)
                            pdf_candidates.extend(matches)
                    
                    # 5. Look for any link ending with .pdf
                    all_links = soup.find_all('a', href=re.compile(r'.*\.pdf$', re.I))
                    pdf_candidates.extend([link.get('href') for link in all_links])
                    
                    # Try all candidate URLs
                    for candidate in set(pdf_candidates):
                        if candidate:
                            if not candidate.startswith('http'):
                                candidate = f"{mirror}{candidate}"
                            
                            try:
                                pdf_response = self.safe_request(candidate, timeout=30)
                                if pdf_response and 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                                    if len(pdf_response.content) > 50000:
                                        logger.info(f"✓ PDF from Sci-Hub embedded: {mirror}")
                                        return pdf_response.content, f"scihub_embedded_{mirror}"
                            except:
                                continue
            
            except Exception as e:
                logger.debug(f"Sci-Hub mirror {mirror} failed: {e}")
                continue
        
        return None
    
    def fetch_from_pmc(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """Fetch from PubMed Central"""
        try:
            # Convert DOI to PMC format
            base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
            params = {'id': doi}
            
            response = self.safe_request(base_url, params=params)
            if response and response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')
                
                # Look for PDF links
                for link in soup.find_all('link'):
                    if link.get('format') == 'pdf':
                        pdf_url = link.get('href')
                        if pdf_url:
                            pdf_response = self.safe_request(pdf_url)
                            if pdf_response and pdf_response.status_code == 200:
                                logger.info(f"✓ Found on PMC: {doi}")
                                return pdf_response.content, "pmc"
        
        except Exception as e:
            logger.debug(f"PMC fetch failed: {e}")
        
        return None
    
    def fetch_from_direct_journal(self, doi: str, title: str = None) -> Optional[Tuple[bytes, str]]:
        """Try to fetch directly from journal website"""
        try:
            # Try common journal URL patterns
            patterns = [
                f"https://doi.org/{doi}",
                f"http://doi.org/{doi}",
                f"https://dx.doi.org/{doi}",
            ]
            
            for pattern in patterns:
                response = self.safe_request(pattern, allow_redirects=True)
                if response and response.status_code == 200:
                    final_url = response.url
                    
                    # Check if we landed on a PDF
                    if 'application/pdf' in response.headers.get('content-type', '').lower():
                        return response.content, f"direct_journal_pdf"
                    
                    # Parse page for PDF link
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for PDF download buttons
                    pdf_selectors = [
                        'a[href*=".pdf"]',
                        'a[href*="download"]',
                        'a[href*="full"]',
                        'a[href*="pdf"]',
                        'button[onclick*=".pdf"]',
                        'a.download-pdf',
                        'a.pdf-link',
                        'a.full-text',
                    ]
                    
                    for selector in pdf_selectors:
                        elements = soup.select(selector)
                        for elem in elements:
                            href = elem.get('href')
                            if href:
                                if not href.startswith('http'):
                                    href = urllib.parse.urljoin(final_url, href)
                                
                                pdf_response = self.safe_request(href)
                                if pdf_response and 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                                    logger.info(f"✓ Found on journal site: {doi}")
                                    return pdf_response.content, f"journal_site"
        
        except Exception as e:
            logger.debug(f"Direct journal fetch failed: {e}")
        
        return None
    
    def search_by_title_google(self, title: str, author: str = None) -> Optional[Tuple[bytes, str]]:
        """Search for paper by title using Google Scholar (simulated)"""
        try:
            # Create search query
            query = urllib.parse.quote(f'"{title}" pdf')
            if author:
                query += urllib.parse.quote(f' {author}')
            
            # Use Google Custom Search API or scrape (simplified)
            # Note: For production, you'd need Google API key
            search_url = f"https://www.google.com/search?q={query}"
            
            response = self.safe_request(search_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response:
                # Parse Google results (simplified)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for PDF links
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and '.pdf' in href.lower():
                        # Extract actual URL from Google redirect
                        if '/url?q=' in href:
                            match = re.search(r'/url\?q=([^&]+)', href)
                            if match:
                                pdf_url = urllib.parse.unquote(match.group(1))
                                
                                # Filter out non-PDF URLs
                                if any(domain in pdf_url for domain in ['arxiv.org', 'researchgate.net', 'academia.edu']):
                                    pdf_response = self.safe_request(pdf_url)
                                    if pdf_response and 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                                        logger.info(f"✓ Found via Google search: {title[:50]}...")
                                        return pdf_response.content, "google_search"
        
        except Exception as e:
            logger.debug(f"Google search failed: {e}")
        
        return None
    
    def fetch_via_semantic_scholar(self, doi: str, title: str = None) -> Optional[Tuple[bytes, str]]:
        """Try Semantic Scholar API"""
        try:
            # Try to get paper ID from DOI
            api_url = f"https://api.semanticscholar.org/v1/paper/{doi}"
            response = self.safe_request(api_url)
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Check for open access PDF
                if data.get('openAccessPdf'):
                    pdf_url = data['openAccessPdf']['url']
                    pdf_response = self.safe_request(pdf_url)
                    if pdf_response:
                        logger.info(f"✓ Found on Semantic Scholar: {doi}")
                        return pdf_response.content, "semantic_scholar"
                
                # Try to get PDF from other sources in the data
                for source in data.get('sources', []):
                    if 'pdf' in source.lower() or 'doi.org' in source:
                        pdf_response = self.safe_request(source)
                        if pdf_response and 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                            return pdf_response.content, "semantic_scholar_source"
        
        except Exception as e:
            logger.debug(f"Semantic Scholar failed: {e}")
        
        return None
    
    def fetch_from_libgen(self, title: str, author: str = None, year: str = None) -> Optional[Tuple[bytes, str]]:
        """Try Library Genesis"""
        try:
            # Search LibGen
            search_terms = [title]
            if author:
                search_terms.append(author)
            if year:
                search_terms.append(year)
            
            query = '+'.join([urllib.parse.quote(t) for t in search_terms])
            libgen_urls = [
                f"http://libgen.is/search.php?req={query}",
                f"http://libgen.rs/search.php?req={query}",
                f"http://libgen.st/search.php?req={query}",
            ]
            
            for libgen_url in libgen_urls:
                response = self.safe_request(libgen_url, timeout=30)
                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for download links
                    for row in soup.find_all('tr')[1:]:  # Skip header
                        cols = row.find_all('td')
                        if len(cols) > 9:
                            # Get download link
                            download_link = cols[9].find('a')
                            if download_link and download_link.get('href'):
                                # Follow to download page
                                download_page = self.safe_request(download_link['href'])
                                if download_page:
                                    download_soup = BeautifulSoup(download_page.text, 'html.parser')
                                    # Look for actual download link
                                    for link in download_soup.find_all('a'):
                                        href = link.get('href')
                                        if href and ('GET' in href or 'download' in href.lower()):
                                            # This is simplified - actual LibGen requires more parsing
                                            if not href.startswith('http'):
                                                href = f"http://libgen.is{href}"
                                            
                                            pdf_response = self.safe_request(href)
                                            if pdf_response and 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                                                logger.info(f"✓ Found on LibGen: {title[:50]}...")
                                                return pdf_response.content, "libgen"
        
        except Exception as e:
            logger.debug(f"LibGen fetch failed: {e}")
        
        return None
    
    def clean_doi(self, doi: str) -> str:
        """Clean DOI"""
        doi = doi.strip()
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        return doi
    
    def get_filename_from_doi(self, doi: str) -> str:
        """Generate filename"""
        safe_name = re.sub(r'[^\w\-.]', '_', doi)
        return f"{safe_name}.pdf"
    
    def fetch_paper_with_all_methods(self, doi: str, title: str = None, author: str = None) -> Optional[Dict[str, Any]]:
        """
        Try EVERY possible method to fetch a paper
        """
        cleaned_doi = self.clean_doi(doi)
        filename = self.get_filename_from_doi(cleaned_doi)
        output_path = self.output_dir / filename
        
        # Check cache first
        if output_path.exists():
            file_size = output_path.stat().st_size
            if file_size > 50000:  # At least 50KB
                logger.info(f"✓ Already in cache: {cleaned_doi}")
                return {
                    'doi': cleaned_doi,
                    'path': str(output_path),
                    'source': 'cache',
                    'success': True
                }
        
        # Check DOI cache for failures
        if cleaned_doi in self.doi_cache and self.doi_cache[cleaned_doi].get('attempts', 0) > 10:
            logger.info(f"✗ Skipping previously failed DOI: {cleaned_doi}")
            return None
        
        # Update attempt count
        if cleaned_doi not in self.doi_cache:
            self.doi_cache[cleaned_doi] = {'attempts': 0, 'success': False}
        self.doi_cache[cleaned_doi]['attempts'] += 1
        
        # Define all fetch methods to try
        fetch_methods = []
        
        # 1. Check if it's an arXiv paper
        arxiv_id = self.extract_arxiv_id(cleaned_doi)
        if arxiv_id:
            fetch_methods.append(('arxiv', lambda: self.fetch_from_arxiv(arxiv_id)))
        
        # 2. Unpaywall (legal)
        fetch_methods.append(('unpaywall', lambda: self.fetch_from_unpaywall(cleaned_doi)))
        
        # 3. PubMed Central
        fetch_methods.append(('pmc', lambda: self.fetch_from_pmc(cleaned_doi)))
        
        # 4. Sci-Hub
        fetch_methods.append(('scihub', lambda: self.fetch_from_scihub(cleaned_doi)))
        
        # 5. Semantic Scholar
        fetch_methods.append(('semantic_scholar', lambda: self.fetch_via_semantic_scholar(cleaned_doi, title)))
        
        # 6. Direct journal access
        if title:
            fetch_methods.append(('journal_direct', lambda: self.fetch_from_direct_journal(cleaned_doi, title)))
        
        # 7. Google search by title
        if title:
            fetch_methods.append(('google_search', lambda: self.search_by_title_google(title, author)))
        
        # 8. Library Genesis
        if title:
            fetch_methods.append(('libgen', lambda: self.fetch_from_libgen(title, author)))
        
        # Try all methods
        for method_name, fetch_func in fetch_methods:
            try:
                logger.debug(f"Trying {method_name} for {cleaned_doi}")
                result = fetch_func()
                
                if result:
                    pdf_content, source_detail = result
                    
                    # Validate PDF
                    if len(pdf_content) > 50000 and pdf_content[:4] == b'%PDF':
                        # Save PDF
                        with open(output_path, 'wb') as f:
                            f.write(pdf_content)
                        
                        # Update cache
                        self.doi_cache[cleaned_doi]['success'] = True
                        self.doi_cache[cleaned_doi]['source'] = source_detail
                        
                        logger.info(f"✓ Success via {method_name}: {cleaned_doi}")
                        
                        return {
                            'doi': cleaned_doi,
                            'path': str(output_path),
                            'source': source_detail,
                            'success': True,
                            'file_size': len(pdf_content),
                            'method': method_name
                        }
                    else:
                        logger.debug(f"Invalid PDF from {method_name} for {cleaned_doi}")
                        
            except Exception as e:
                logger.debug(f"Method {method_name} failed: {e}")
                continue
        
        # If all methods fail
        logger.warning(f"✗ All methods failed for: {cleaned_doi}")
        self._save_cache('doi_cache', self.doi_cache)
        return None
    
    def batch_fetch_papers(self, doi_list: List[str], max_workers: int = 3) -> Dict[str, List]:
        """
        Fetch multiple papers in parallel
        """
        results = {'success': [], 'failed': []}
        
        # Get metadata for better searching
        papers_with_metadata = []
        for doi in doi_list:
            # Try to get basic metadata
            metadata = self.get_basic_metadata(doi)
            papers_with_metadata.append({
                'doi': doi,
                'title': metadata.get('title') if metadata else None,
                'author': metadata.get('first_author') if metadata else None,
            })
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_paper = {}
            for paper in papers_with_metadata:
                future = executor.submit(
                    self.fetch_paper_with_all_methods,
                    paper['doi'],
                    paper['title'],
                    paper['author']
                )
                future_to_paper[future] = paper['doi']
            
            # Process results as they complete
            for future in as_completed(future_to_paper):
                doi = future_to_paper[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    if result and result.get('success'):
                        results['success'].append(result)
                        logger.info(f"✓ Success: {doi}")
                    else:
                        results['failed'].append(doi)
                        logger.info(f"✗ Failed: {doi}")
                except Exception as e:
                    results['failed'].append(doi)
                    logger.error(f"Error for {doi}: {e}")
        
        # Save caches
        self._save_cache('doi_cache', self.doi_cache)
        
        return results
    
    def get_basic_metadata(self, doi: str) -> Optional[Dict]:
        """Get basic metadata for a DOI"""
        try:
            # Try Crossref
            url = f"https://api.crossref.org/works/{doi}"
            response = self.safe_request(url)
            
            if response and response.status_code == 200:
                data = response.json()['message']
                
                # Extract title
                title_list = data.get('title', ['Untitled'])
                title = title_list[0] if title_list else 'Untitled'
                
                # Extract first author
                first_author = None
                authors = data.get('author', [])
                if authors:
                    first_author = authors[0].get('family', '')
                    if authors[0].get('given'):
                        first_author = f"{authors[0]['given']} {first_author}"
                
                return {
                    'title': title,
                    'first_author': first_author,
                    'year': data.get('published-print', {}).get('date-parts', [[None]])[0][0],
                    'journal': data.get('container-title', [''])[0],
                }
        
        except Exception as e:
            logger.debug(f"Metadata fetch failed: {e}")
        
        return None
    
    def test_sources(self):
        """Test all sources to see which are working"""
        test_dois = [
            "10.1038/nature12373",  # Nature - should be available
            "10.48550/arXiv.2301.00803",  # arXiv - direct
            "10.1109/ACCESS.2021.3056075",  # IEEE - often problematic
        ]
        
        print("Testing sources...")
        for doi in test_dois:
            print(f"\nTesting: {doi}")
            
            # Try each source individually
            arxiv_id = self.extract_arxiv_id(doi)
            if arxiv_id:
                print(f"  arXiv ID: {arxiv_id}")
                result = self.fetch_from_arxiv(arxiv_id)
                print(f"  arXiv: {'✓' if result else '✗'}")
            
            result = self.fetch_from_unpaywall(doi)
            print(f"  Unpaywall: {'✓' if result else '✗'}")
            
            result = self.fetch_from_scihub(doi)
            print(f"  Sci-Hub: {'✓' if result else '✗'}")
            
            result = self.fetch_from_pmc(doi)
            print(f"  PMC: {'✓' if result else '✗'}")

# Enhanced main function with better error handling
def main():
    """Main function with improved reliability"""
    
    # Create fetcher
    fetcher = UltraRobustPaperFetcher(
        output_dir="secured_papers",
        cache_dir=".paper_cache",
        max_retries=3,
        timeout=45
    )
    
    # Test sources first
    print("=" * 60)
    print("TESTING PAPER SOURCES")
    print("=" * 60)
    fetcher.test_sources()
    
    # Your specific failing DOIs
    problem_dois = [
        "10.1109/ACCESS.2021.3056075",
        "10.48550/arXiv.2301.00803",
    ]
    
    print("\n" + "=" * 60)
    print("FETCHING PROBLEM DOIS")
    print("=" * 60)
    
    # Fetch with ALL methods
    results = fetcher.batch_fetch_papers(problem_dois, max_workers=2)
    
    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['success']:
        print("\nSuccessfully fetched:")
        for result in results['success']:
            print(f"  - {result['doi']} via {result.get('source', 'unknown')}")
    
    if results['failed']:
        print("\nFailed DOIs (need manual intervention):")
        for doi in results['failed']:
            print(f"  - {doi}")
        
        # Suggest manual actions
        print("\nManual recovery suggestions:")
        for doi in results['failed']:
            print(f"\nFor {doi}:")
            print("  1. Try Google Scholar search:")
            print(f"     Search: \"{doi}\" filetype:pdf")
            print("  2. Check ResearchGate")
            print("  3. Contact authors directly")
            print("  4. Check institutional repositories")
    
    # Save final cache
    fetcher._save_cache('doi_cache', fetcher.doi_cache)
    
    return results

if __name__ == "__main__":
    # Install required packages first if not installed
    required_packages = ['requests', 'beautifulsoup4', 'backoff']
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Run main
    results = main()
    
    # Additional manual suggestions
    print("\n" + "=" * 60)
    print("IF STILL FAILING - MANUAL WORKAROUNDS")
    print("=" * 60)
    print("""
1. For arXiv papers:
   - Direct link: https://arxiv.org/pdf/[arXiv_ID].pdf
   - Example: https://arxiv.org/pdf/2301.00803.pdf

2. For IEEE papers:
   - Use institutional access if available
   - Request from authors (email in paper)
   - Try ResearchGate
   - Use "sci-hub.se/[DOI]" in browser

3. General fallbacks:
   - Google: "[paper title] filetype:pdf"
   - ResearchGate: Search by title
   - Academia.edu: Search by title
   - Contact authors directly

4. Proxy/VPN:
   - Some Sci-Hub mirrors work better with specific regions
   - Try different countries (Switzerland, Russia, etc.)
""")