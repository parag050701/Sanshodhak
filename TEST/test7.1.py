#!/usr/bin/env python3
"""
research_pipeline.py

Complete research pipeline:
 - CrossRef search (unchanged)
 - Paper filtering & ranking
 - Sci-Hub + fallback downloads
 - DOI standardization in the download layer (safe option A)
"""

import requests
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Data class for paper metadata"""
    doi: str
    title: str
    authors: List[str]
    year: int
    citations: int
    journal: str
    url: str
    abstract: Optional[str] = None
    
    def __str__(self):
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"[{self.citations} cites] {self.title}\n  {authors_str} ({self.year}) - {self.journal}"


class CrossRefSearcher:
    """Search and retrieve papers from CrossRef API"""
    
    BASE_URL = "https://api.crossref.org/works"
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize CrossRef searcher
        
        Args:
            email: Your email for polite API usage (gets faster response)
        """
        self.email = email
        self.headers = {}
        if email:
            self.headers['User-Agent'] = f'ResearchPipeline/1.0 (mailto:{email})'
    
    def search(self, query: str, max_results: int = 100, 
               min_year: Optional[int] = None,
               sort_by: str = "relevance") -> List[Paper]:
        """
        Search CrossRef for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_year: Minimum publication year
            sort_by: Sort order - 'relevance', 'published', 'is-referenced-by-count'
            
        Returns:
            List of Paper objects
        """
        papers = []
        rows_per_request = min(100, max_results)
        offset = 0
        
        logger.info(f"Searching CrossRef for: '{query}'")
        
        while len(papers) < max_results:
            params = {
                'query': query,
                'rows': rows_per_request,
                'offset': offset,
                'sort': sort_by,
                'select': 'DOI,title,author,published-print,is-referenced-by-count,container-title,abstract,URL'
            }
            
            if min_year:
                params['filter'] = f'from-pub-date:{min_year}'
            
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.headers,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                items = data.get('message', {}).get('items', [])
                if not items:
                    break
                
                for item in items:
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                    
                    if len(papers) >= max_results:
                        break
                
                offset += rows_per_request
                time.sleep(1)  # Be polite to API
                
            except Exception as e:
                logger.error(f"Error fetching from CrossRef: {e}")
                break
        
        logger.info(f"Found {len(papers)} papers")
        return papers
    
    def _parse_paper(self, item: Dict) -> Optional[Paper]:
        """Parse CrossRef item into Paper object"""
        try:
            # Extract DOI
            doi = item.get('DOI')
            if not doi:
                return None
            
            # Extract title
            title_list = item.get('title', [])
            title = title_list[0] if title_list else "No title"
            
            # Extract authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    authors.append(f"{given} {family}".strip())
            
            # Extract year
            published = item.get('published-print') or item.get('published-online') or {}
            date_parts = published.get('date-parts', [[None]])
            year = date_parts[0][0] if date_parts[0] else None
            
            # Extract citations
            citations = item.get('is-referenced-by-count', 0)
            
            # Extract journal
            container = item.get('container-title', [])
            journal = container[0] if container else "Unknown Journal"
            
            # Extract URL
            url = item.get('URL', f"https://doi.org/{doi}")
            
            # Extract abstract
            abstract = item.get('abstract')
            
            return Paper(
                doi=doi,
                title=title,
                authors=authors,
                year=year or 0,
                citations=citations,
                journal=journal,
                url=url,
                abstract=abstract
            )
        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            return None


class PaperFilter:
    """Filter and rank papers based on various criteria"""
    
    @staticmethod
    def filter_by_citations(papers: List[Paper], min_citations: int = 0) -> List[Paper]:
        """Filter papers by minimum citation count"""
        return [p for p in papers if p.citations >= min_citations]
    
    @staticmethod
    def filter_by_year(papers: List[Paper], min_year: int, max_year: Optional[int] = None) -> List[Paper]:
        """Filter papers by year range"""
        if max_year is None:
            max_year = datetime.now().year
        return [p for p in papers if min_year <= p.year <= max_year]
    
    @staticmethod
    def filter_by_journal(papers: List[Paper], journals: List[str]) -> List[Paper]:
        """Filter papers by journal names (case-insensitive)"""
        journals_lower = [j.lower() for j in journals]
        return [p for p in papers if any(j in p.journal.lower() for j in journals_lower)]
    
    @staticmethod
    def sort_by_citations(papers: List[Paper], descending: bool = True) -> List[Paper]:
        """Sort papers by citation count"""
        return sorted(papers, key=lambda p: p.citations, reverse=descending)
    
    @staticmethod
    def sort_by_year(papers: List[Paper], descending: bool = True) -> List[Paper]:
        """Sort papers by publication year"""
        return sorted(papers, key=lambda p: p.year, reverse=descending)
    
    @staticmethod
    def get_top_papers(papers: List[Paper], n: int, 
                      citation_weight: float = 0.7,
                      recency_weight: float = 0.3) -> List[Paper]:
        """
        Get top N papers using weighted scoring
        
        Args:
            papers: List of papers
            n: Number of papers to return
            citation_weight: Weight for citations (0-1)
            recency_weight: Weight for recency (0-1)
        """
        if not papers:
            return []
        
        current_year = datetime.now().year
        max_citations = max(p.citations for p in papers) or 1
        
        scored_papers = []
        for paper in papers:
            # Normalize citation score (0-1)
            citation_score = paper.citations / max_citations
            
            # Normalize recency score (0-1)
            year_diff = current_year - (paper.year or current_year)
            recency_score = max(0, 1 - (year_diff / 20))  # Papers older than 20 years get 0
            
            # Calculate weighted score
            total_score = (citation_weight * citation_score + 
                          recency_weight * recency_score)
            
            scored_papers.append((paper, total_score))
        
        # Sort by score and return top N
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_papers[:n]]


class MirrorDiscovery:
    """Automatically discover and validate Sci-Hub mirrors"""
    
    KNOWN_MIRRORS = [
        "https://sci-hub.se",
        "https://sci-hub.st",
        "https://sci-hub.ru",
        "https://sci-hub.ren",
        "https://sci-hub.wf",
        "https://sci-hub.ee",
        "https://sci-hub.tf",
        "https://sci-hub.do",
        "https://sci-hub.shop",
        "https://sci-hub.mksa.top",
        "https://sci-hub.hkvisa.net",
        "https://sci-hub.cat",
    ]
    
    def __init__(self, cache_file: str = "mirror_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                    if datetime.now() - cache_time < timedelta(hours=24):
                        return cache
            except:
                pass
        return {'mirrors': self.KNOWN_MIRRORS, 'timestamp': datetime.now().isoformat()}
    
    def _save_cache(self):
        try:
            self.cache['timestamp'] = datetime.now().isoformat()
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def test_mirror(self, mirror: str, timeout: int = 10) -> bool:
        try:
            test_doi = "10.1126/science.169.3946.635"
            response = requests.get(
                f"{mirror}/{test_doi}",
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0'},
                allow_redirects=True
            )
            return response.status_code == 200
        except:
            return False
    
    def get_working_mirrors(self) -> List[str]:
        # For now return cache mirrors (MirrorDiscovery can be extended to actively test and update)
        return self.cache.get('mirrors', self.KNOWN_MIRRORS)


class DOIPDFFetcher:
    """Fetch PDFs from DOIs using Sci-Hub with DOI standardization (safe)"""
    
    def __init__(self, output_dir: str = "papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.mirror_discovery = MirrorDiscovery()
        self.scihub_mirrors = self.mirror_discovery.get_working_mirrors()
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.stats = {'attempts': 0, 'successes': 0, 'failures': 0}
    
    # --- Standardizer (safe, used only for downloads/alternative fetches) ---
    def standardize_doi(self, doi: str) -> str:
        """
        Fully standardize any DOI input into a canonical form for requesting (safe option).
        This function is deliberately conservative: it strips prefixes, unquotes encodings,
        removes accidental spaces, and validates the basic DOI pattern without changing case.
        """
        original = doi
        if not isinstance(doi, str):
            raise ValueError("DOI must be a string")
        doi = doi.strip()

        # Remove common prefixes (case-insensitive)
        doi = re.sub(r"^(https?://)?(www\.)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)

        # Remove whitespace and newlines inside
        doi = doi.replace("\n", "").replace("\r", "")
        doi = doi.replace(" ", "")

        # Unquote URL-encoded sequences (%2F -> /)
        doi = requests.utils.unquote(doi)

        # Remove accidental double slashes
        doi = re.sub(r"/{2,}", "/", doi)

        # Basic validation: must start with 10.<digits>/<suffix>
        pattern = r"^10\.\d{4,9}/.+$"
        if not re.match(pattern, doi):
            raise ValueError(f"DOI could not be standardized: '{original}' -> '{doi}'")

        return doi

    # Backwards-compatible clean (kept for other code paths if needed)
    def clean_doi(self, doi: str) -> str:
        doi = doi.strip()
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        return doi
    
    def get_filename_from_doi(self, doi: str) -> str:
        safe_name = re.sub(r'[^\w\-.]', '_', doi)
        return f"{safe_name}.pdf"
    
    def verify_pdf(self, content: bytes) -> bool:
        if not content or len(content) < 1024:
            return False
        if not content.startswith(b'%PDF'):
            return False
        return True
    
    def fetch_from_scihub(self, doi: str, mirror: str) -> Optional[bytes]:
        try:
            # Try multiple URL formats
            url_formats = [
                f"{mirror}/{doi}",
                f"{mirror}/https://doi.org/{doi}",
                f"{mirror}/downloads/{doi}",
            ]
            
            for pdf_url in url_formats:
                try:
                    logger.debug(f"Trying: {pdf_url}")
                    response = self.session.get(pdf_url, timeout=30, allow_redirects=True)
                    
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # Direct PDF response
                    if 'application/pdf' in content_type:
                        content = response.content
                        if self.verify_pdf(content):
                            logger.info(f"✓ Found via direct URL: {mirror}")
                            return content
                    
                    # Parse HTML for PDF links
                    if response.status_code == 200 and 'text/html' in content_type:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Method 1: Look for iframe with PDF
                        iframe = soup.find('iframe', {'id': 'pdf'})
                        if iframe and iframe.get('src'):
                            pdf_link = iframe['src']
                            if not pdf_link.startswith('http'):
                                pdf_link = mirror + pdf_link
                            
                            logger.debug(f"Found iframe: {pdf_link}")
                            pdf_response = self.session.get(pdf_link, timeout=30)
                            if 'application/pdf' in pdf_response.headers.get('content-type', ''):
                                content = pdf_response.content
                                if self.verify_pdf(content):
                                    logger.info(f"✓ Found via iframe: {mirror}")
                                    return content
                        
                        # Method 2: Look for embed tag
                        embed = soup.find('embed', {'type': 'application/pdf'})
                        if embed and embed.get('src'):
                            pdf_link = embed['src']
                            if not pdf_link.startswith('http'):
                                pdf_link = mirror + pdf_link
                            
                            logger.debug(f"Found embed: {pdf_link}")
                            pdf_response = self.session.get(pdf_link, timeout=30)
                            if 'application/pdf' in pdf_response.headers.get('content-type', ''):
                                content = pdf_response.content
                                if self.verify_pdf(content):
                                    logger.info(f"✓ Found via embed: {mirror}")
                                    return content
                        
                        # Method 3: Look for download button/link
                        for button in soup.find_all(['button', 'a'], class_=re.compile(r'download', re.I)):
                            onclick = button.get('onclick', '')
                            href = button.get('href', '')
                            
                            # Extract URL from onclick
                            url_match = re.search(r'location\.href=["\']([^"\']+)["\']', onclick)
                            if url_match:
                                pdf_link = url_match.group(1)
                            elif href and '.pdf' in href.lower():
                                pdf_link = href
                            else:
                                continue
                            
                            if not pdf_link.startswith('http'):
                                pdf_link = mirror + pdf_link
                            
                            try:
                                logger.debug(f"Found button link: {pdf_link}")
                                pdf_response = self.session.get(pdf_link, timeout=30)
                                if 'application/pdf' in pdf_response.headers.get('content-type', ''):
                                    content = pdf_response.content
                                    if self.verify_pdf(content):
                                        logger.info(f"✓ Found via button: {mirror}")
                                        return content
                            except:
                                continue
                        
                        # Method 4: Look for any PDF links
                        for link in soup.find_all('a', href=True):
                            if '.pdf' in link['href'].lower():
                                pdf_link = link['href']
                                if not pdf_link.startswith('http'):
                                    pdf_link = mirror + pdf_link
                                
                                try:
                                    logger.debug(f"Found PDF link: {pdf_link}")
                                    pdf_response = self.session.get(pdf_link, timeout=30)
                                    if 'application/pdf' in pdf_response.headers.get('content-type', ''):
                                        content = pdf_response.content
                                        if self.verify_pdf(content):
                                            logger.info(f"✓ Found via link: {mirror}")
                                            return content
                                except:
                                    continue
                
                except requests.exceptions.RequestException:
                    continue
            
            return None
        except Exception as e:
            logger.debug(f"Error with {mirror}: {e}")
            return None
    
    def fetch_pdf(self, doi: str, paper_title: str = "") -> Optional[Path]:
        """
        Download PDF for a given DOI. This function STANDARDIZES the DOI first
        (safe option A) then attempts Sci-Hub mirrors.
        """
        # Standardize DOI for all network calls
        try:
            doi = self.standardize_doi(doi)
        except Exception as e:
            logger.error(f"Invalid DOI: {doi} ({e})")
            self.stats['failures'] += 1
            return None

        filename = self.get_filename_from_doi(doi)
        output_path = self.output_dir / filename
        
        if output_path.exists():
            logger.info(f"✓ Already downloaded: {paper_title}")
            self.stats['successes'] += 1
            return output_path
        
        logger.info(f"Downloading: {paper_title} ({doi})")
        
        # Try each mirror
        for i, mirror in enumerate(self.scihub_mirrors, 1):
            logger.debug(f"Trying mirror {i}/{len(self.scihub_mirrors)}: {mirror}")
            pdf_content = self.fetch_from_scihub(doi, mirror)
            
            if pdf_content:
                try:
                    with open(output_path, 'wb') as f:
                        f.write(pdf_content)
                    logger.info(f"✓ Downloaded ({len(pdf_content)} bytes): {paper_title}")
                    self.stats['successes'] += 1
                    return output_path
                except Exception as e:
                    logger.error(f"Failed to save: {e}")
                    self.stats['failures'] += 1
                    return None
            
            time.sleep(0.5)  # Small delay between mirrors
        
        logger.warning(f"✗ Failed (all {len(self.scihub_mirrors)} mirrors): {paper_title}")
        self.stats['failures'] += 1
        return None
    
    def batch_fetch(self, papers: List[Paper], delay: float = 2.0) -> Dict:
        results = {'success': [], 'failed': []}
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper.doi}")
            
            try:
                # Use standardized DOI internally
                pdf_path = self.fetch_pdf(paper.doi, paper.title)
                
                if pdf_path:
                    results['success'].append({
                        'paper': paper,
                        'path': str(pdf_path)
                    })
                else:
                    results['failed'].append(paper)
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                results['failed'].append(paper)
            
            if i < len(papers):
                time.sleep(delay)
        
        results['statistics'] = self.stats
        return results


class AlternativeDownloader:
    """Alternative methods to download papers"""
    
    @staticmethod
    def try_unpaywall(doi: str, email: str = "user@example.com") -> Optional[str]:
        """Try to get open access PDF link from Unpaywall (doi must be canonical)"""
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('is_oa'):
                    best_oa = data.get('best_oa_location')
                    if best_oa and best_oa.get('url_for_pdf'):
                        return best_oa['url_for_pdf']
        except Exception as e:
            logger.debug(f"Unpaywall error for {doi}: {e}")
        return None
    
    @staticmethod
    def try_arxiv(doi: str) -> Optional[str]:
        """Try to find ArXiv version via CrossRef links (requires canonical doi)"""
        try:
            response = requests.get(
                f"https://api.crossref.org/works/{doi}",
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {})
                
                # Check for 'link' entries containing arXiv or check 'reference' (best-effort)
                # Also check 'relation' or 'URL' fields for arXiv
                if 'link' in message:
                    for link in message.get('link', []):
                        url = link.get('URL', '')
                        if 'arxiv.org' in url:
                            arxiv_id = re.search(r'arxiv\.org/abs/([\w\.-]+)', url)
                            if arxiv_id:
                                return f"https://arxiv.org/pdf/{arxiv_id.group(1)}.pdf"
                
                # As fallback, search DOI string on arXiv (simple HTML search) - beware rate limits
                # This is a weak fallback and may not always work.
                search_url = f"https://export.arxiv.org/api/query?search_query=all:{requests.utils.quote(doi)}&start=0&max_results=1"
                r = requests.get(search_url, timeout=15)
                if r.status_code == 200 and 'entry' in r.text:
                    # Try to extract id
                    m = re.search(r'<id>https?://arxiv.org/abs/([\w\.-]+)</id>', r.text)
                    if m:
                        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
        except Exception as e:
            logger.debug(f"ArXiv lookup error for {doi}: {e}")
        return None
    
    @staticmethod
    def download_from_url(url: str, output_path: Path) -> bool:
        """Download PDF from direct URL"""
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                content = response.content
                if content.startswith(b'%PDF'):
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    return True
        except Exception as e:
            logger.debug(f"Download from url failed: {e}")
        return False


class ResearchPipeline:
    """Complete pipeline from search to PDF download"""
    
    def __init__(self, email: Optional[str] = None, output_dir: str = "papers"):
        self.searcher = CrossRefSearcher(email=email)
        self.filter = PaperFilter()
        self.fetcher = DOIPDFFetcher(output_dir=output_dir)
        self.alt_downloader = AlternativeDownloader()
        self.email = email or "user@example.com"
    
    def search_and_download(self,
                           query: str,
                           max_results: int = 50,
                           min_citations: int = 10,
                           min_year: Optional[int] = None,
                           top_n: Optional[int] = None,
                           citation_weight: float = 0.7,
                           download: bool = True,
                           try_alternatives: bool = True) -> Dict:
        """
        Complete pipeline: search, filter, and download papers
        
        Args:
            query: Search query
            max_results: Maximum results to fetch from CrossRef
            min_citations: Minimum citation count filter
            min_year: Minimum publication year
            top_n: Number of top papers to select (None = all matching)
            citation_weight: Weight for citations in ranking (0-1)
            download: Whether to download PDFs
            
        Returns:
            Dictionary with papers and download results
        """
        print("\n" + "="*80)
        print(f"RESEARCH PIPELINE: {query}")
        print("="*80)
        
        # Step 1: Search CrossRef
        print("\n[1/4] Searching CrossRef...")
        papers = self.searcher.search(
            query=query,
            max_results=max_results,
            min_year=min_year,
            sort_by='is-referenced-by-count'
        )
        print(f"Found {len(papers)} papers")
        
        if not papers:
            return {'papers': [], 'downloads': None}
        
        # Step 2: Filter by citations
        print(f"\n[2/4] Filtering (min {min_citations} citations)...")
        filtered_papers = self.filter.filter_by_citations(papers, min_citations)
        print(f"After filtering: {len(filtered_papers)} papers")
        
        # Step 3: Select top papers
        print(f"\n[3/4] Ranking papers...")
        if top_n and top_n < len(filtered_papers):
            recency_weight = 1 - citation_weight
            final_papers = self.filter.get_top_papers(
                filtered_papers,
                n=top_n,
                citation_weight=citation_weight,
                recency_weight=recency_weight
            )
            print(f"Selected top {len(final_papers)} papers")
        else:
            final_papers = self.filter.sort_by_citations(filtered_papers)
            print(f"Using all {len(final_papers)} papers")
        
        # Display papers
        print("\n" + "-"*80)
        print("SELECTED PAPERS:")
        print("-"*80)
        for i, paper in enumerate(final_papers, 1):
            print(f"\n{i}. {paper}")
            print(f"   DOI: {paper.doi}")
        
        # Step 4: Download PDFs
        download_results = None
        if download and final_papers:
            print("\n" + "-"*80)
            print(f"[4/4] Downloading {len(final_papers)} PDFs...")
            print("-"*80)
            
            download_results = {'success': [], 'failed': [], 'statistics': {}}
            
            for i, paper in enumerate(final_papers, 1):
                logger.info(f"\n[{i}/{len(final_papers)}] {paper.doi}")
                logger.info(f"Title: {paper.title}")
                
                # Standardize DOI once (use fetcher's standardizer)
                try:
                    canonical_doi = self.fetcher.standardize_doi(paper.doi)
                except Exception as e:
                    logger.error(f"Skipping invalid DOI {paper.doi}: {e}")
                    download_results['failed'].append(paper)
                    continue
                
                # Try Sci-Hub first (fetcher will standardize again internally; harmless)
                pdf_path = self.fetcher.fetch_pdf(canonical_doi, paper.title)
                
                # If Sci-Hub fails and alternatives enabled, try other sources
                if not pdf_path and try_alternatives:
                    logger.info("Trying alternative sources...")
                    
                    # Try Unpaywall (pass canonical doi)
                    unpaywall_url = self.alt_downloader.try_unpaywall(canonical_doi, self.email)
                    if unpaywall_url:
                        logger.info(f"Found on Unpaywall: {unpaywall_url}")
                        output_path = self.fetcher.output_dir / self.fetcher.get_filename_from_doi(canonical_doi)
                        if self.alt_downloader.download_from_url(unpaywall_url, output_path):
                            logger.info(f"✓ Downloaded via Unpaywall")
                            pdf_path = output_path
                    
                    # Try ArXiv (pass canonical doi)
                    if not pdf_path:
                        arxiv_url = self.alt_downloader.try_arxiv(canonical_doi)
                        if arxiv_url:
                            logger.info(f"Found on ArXiv: {arxiv_url}")
                            output_path = self.fetcher.output_dir / self.fetcher.get_filename_from_doi(canonical_doi)
                            if self.alt_downloader.download_from_url(arxiv_url, output_path):
                                logger.info(f"✓ Downloaded via ArXiv")
                                pdf_path = output_path
                
                if pdf_path:
                    download_results['success'].append({'paper': paper, 'path': str(pdf_path)})
                else:
                    download_results['failed'].append(paper)
                
                if i < len(final_papers):
                    time.sleep(2.0)
            
            download_results['statistics'] = self.fetcher.stats
            
            print("\n" + "="*80)
            print("DOWNLOAD SUMMARY")
            print("="*80)
            print(f"✓ Success: {len(download_results['success'])}")
            print(f"✗ Failed: {len(download_results['failed'])}")
            
            if download_results['success']:
                print("\nDownloaded papers:")
                for item in download_results['success']:
                    print(f"  • {item['paper'].title}")
                    print(f"    → {item['path']}")
            
            if download_results['failed']:
                print("\nFailed papers:")
                for paper in download_results['failed']:
                    print(f"  • {paper.title}")
                    print(f"    DOI: {paper.doi}")
        
        return {
            'papers': final_papers,
            'downloads': download_results
        }
    
    def interactive_search(self):
        """Interactive mode for searching and downloading papers"""
        print("\n" + "="*80)
        print("INTERACTIVE RESEARCH PAPER PIPELINE")
        print("="*80)
        
        # Get search query
        query = input("\nEnter search query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return
        
        # Get parameters
        try:
            max_results = int(input("Maximum results to search (default 50): ") or "50")
            min_citations = int(input("Minimum citations (default 10): ") or "10")
            min_year_input = input("Minimum year (default none, e.g. 2015): ").strip()
            min_year = int(min_year_input) if min_year_input else None
            top_n_input = input("How many papers to download (default all): ").strip()
            top_n = int(top_n_input) if top_n_input else None
            
        except ValueError:
            print("Invalid input. Using defaults.")
            max_results, min_citations, min_year, top_n = 50, 10, None, None
        
        # Run pipeline
        results = self.search_and_download(
            query=query,
            max_results=max_results,
            min_citations=min_citations,
            min_year=min_year,
            top_n=top_n,
            download=True
        )
        
        # Save metadata
        if results['papers']:
            metadata_file = Path(self.fetcher.output_dir) / "metadata.json"
            with open(metadata_file, 'w') as f:
                papers_data = [asdict(p) for p in results['papers']]
                json.dump(papers_data, f, indent=2)
            print(f"\n✓ Metadata saved to: {metadata_file}")


# Example usage (script entrypoint)
if __name__ == "__main__":
    # Diagnostic: Test mirrors first (optional)
    print("="*80)
    print("TESTING SCI-HUB MIRRORS")
    print("="*80)
    
    mirror_discovery = MirrorDiscovery()
    working_mirrors = []
    for mirror in mirror_discovery.KNOWN_MIRRORS:
        print(f"\nTesting: {mirror}")
        if mirror_discovery.test_mirror(mirror):
            print(f"  ✓ WORKING")
            working_mirrors.append(mirror)
        else:
            print(f"  ✗ Not working")
    
    print(f"\n{'='*80}")
    print(f"Working mirrors: {len(working_mirrors)}/{len(mirror_discovery.KNOWN_MIRRORS)}")
    print("="*80)
    
    if not working_mirrors:
        print("\n⚠️  WARNING: No Sci-Hub mirrors are working!")
        print("This could be due to:")
        print("  1. Network restrictions/firewall")
        print("  2. ISP blocking Sci-Hub")
        print("  3. All mirrors are temporarily down")
        print("\nThe system will try alternative sources (Unpaywall, ArXiv)")
    
    print("\n" + "="*80)
    print("STARTING RESEARCH PIPELINE")
    print("="*80)
    
    pipeline = ResearchPipeline(
        email="your-email@example.com",
        output_dir="research_papers"
    )
    
    # Example run: adjust query and filters as needed
    results = pipeline.search_and_download(
        query="transformers",
        max_results=100,
        min_citations=50,
        min_year=2017,
        top_n=10,
        citation_weight=0.8,
        download=True,
        try_alternatives=True
    )
    
    # Save failed DOIs for manual retry
    if results.get('downloads') and results['downloads']['failed']:
        failed_file = Path(pipeline.fetcher.output_dir) / "failed_dois.txt"
        with open(failed_file, 'w') as f:
            for paper in results['downloads']['failed']:
                f.write(f"{paper.doi}\t{paper.title}\n")
        print(f"\n✓ Failed DOIs saved to: {failed_file}")
