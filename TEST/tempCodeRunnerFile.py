#!/usr/bin/env python3
"""
ULTRA-ROBUST Research Pipeline v2.5 - MAXIMUM RESILIENCE
Enhancements:
- Exponential backoff with jitter
- Circuit breaker pattern for failing sources
- Intelligent source rotation
- Enhanced PDF verification
- Progressive timeout strategy
- Better connection pooling
- Automatic retry with different strategies
"""

import requests
import time
import re
import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
import warnings
import random
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CIRCUIT BREAKER ====================

class CircuitBreaker:
    """Prevents repeated calls to failing services"""
    def __init__(self, failure_threshold=3, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = {}
        self.last_failure_time = {}
    
    def can_attempt(self, source_name: str) -> bool:
        """Check if we should attempt this source"""
        if source_name not in self.failures:
            return True
        
        # Check if timeout has passed
        if time.time() - self.last_failure_time.get(source_name, 0) > self.timeout:
            self.failures[source_name] = 0
            return True
        
        return self.failures[source_name] < self.failure_threshold
    
    def record_failure(self, source_name: str):
        """Record a failure"""
        self.failures[source_name] = self.failures.get(source_name, 0) + 1
        self.last_failure_time[source_name] = time.time()
    
    def record_success(self, source_name: str):
        """Record a success"""
        if source_name in self.failures:
            self.failures[source_name] = max(0, self.failures[source_name] - 1)

# ==================== ENHANCED DATACLASS ====================

@dataclass
class Paper:
    doi: str
    title: str
    authors: List[str]
    year: int
    citations: int
    journal: str
    url: str
    abstract: Optional[str] = None
    relevance_score: float = 0.0

    def __str__(self):
        authors_str = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        return f"[{self.citations} cites, {self.relevance_score:.2f}] {self.title}\n  {authors_str} ({self.year}) - {self.journal}"

# ==================== SEMANTIC SEARCH ENGINE ====================

class SemanticSearcher:
    def __init__(self, email: str):
        self.email = email
        self.crossref_url = "https://api.crossref.org/works"
        self.semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Load Sentence-BERT model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded Sentence-BERT model for semantic search")
        except Exception as e:
            logger.warning(f"Could not load Sentence-BERT model: {e}")
            self.model = None
        
        self.session = self._create_session()
        
    def _create_session(self):
        """Create a robust session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100,
            pool_block=False
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': f'ResearchPipeline/2.5 (mailto:{self.email})',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        return session
    
    def calculate_relevance(self, query: str, title: str, abstract: str = "") -> float:
        """Calculate semantic similarity"""
        if not self.model:
            # Fallback to keyword matching
            query_lower = query.lower()
            title_lower = title.lower()
            abstract_lower = abstract.lower() if abstract else ""
            
            score = 0
            if query_lower in title_lower:
                score += 0.7
            if query_lower in abstract_lower:
                score += 0.3
                
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            abstract_words = set(abstract_lower.split())
            
            title_overlap = len(query_words & title_words) / max(len(query_words), 1)
            abstract_overlap = len(query_words & abstract_words) / max(len(query_words), 1)
            
            score += title_overlap * 0.5 + abstract_overlap * 0.2
            return min(score, 1.0)
        
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            title_embedding = self.model.encode(title, convert_to_tensor=True)
            
            from torch import nn
            cos_sim = nn.CosineSimilarity(dim=0)
            similarity = cos_sim(query_embedding, title_embedding).item()
            
            return max(0, (similarity + 1) / 2)
        except Exception as e:
            logger.debug(f"Relevance calculation error: {e}")
            return 0.5
    
    def search_semantic_scholar(self, query: str, max_results: int = 50) -> List[Paper]:
        """Search Semantic Scholar with robust error handling"""
        papers = []
        for attempt in range(3):
            try:
                params = {
                    'query': query,
                    'limit': min(max_results, 100),
                    'fields': 'title,authors,year,citationCount,venue,url,abstract,externalIds'
                }
                
                response = self.session.get(
                    self.semantic_scholar_url, 
                    params=params, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('data', []):
                        try:
                            doi = item.get('externalIds', {}).get('DOI', '')
                            if not doi:
                                continue
                            
                            authors = [author.get('name', '') for author in item.get('authors', [])]
                            title = item.get('title', '')
                            abstract = item.get('abstract', '')
                            
                            relevance = self.calculate_relevance(query, title, abstract)
                            
                            if relevance < 0.3:
                                continue
                            
                            papers.append(Paper(
                                doi=doi,
                                title=title,
                                authors=authors,
                                year=item.get('year', 2000),
                                citations=item.get('citationCount', 0),
                                journal=item.get('venue', 'Unknown'),
                                url=item.get('url', f'https://doi.org/{doi}'),
                                abstract=abstract,
                                relevance_score=relevance
                            ))
                            
                            if len(papers) >= max_results:
                                break
                        except Exception as e:
                            logger.debug(f"Paper parsing error: {e}")
                            continue
                    
                    logger.info(f"Semantic Scholar found {len(papers)} papers")
                    return papers
                    
                elif response.status_code == 429:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.warning(f"Semantic Scholar attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        
        return papers
    
    def search_crossref(self, query: str, max_results: int = 100, min_year: int = 2015) -> List[Paper]:
        """Search CrossRef with robust error handling"""
        papers = []
        
        for attempt in range(3):
            try:
                params = {
                    'query': query,
                    'rows': min(max_results, 100),
                    'sort': 'is-referenced-by-count',
                    'order': 'desc',
                    'filter': f'from-pub-date:{min_year}',
                    'select': 'DOI,title,author,created,publisher,is-referenced-by-count,URL,abstract'
                }
                
                response = self.session.get(self.crossref_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('message', {}).get('items', []):
                        if len(papers) >= max_results:
                            break
                        
                        try:
                            title = item.get('title', [''])[0]
                            if not title:
                                continue
                            
                            authors = []
                            for author in item.get('author', []):
                                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                                if name:
                                    authors.append(name)
                            
                            abstract = item.get('abstract', '')
                            if isinstance(abstract, str):
                                abstract = re.sub(r'<[^>]+>', '', abstract)
                            
                            relevance = self.calculate_relevance(query, title, abstract)
                            
                            if relevance < 0.3:
                                continue
                            
                            year = 2000
                            if 'created' in item and 'date-parts' in item['created']:
                                date_parts = item['created']['date-parts'][0]
                                if date_parts:
                                    year = int(date_parts[0])
                            
                            doi = item.get('DOI', '')
                            if not doi:
                                continue
                            
                            papers.append(Paper(
                                doi=doi,
                                title=title,
                                authors=authors,
                                year=year,
                                citations=item.get('is-referenced-by-count', 0),
                                journal=item.get('publisher', 'Unknown'),
                                url=item.get('URL', f'https://doi.org/{doi}'),
                                abstract=abstract,
                                relevance_score=relevance
                            ))
                        except Exception as e:
                            logger.debug(f"Paper parsing error: {e}")
                            continue
                    
                    logger.info(f"CrossRef found {len(papers)} papers")
                    return papers
                    
                elif response.status_code == 429:
                    wait_time = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.warning(f"CrossRef attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        
        return papers
    
    def search(self, query: str, max_results: int = 100, min_year: int = 2015) -> List[Paper]:
        """Main search with fallback mechanism"""
        logger.info(f"Searching for: {query}")
        
        papers = []
        
        # Try Semantic Scholar first
        try:
            papers = self.search_semantic_scholar(query, max_results)
        except Exception as e:
            logger.error(f"Semantic Scholar completely failed: {e}")
        
        # Fall back to CrossRef if needed
        if len(papers) < max_results // 2:
            try:
                crossref_papers = self.search_crossref(query, max_results, min_year)
                existing_dois = {p.doi for p in papers}
                
                for paper in crossref_papers:
                    if paper.doi not in existing_dois:
                        papers.append(paper)
                        existing_dois.add(paper.doi)
            except Exception as e:
                logger.error(f"CrossRef completely failed: {e}")
        
        logger.info(f"Total papers found: {len(papers)}")
        return papers

# ==================== PAPER FILTER ====================

class PaperFilter:
    def filter_by_citations(self, papers: List[Paper], min_citations: int = 50) -> List[Paper]:
        filtered = [p for p in papers if p.citations >= min_citations]
        logger.info(f"Filtered {len(papers)} -> {len(filtered)} papers (min {min_citations} citations)")
        return filtered
    
    def get_top_papers(self, papers: List[Paper], n: int = 20, 
                       relevance_weight: float = 0.4,
                       citation_weight: float = 0.4, 
                       recency_weight: float = 0.2) -> List[Paper]:
        if not papers:
            return []
        
        max_relevance = max(p.relevance_score for p in papers) if papers else 1
        max_cites = max(p.citations for p in papers) if papers else 1
        current_year = datetime.now().year
        max_recency = current_year - min(p.year for p in papers) if papers else 1
        
        scored_papers = []
        for paper in papers:
            relevance_score = paper.relevance_score / max_relevance if max_relevance > 0 else 0
            cite_score = paper.citations / max_cites if max_cites > 0 else 0
            recency_score = (paper.year - (current_year - max_recency)) / max_recency if max_recency > 0 else 0
            
            total_score = (relevance_score * relevance_weight + 
                          cite_score * citation_weight + 
                          recency_score * recency_weight)
            
            scored_papers.append((paper, total_score))
        
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        top_papers = [p[0] for p in scored_papers[:n]]
        
        for i, (paper, score) in enumerate(scored_papers[:n]):
            paper.relevance_score = score
        
        return top_papers

# ==================== ULTRA-ROBUST DOWNLOADER ====================

class UltraRobustDownloader:
    def __init__(self, output_dir: str = "papers", max_workers: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.session = self._create_download_session()
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=120)
        
    def _create_download_session(self):
        """Create session with aggressive retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=50,
            pool_maxsize=50,
            pool_block=False
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        return session
    
    def standardize_doi(self, doi: str) -> str:
        """Robust DOI standardization"""
        doi = doi.strip()
        doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
        doi = doi.replace(" ", "").replace("\n", "")
        doi = requests.utils.unquote(doi)
        if not re.match(r"^10\.\d{4,9}/.+", doi):
            raise ValueError(f"Invalid DOI: {doi}")
        return doi
    
    def get_filename(self, doi: str, title: str = "") -> Path:
        """Create safe filename"""
        safe_doi = re.sub(r'[^\w\-.]', '_', doi[:30])
        
        if title:
            title_slug = re.sub(r'[^\w\s-]', '', title.lower())
            title_slug = re.sub(r'[-\s]+', '_', title_slug)[:50]
            filename = f"{safe_doi}_{title_slug}.pdf"
        else:
            filename = f"{safe_doi}.pdf"
            
        return self.output_dir / filename
    
    def verify_pdf(self, content: bytes) -> bool:
        """Enhanced PDF verification"""
        if len(content) < 1024:
            return False
        
        # Check PDF header
        if content[:4] == b'%PDF':
            return True
        
        # Check for PDF in first 1KB
        if b'%PDF' in content[:1024]:
            return True
        
        # Check for common PDF markers
        pdf_markers = [b'/Type/Catalog', b'/Pages', b'endobj']
        marker_count = sum(1 for marker in pdf_markers if marker in content[:10000])
        
        return marker_count >= 2
    
    def exponential_backoff(self, attempt: int) -> float:
        """Calculate backoff time with jitter"""
        base_delay = min(2 ** attempt, 30)
        jitter = random.uniform(0, base_delay * 0.3)
        return base_delay + jitter
    
    def try_direct_doi(self, doi: str) -> Optional[bytes]:
        """Try direct DOI resolution"""
        if not self.circuit_breaker.can_attempt("direct_doi"):
            return None
        
        try:
            url = f"https://doi.org/{doi}"
            response = self.session.get(url, timeout=20, allow_redirects=True)
            
            if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', ''):
                if self.verify_pdf(response.content):
                    self.circuit_breaker.record_success("direct_doi")
                    logger.debug(f"Found via direct DOI: {doi}")
                    return response.content
        except Exception as e:
            logger.debug(f"Direct DOI failed: {e}")
            self.circuit_breaker.record_failure("direct_doi")
        
        return None
    
    def try_open_access(self, doi: str) -> Optional[bytes]:
        """Try multiple open access sources"""
        if not self.circuit_breaker.can_attempt("open_access"):
            return None
        
        sources = [
            ("semantic_scholar", f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=openAccessPdf"),
            ("unpaywall", f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"),
        ]
        
        for source_name, url in sources:
            for attempt in range(2):
                try:
                    response = self.session.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Semantic Scholar
                        if source_name == "semantic_scholar":
                            pdf_url = data.get('openAccessPdf', {}).get('url')
                        # Unpaywall
                        else:
                            pdf_url = data.get('best_oa_location', {}).get('url_for_pdf')
                        
                        if pdf_url:
                            pdf_response = self.session.get(pdf_url, timeout=30)
                            if self.verify_pdf(pdf_response.content):
                                self.circuit_breaker.record_success("open_access")
                                logger.debug(f"Found via {source_name}: {doi}")
                                return pdf_response.content
                    
                    elif response.status_code == 429:
                        time.sleep(self.exponential_backoff(attempt))
                        
                except Exception as e:
                    logger.debug(f"{source_name} failed: {e}")
                    if attempt < 1:
                        time.sleep(self.exponential_backoff(attempt))
        
        self.circuit_breaker.record_failure("open_access")
        return None
    
    def try_arxiv(self, doi: str) -> Optional[bytes]:
        """Try arXiv"""
        if not self.circuit_breaker.can_attempt("arxiv"):
            return None
        
        try:
            response = self.session.get(
                f"https://export.arxiv.org/api/query?search_query=doi:{doi}",
                timeout=10
            )
            
            if 'arxiv.org/abs' in response.text:
                match = re.search(r'arxiv\.org/abs/([\w\.\-/]+)', response.text)
                if match:
                    arxiv_id = match.group(1)
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    
                    pdf_response = self.session.get(pdf_url, timeout=30)
                    if self.verify_pdf(pdf_response.content):
                        self.circuit_breaker.record_success("arxiv")
                        logger.debug(f"Found via arXiv: {doi}")
                        return pdf_response.content
        except Exception as e:
            logger.debug(f"arXiv failed: {e}")
            self.circuit_breaker.record_failure("arxiv")
        
        return None
    
    def try_scihub_mirrors(self, doi: str) -> Optional[bytes]:
        """Try Sci-Hub with multiple mirrors"""
        if not self.circuit_breaker.can_attempt("scihub"):
            return None
        
        mirrors = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru",
            "https://sci-hub.wf",
            "https://sci-hub.mksa.top"
        ]
        
        # Shuffle mirrors for load balancing
        random.shuffle(mirrors)
        
        for mirror in mirrors[:3]:  # Try first 3 mirrors
            for attempt in range(2):
                try:
                    url = f"{mirror}/{doi}"
                    response = self.session.get(url, timeout=20, allow_redirects=True)
                    
                    # Direct PDF response
                    if 'application/pdf' in response.headers.get('content-type', ''):
                        if self.verify_pdf(response.content):
                            self.circuit_breaker.record_success("scihub")
                            logger.debug(f"Found via Sci-Hub ({mirror}): {doi}")
                            return response.content
                    
                    # Parse HTML for PDF link
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        pdf_element = soup.find('iframe', id='pdf') or \
                                      soup.find('embed', type='application/pdf') or \
                                      soup.find('a', href=re.compile(r'\.pdf'))
                        
                        if pdf_element and pdf_element.get('src' if pdf_element.name != 'a' else 'href'):
                            pdf_url = pdf_element.get('src' if pdf_element.name != 'a' else 'href')
                            
                            if not pdf_url.startswith('http'):
                                pdf_url = mirror + pdf_url.lstrip('/')
                            
                            pdf_response = self.session.get(pdf_url, timeout=30)
                            if self.verify_pdf(pdf_response.content):
                                self.circuit_breaker.record_success("scihub")
                                logger.debug(f"Found via Sci-Hub iframe ({mirror}): {doi}")
                                return pdf_response.content
                    
                    time.sleep(self.exponential_backoff(attempt))
                    
                except Exception as e:
                    logger.debug(f"Sci-Hub mirror {mirror} attempt {attempt+1} failed: {e}")
                    if attempt < 1:
                        time.sleep(self.exponential_backoff(attempt))
        
        self.circuit_breaker.record_failure("scihub")
        return None
    
    def download_one(self, paper: Paper) -> Dict[str, Any]:
        """Download with multiple strategies"""
        doi = paper.doi
        
        try:
            doi = self.standardize_doi(doi)
        except Exception as e:
            logger.warning(f"Invalid DOI {doi}: {e}")
            return {"paper": paper, "status": "invalid_doi"}
        
        path = self.get_filename(doi, paper.title)
        if path.exists() and path.stat().st_size > 10000:
            logger.info(f"Already exists: {paper.title[:60]}...")
            return {"paper": paper, "status": "exists", "path": str(path)}
        
        logger.info(f"Downloading: {paper.title[:70]}...")
        
        # Try sources in order with circuit breaker
        download_strategies = [
            ("direct_doi", lambda: self.try_direct_doi(doi)),
            ("open_access", lambda: self.try_open_access(doi)),
            ("arxiv", lambda: self.try_arxiv(doi)),
            ("scihub", lambda: self.try_scihub_mirrors(doi)),
        ]
        
        for source_name, source_func in download_strategies:
            try:
                content = source_func()
                
                if content and len(content) > 10000:
                    try:
                        with open(path, 'wb') as f:
                            f.write(content)
                        
                        # Verify file was written correctly
                        if path.exists() and path.stat().st_size > 10000:
                            logger.info(f"✓ SUCCESS via {source_name}: {paper.title[:60]}...")
                            return {
                                "paper": paper,
                                "status": "success",
                                "path": str(path),
                                "source": source_name
                            }
                    except Exception as e:
                        logger.error(f"File write error for {doi}: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Source {source_name} error: {e}")
                continue
        
        logger.warning(f"✗ FAILED all sources: {doi}")
        return {"paper": paper, "status": "failed"}
    
    def download_batch(self, papers: List[Paper]) -> Dict[str, Any]:
        """Download batch with controlled concurrency"""
        results = {"success": [], "failed": [], "exists": []}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {}
            
            for i, paper in enumerate(papers):
                if i > 0 and i % 2 == 0:
                    time.sleep(0.3)
                
                future = executor.submit(self.download_one, paper)
                future_to_paper[future] = paper
            
            completed = 0
            for future in as_completed(future_to_paper, timeout=600):
                try:
                    res = future.result(timeout=120)
                except Exception as e:
                    paper = future_to_paper[future]
                    logger.error(f"Error downloading {paper.doi}: {e}")
                    results["failed"].append({"paper": paper, "status": "error"})
                else:
                    if res["status"] == "success":
                        results["success"].append(res)
                    elif res["status"] == "exists":
                        results["exists"].append(res)
                    else:
                        results["failed"].append(res)
                
                completed += 1
                status = res.get('status', 'unknown').upper()
                print(f"  → [{completed}/{len(papers)}] {status}")
        
        return results

# ==================== RESEARCH PIPELINE ====================

class ResearchPipeline:
    def __init__(self, email: str = "sarvadnya.bhatlawande2005@gmail.com", output_dir: str = "papers"):
        self.searcher = SemanticSearcher(email=email)
        self.filter = PaperFilter()
        self.downloader = UltraRobustDownloader(output_dir=output_dir, max_workers=4)
        
    def refine_query(self, query: str) -> str:
        """Refine query"""
        corrections = {
            "retreival": "retrieval",
            "augumented": "augmented",
        }
        
        for wrong, right in corrections.items():
            query = re.sub(rf'\b{wrong}\b', right, query, flags=re.IGNORECASE)
        
        if "retrieval" in query.lower() and "augmented" in query.lower():
            query += " transformer large language model"
        
        return query
    
    def run(self, query: str, top_n: int = 15, min_citations: int = 50, min_year: int = 2018):
        """Run complete pipeline with error recovery"""
        refined_query = self.refine_query(query)
        
        print(f"\n{'='*100}")
        print(f"ULTRA-ROBUST RESEARCH PIPELINE v2.5")
        print(f"Original query: {query}")
        print(f"Refined query: {refined_query}")
        print(f"{'='*100}\n")
        
        # Search with retry
        papers = []
        for attempt in range(3):
            try:
                papers = self.searcher.search(refined_query, max_results=100, min_year=min_year)
                if papers:
                    break
            except Exception as e:
                logger.error(f"Search attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        
        if not papers:
            print("❌ No papers found after all retry attempts!")
            return {"success": [], "failed": [], "exists": []}
        
        # Filter and rank
        papers = self.filter.filter_by_citations(papers, min_citations)
        papers = self.filter.get_top_papers(
            papers, 
            n=top_n, 
            relevance_weight=0.5,
            citation_weight=0.3, 
            recency_weight=0.2
        )
        
        print(f"\nSelected {len(papers)} highly relevant papers:")
        print("-" * 100)
        
        for i, p in enumerate(papers, 1):
            print(f"{i:2d}. {p}")
            if p.abstract:
                print(f"     Abstract: {p.abstract[:150]}...")
            print()
        
        print(f"\nStarting ultra-robust download...")
        print("-" * 100)
        
        results = self.downloader.download_batch(papers)
        
        print(f"\n{'='*100}")
        print("FINAL RESULTS")
        print(f"{'='*100}")
        print(f"✓ Success: {len(results['success'])}")
        print(f"○ Already exists: {len(results['exists'])}")
        print(f"✗ Failed: {len(results['failed'])}")
        
        # Save metadata
        if results['success'] or results['exists']:
            metadata = []
            for res in results['success'] + results['exists']:
                paper_dict = asdict(res['paper'])
                if 'source' in res:
                    paper_dict['download_source'] = res.get('source')
                if 'path' in res:
                    paper_dict['local_path'] = res.get('path')
                metadata.append(paper_dict)
            
            metadata_path = self.downloader.output_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Metadata saved to: {metadata_path}")
        
        # Save failed DOIs
        if results['failed']:
            failed_path = self.downloader.output_dir / "failed_dois.txt"
            with open(failed_path, 'w', encoding='utf-8') as f:
                for r in results['failed']:
                    f.write(f"{r['paper'].doi}\t{r['paper'].title}\n")
            print(f"✗ Failed DOIs saved to: {failed_path}")
        
        # Summary
        print(f"\n{'='*100}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*100}")
        
        for res in results['success']:
            source = res.get('source', 'unknown')
            print(f"✓ [{source.upper()}] {res['paper'].title[:70]}...")
        
        return results

# ==================== CSV BATCH MODE ====================

def batch_from_csv(csv_path: str, output_dir: str = "csv_downloads"):
    """Process DOIs from CSV"""
    pipeline = ResearchPipeline(output_dir=output_dir)
    
    dois = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                dois.append(row[0].strip())
    
    papers = []
    for doi in dois:
        try:
            doi_clean = pipeline.downloader.standardize_doi(doi)
            papers.append(Paper(
                doi=doi_clean,
                title=f"DOI: {doi_clean[:30]}...",
                authors=[],
                year=0,
                citations=0,
                journal="Unknown",
                url=f"https://doi.org/{doi_clean}"
            ))
        except Exception as e:
            logger.warning(f"Invalid DOI skipped: {doi} - {e}")
    
    print(f"Loaded {len(papers)} valid DOIs from CSV")
    
    results = pipeline.downloader.download_batch(papers)
    
    summary_path = Path(output_dir) / "download_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['DOI', 'Status', 'Path', 'Source'])
        for res in results['success']:
            writer.writerow([
                res['paper'].doi,
                'success',
                res.get('path', ''),
                res.get('source', '')
            ])
        for res in results['failed']:
            writer.writerow([res['paper'].doi, 'failed', '', ''])
    
    print(f"\nSummary saved to: {summary_path}")
    return results

# ==================== MAIN ====================

if __name__ == "__main__":
    try:
        import sentence_transformers
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers", "beautifulsoup4", "lxml"])
        print("Packages installed. Please run again.")
        exit(1)
    
    pipeline = ResearchPipeline(email="sarvadnya.bhatlawande2005@gmail.com")
    
    
    # Run search with robust settings
    results = pipeline.run(
        query="Machine Learning in medical diagnosis",
        top_n=10,
        min_citations=50,
        min_year=2020
    )