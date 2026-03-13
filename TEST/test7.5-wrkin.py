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
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import random
from collections import Counter

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    references: List[str] = None
    citations: int = 0
    pdf_url: Optional[str] = None
    source: Optional[str] = None
    is_open_access: bool = False
    score: float = 0.0
    
    def to_dict(self):
        return asdict(self)

class SmartResearchPipeline:
    """
    Intelligent research paper collection pipeline with iterative improvement
    Searches Crossref, fetches papers from multiple sources, uses embeddings,
    and retries with feedback loops to achieve high success rates
    """
    
    def __init__(self, 
                 output_dir: str = "research_papers",
                 cache_dir: str = "cache",
                 min_success_rate: float = 0.7,
                 max_iterations: int = 3):
        
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.min_success_rate = min_success_rate
        self.max_iterations = max_iterations
        
        # Multiple Sci-Hub mirrors (rotating list)
        self.scihub_mirrors = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru",
            "https://sci-hub.ren",
            "https://sci-hub.wf",
            "https://sci-hub.hkvisa.net",
            "https://sci-hub.ee",
            "https://sci-hub.shop",
        ]
        
        # Anna's Archive configuration
        self.annas_archive_url = "https://annas-archive.org"
        self.annas_archive_seeds = [
            "libgen.lc", "libgen.rs", "libgen.st", "libgen.is",
            "z-library", "booksc.org", "booksc.eu"
        ]
        
        # Additional academic sources
        self.additional_sources = [
            "https://arxiv.org/pdf/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/",
            "https://www.researchgate.net/publication/",
            "https://zenodo.org/record/",
            "https://core.ac.uk/download/pdf/",
            "https://dspace.mit.edu/bitstream/handle/",
        ]
        
        # Unpaywall API
        self.unpaywall_email = None  # Set your email for legal access
        
        # Semantic search model
        self.embedding_model = None
        self.embedding_cache = {}
        self.similarity_threshold = 0.75
        
        # Topic modeling and search expansion
        self.search_expansion_terms = {}
        
        # Initialize session with better headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html, application/xhtml+xml, application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Rate limiting
        self.request_delay = 1.0
        self.last_request = datetime.now()
        
        # Load caches
        self._load_caches()
        
    def _load_caches(self):
        """Load all caches from disk"""
        # Embedding cache
        self.embedding_cache_file = self.cache_dir / "embeddings.pkl"
        if self.embedding_cache_file.exists():
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
            except:
                self.embedding_cache = {}
        
        # DOI cache
        self.doi_cache_file = self.cache_dir / "doi_cache.json"
        self.doi_cache = {}
        if self.doi_cache_file.exists():
            try:
                with open(self.doi_cache_file, 'r') as f:
                    self.doi_cache = json.load(f)
            except:
                self.doi_cache = {}
        
        # Search results cache
        self.search_cache_file = self.cache_dir / "search_cache.json"
        self.search_cache = {}
        if self.search_cache_file.exists():
            try:
                with open(self.search_cache_file, 'r') as f:
                    self.search_cache = json.load(f)
            except:
                self.search_cache = {}
    
    def _save_caches(self):
        """Save all caches to disk"""
        # Embedding cache
        if HAS_EMBEDDINGS:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        
        # DOI cache
        with open(self.doi_cache_file, 'w') as f:
            json.dump(self.doi_cache, f)
        
        # Search cache
        with open(self.search_cache_file, 'w') as f:
            json.dump(self.search_cache, f)
    
    def _rate_limit(self):
        """Implement rate limiting"""
        now = datetime.now()
        elapsed = (now - self.last_request).total_seconds()
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = datetime.now()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using all-MiniLM-L6-v2 model"""
        if not HAS_EMBEDDINGS:
            return None
        
        if self.embedding_model is None:
            try:
                logger.info("Loading all-MiniLM-L6-v2 embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Could not load embedding model: {e}")
                return None
        
        # Clean and truncate text
        text = text.strip()
        if len(text.split()) > 500:
            text = ' '.join(text.split()[:500])
        
        # Create hash for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            embedding = self.embedding_model.encode(text)
            self.embedding_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    def _extract_keywords_from_text(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        if not text:
            return []
        
        # Remove common words and symbols
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stopwords = set(['this', 'that', 'with', 'from', 'have', 'which', 'their',
                        'there', 'what', 'will', 'would', 'could', 'about', 'when',
                        'were', 'them', 'like', 'such', 'also', 'more', 'most'])
        
        word_freq = Counter(words)
        for word in stopwords:
            if word in word_freq:
                del word_freq[word]
        
        return [word for word, _ in word_freq.most_common(num_keywords)]
    
    def search_crossref(self, query: str, max_results: int = 50, 
                       min_year: int = None, max_year: int = None) -> List[PaperMetadata]:
        """
        Search Crossref API for papers on a topic
        
        Args:
            query: Search query/topic
            max_results: Maximum number of results to return
            min_year: Minimum publication year
            max_year: Maximum publication year
            
        Returns:
            List of PaperMetadata objects
        """
        # Check cache first
        cache_key = f"{query}_{max_results}_{min_year}_{max_year}"
        if cache_key in self.search_cache:
            logger.info(f"Using cached search results for: {query}")
            cached = self.search_cache[cache_key]
            return [PaperMetadata(**item) for item in cached]
        
        papers = []
        rows = min(100, max_results)  # Crossref max is 1000 per request
        pages = math.ceil(max_results / rows)
        
        for page in range(pages):
            try:
                self._rate_limit()
                
                params = {
                    'query': query,
                    'rows': rows,
                    'offset': page * rows,
                    'sort': 'relevance',
                    'filter': 'type:journal-article',
                }
                
                if min_year:
                    params['filter'] = f'{params.get("filter", "")},from-pub-date:{min_year}'
                if max_year:
                    params['filter'] = f'{params.get("filter", "")},until-pub-date:{max_year}'
                
                url = "https://api.crossref.org/works"
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data['message']['items']:
                        try:
                            doi = item.get('DOI')
                            if not doi:
                                continue
                            
                            # Extract title
                            title_list = item.get('title', ['Untitled'])
                            title = title_list[0] if title_list else 'Untitled'
                            
                            # Extract authors
                            authors = []
                            for author in item.get('author', []):
                                given = author.get('given', '')
                                family = author.get('family', '')
                                if given and family:
                                    authors.append(f"{given} {family}")
                                elif 'name' in author:
                                    authors.append(author['name'])
                            
                            # Extract year
                            year = None
                            date_parts = item.get('published-print', {}).get('date-parts', [])
                            if date_parts and date_parts[0]:
                                year = date_parts[0][0]
                            
                            # Extract journal
                            journal = ''
                            container = item.get('container-title', [''])
                            if container:
                                journal = container[0]
                            
                            # Extract abstract (from multiple possible fields)
                            abstract = ''
                            if 'abstract' in item:
                                abstract = item['abstract']
                            elif 'description' in item:
                                abstract = item['description']
                            
                            # Extract keywords
                            keywords = []
                            for keyword_list in item.get('subject', []):
                                keywords.extend([k.lower() for k in keyword_list.split(', ')])
                            
                            # Get citation count
                            citations = item.get('is-referenced-by-count', 0)
                            
                            paper = PaperMetadata(
                                doi=doi,
                                title=title,
                                authors=authors,
                                year=year,
                                journal=journal,
                                abstract=abstract,
                                keywords=keywords,
                                citations=citations,
                                score=item.get('score', 0.0)
                            )
                            
                            papers.append(paper)
                            
                            if len(papers) >= max_results:
                                break
                                
                        except Exception as e:
                            logger.debug(f"Error processing paper: {e}")
                            continue
                
                if len(papers) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching Crossref: {e}")
                break
        
        # Cache the results
        self.search_cache[cache_key] = [p.to_dict() for p in papers]
        
        logger.info(f"Found {len(papers)} papers for query: '{query}'")
        return papers
    
    def expand_search_queries(self, topic: str, success_rate: float, 
                            failed_papers: List[PaperMetadata]) -> List[str]:
        """
        Expand search queries based on failed papers and success rate
        """
        expanded_queries = [topic]
        
        # If success rate is low, expand the search
        if success_rate < 0.5:
            # Extract keywords from failed papers
            all_keywords = []
            for paper in failed_papers:
                if paper.abstract:
                    keywords = self._extract_keywords_from_text(paper.abstract, 5)
                    all_keywords.extend(keywords)
                if paper.title:
                    keywords = self._extract_keywords_from_text(paper.title, 3)
                    all_keywords.extend(keywords)
            
            # Get most common keywords
            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, cnt in keyword_counts.most_common(10)]
            
            # Create expanded queries
            for keyword in top_keywords[:5]:
                expanded_queries.append(f"{topic} {keyword}")
                expanded_queries.append(f"{keyword}")
            
            # Add synonyms and related terms
            related_terms = {
                "machine learning": ["deep learning", "neural networks", "artificial intelligence"],
                "neuroscience": ["brain", "cognitive", "neurobiology"],
                "biology": ["genetics", "biochemistry", "molecular biology"],
                "physics": ["quantum", "astrophysics", "thermodynamics"],
                "chemistry": ["organic chemistry", "inorganic chemistry", "biochemistry"],
            }
            
            for main_topic, synonyms in related_terms.items():
                if main_topic.lower() in topic.lower():
                    expanded_queries.extend([f"{synonym} {topic}" for synonym in synonyms[:2]])
        
        return list(set(expanded_queries))
    
    def fetch_from_scihub(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """Fetch PDF from Sci-Hub with multiple mirror attempts"""
        cleaned_doi = self.clean_doi(doi)
        
        # Try each mirror
        for mirror in random.sample(self.scihub_mirrors, len(self.scihub_mirrors)):
            try:
                self._rate_limit()
                
                # Try direct PDF URL
                pdf_url = f"{mirror}/{cleaned_doi}"
                logger.debug(f"Trying Sci-Hub mirror: {mirror}")
                
                response = self.session.get(pdf_url, timeout=30, allow_redirects=True)
                
                # Check for PDF
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/pdf' in content_type and len(response.content) > 10000:
                    logger.info(f"✓ PDF found at Sci-Hub: {mirror}")
                    return response.content, f"scihub_{mirror}"
                
                # Parse page for PDF links
                if response.status_code == 200:
                    # Look for embedded PDFs
                    pdf_patterns = [
                        r'src=["\']([^"\']*\.pdf[^"\']*)["\']',
                        r'<iframe[^>]+src=["\']([^"\']+)["\']',
                        r'<embed[^>]+src=["\']([^"\']+)["\']',
                        r'download[^>]*href=["\']([^"\']+\.pdf)["\']',
                        r'location\.href=["\']([^"\']+\.pdf)["\']',
                    ]
                    
                    for pattern in pdf_patterns:
                        matches = re.findall(pattern, response.text, re.IGNORECASE)
                        for match in matches:
                            pdf_link = match if match.startswith('http') else f"{mirror}{match}"
                            try:
                                pdf_response = self.session.get(pdf_link, timeout=30)
                                if 'application/pdf' in pdf_response.headers.get('content-type', '').lower():
                                    if len(pdf_response.content) > 10000:
                                        logger.info(f"✓ PDF from embedded link: {mirror}")
                                        return pdf_response.content, f"scihub_embedded_{mirror}"
                            except:
                                continue
                
            except Exception as e:
                logger.debug(f"Sci-Hub mirror {mirror} failed: {e}")
                continue
        
        return None
    
    def fetch_from_annas_archive(self, metadata: PaperMetadata) -> Optional[Tuple[bytes, str]]:
        """Fetch paper from Anna's Archive using semantic search"""
        if not HAS_EMBEDDINGS or self.embedding_model is None:
            return None
        
        try:
            # Create search query
            search_query = f"{metadata.title} {' '.join(metadata.authors[:2]) if metadata.authors else ''}"
            
            # Search Anna's Archive (simulated - in reality, you'd use their API or scrape)
            # This is a placeholder implementation
            logger.debug(f"Searching Anna's Archive for: {metadata.title[:50]}...")
            
            # Generate embedding for the paper
            paper_text = f"{metadata.title} {metadata.abstract or ''}"
            paper_embedding = self._get_embedding(paper_text)
            
            if paper_embedding is None:
                return None
            
            # In a real implementation, you would:
            # 1. Search Anna's Archive with the query
            # 2. Get results and their embeddings
            # 3. Calculate cosine similarity
            # 4. Download the best match
            
            # For now, return None (placeholder)
            return None
            
        except Exception as e:
            logger.debug(f"Anna's Archive search failed: {e}")
            return None
    
    def fetch_from_unpaywall(self, doi: str) -> Optional[Tuple[bytes, str]]:
        """Fetch PDF from Unpaywall (legal open access)"""
        if not self.unpaywall_email:
            return None
        
        try:
            self._rate_limit()
            
            url = f"https://api.unpaywall.org/v2/{doi}?email={self.unpaywall_email}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('is_oa', False) and data.get('best_oa_location'):
                    pdf_url = data['best_oa_location']['url']
                    
                    # Try to download PDF
                    pdf_response = self.session.get(pdf_url, timeout=30)
                    if pdf_response.status_code == 200:
                        content_type = pdf_response.headers.get('content-type', '').lower()
                        if 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf'):
                            if len(pdf_response.content) > 10000:
                                logger.info(f"✓ Found open access PDF via Unpaywall: {doi}")
                                return pdf_response.content, "unpaywall"
        
        except Exception as e:
            logger.debug(f"Unpaywall failed for {doi}: {e}")
        
        return None
    
    def fetch_from_other_sources(self, metadata: PaperMetadata) -> Optional[Tuple[bytes, str]]:
        """Try additional academic sources"""
        doi = metadata.doi
        title = metadata.title.lower()
        authors = ' '.join(metadata.authors[:2]).lower() if metadata.authors else ''
        
        # Try arXiv if title suggests it's a preprint
        if any(term in title for term in ['arxiv', 'preprint', 'working paper']):
            try:
                # Search arXiv API
                arxiv_query = f"ti:{metadata.title.split()[0]}"
                arxiv_url = f"http://export.arxiv.org/api/query?search_query={arxiv_query}&max_results=1"
                
                response = self.session.get(arxiv_url, timeout=10)
                if response.status_code == 200:
                    # Parse arXiv response (simplified)
                    if 'arxiv.org' in response.text:
                        # Extract PDF URL
                        pdf_match = re.search(r'<link href="([^"]+\.pdf)"', response.text)
                        if pdf_match:
                            pdf_url = pdf_match.group(1)
                            pdf_response = self.session.get(pdf_url, timeout=30)
                            if pdf_response.status_code == 200:
                                return pdf_response.content, "arxiv"
            except:
                pass
        
        # Try ResearchGate
        try:
            search_url = f"https://www.researchgate.net/search/publication?q={requests.utils.quote(metadata.title)}"
            self.session.get(search_url, timeout=10)
            # ResearchGate requires JavaScript rendering, so simple requests may not work
        except:
            pass
        
        return None
    
    def clean_doi(self, doi: str) -> str:
        """Clean and normalize DOI"""
        doi = doi.strip()
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        return doi
    
    def get_filename_from_doi(self, doi: str) -> str:
        """Generate a safe filename from DOI"""
        safe_name = re.sub(r'[^\w\-.]', '_', doi)
        return f"{safe_name}.pdf"
    
    def download_paper(self, metadata: PaperMetadata, output_path: Path) -> Optional[Dict[str, Any]]:
        """
        Download a single paper using all available sources
        Returns: Dict with download info or None if failed
        """
        doi = metadata.doi
        
        # Check if already downloaded
        if output_path.exists():
            file_size = output_path.stat().st_size
            if file_size > 10000:  # Reasonable PDF size
                logger.info(f"✓ Paper already exists: {output_path}")
                return {
                    'doi': doi,
                    'path': str(output_path),
                    'source': 'cache',
                    'success': True,
                    'file_size': file_size
                }
        
        # Try sources in order of preference
        sources = [
            ('unpaywall', lambda: self.fetch_from_unpaywall(doi)),
            ('scihub', lambda: self.fetch_from_scihub(doi)),
            ('other_sources', lambda: self.fetch_from_other_sources(metadata)),
            ('annas_archive', lambda: self.fetch_from_annas_archive(metadata)),
        ]
        
        for source_name, fetch_func in sources:
            try:
                result = fetch_func()
                if result:
                    pdf_content, source_detail = result
                    
                    # Validate PDF
                    if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
                        # Save PDF
                        with open(output_path, 'wb') as f:
                            f.write(pdf_content)
                        
                        logger.info(f"✓ Downloaded via {source_name}: {metadata.title[:50]}...")
                        
                        return {
                            'doi': doi,
                            'path': str(output_path),
                            'source': source_detail,
                            'success': True,
                            'file_size': len(pdf_content),
                            'metadata': metadata.to_dict()
                        }
                    
            except Exception as e:
                logger.debug(f"Source {source_name} failed: {e}")
                continue
        
        logger.warning(f"✗ Failed to download: {metadata.title[:50]}...")
        return None
    
    def intelligent_paper_collection(self, topic: str, target_papers: int = 50, 
                                   min_year: int = 2010) -> Dict[str, Any]:
        """
        Main intelligent paper collection pipeline with iterative improvement
        
        Args:
            topic: Research topic
            target_papers: Target number of papers to collect
            min_year: Minimum publication year
            
        Returns:
            Comprehensive results dictionary
        """
        logger.info(f"Starting intelligent paper collection for topic: '{topic}'")
        logger.info(f"Target: {target_papers} papers, Min year: {min_year}")
        
        all_papers = []
        collected_papers = []
        failed_papers = []
        
        # Initial search
        logger.info("\n=== Initial Crossref Search ===")
        initial_papers = self.search_crossref(
            query=topic, 
            max_results=target_papers * 2,  # Start with more papers
            min_year=min_year
        )
        
        all_papers.extend(initial_papers)
        
        # Download papers
        iteration = 1
        while iteration <= self.max_iterations:
            logger.info(f"\n=== Iteration {iteration}/{self.max_iterations} ===")
            
            # Filter out already processed papers
            processed_dois = {p['doi'] for p in collected_papers}
            papers_to_process = [p for p in all_papers if p.doi not in processed_dois]
            
            if not papers_to_process:
                logger.info("No new papers to process")
                break
            
            # Process papers in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_paper = {}
                
                for paper in papers_to_process[:target_papers * 2]:
                    filename = self.get_filename_from_doi(paper.doi)
                    output_path = self.output_dir / filename
                    
                    future = executor.submit(self.download_paper, paper, output_path)
                    future_to_paper[future] = paper
                
                # Collect results
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    try:
                        result = future.result(timeout=60)
                        if result and result.get('success'):
                            collected_papers.append(result)
                        else:
                            failed_papers.append(paper)
                    except Exception as e:
                        logger.error(f"Error processing {paper.doi}: {e}")
                        failed_papers.append(paper)
            
            # Calculate success rate
            total_processed = len(collected_papers) + len(failed_papers)
            if total_processed > 0:
                success_rate = len(collected_papers) / total_processed
            else:
                success_rate = 0
            
            logger.info(f"Iteration {iteration} Results:")
            logger.info(f"  Collected: {len(collected_papers)} papers")
            logger.info(f"  Failed: {len(failed_papers)} papers")
            logger.info(f"  Success rate: {success_rate:.1%}")
            
            # Check if we should continue
            if success_rate >= self.min_success_rate and len(collected_papers) >= target_papers:
                logger.info(f"✓ Target achieved! Success rate: {success_rate:.1%}")
                break
            
            # If success rate is too low, expand search
            if success_rate < self.min_success_rate and iteration < self.max_iterations:
                logger.info(f"Success rate too low ({success_rate:.1%}), expanding search...")
                
                # Generate expanded queries
                expanded_queries = self.expand_search_queries(
                    topic, success_rate, failed_papers[-10:]  # Last 10 failed papers
                )
                
                # Search with expanded queries
                new_papers = []
                for expanded_query in expanded_queries[:3]:  # Try top 3 expansions
                    logger.info(f"  Searching with: '{expanded_query}'")
                    
                    additional_papers = self.search_crossref(
                        query=expanded_query,
                        max_results=target_papers // 2,
                        min_year=min_year
                    )
                    
                    # Filter out duplicates
                    existing_dois = {p.doi for p in all_papers}
                    unique_papers = [p for p in additional_papers if p.doi not in existing_dois]
                    
                    new_papers.extend(unique_papers)
                    
                    if len(new_papers) >= target_papers:
                        break
                
                if new_papers:
                    logger.info(f"  Found {len(new_papers)} new papers")
                    all_papers.extend(new_papers)
                
                # Add small delay before next iteration
                time.sleep(2)
            
            iteration += 1
        
        # Final results
        logger.info("\n" + "="*60)
        logger.info("PAPER COLLECTION COMPLETE")
        logger.info("="*60)
        
        total_attempted = len(collected_papers) + len(failed_papers)
        final_success_rate = len(collected_papers) / total_attempted if total_attempted > 0 else 0
        
        # Generate summary
        summary = {
            'topic': topic,
            'total_collected': len(collected_papers),
            'total_failed': len(failed_papers),
            'success_rate': final_success_rate,
            'iterations': iteration - 1,
            'collected_papers': collected_papers,
            'failed_dois': [p.doi for p in failed_papers],
            'sources_used': Counter([p.get('source', 'unknown') for p in collected_papers]),
            'collection_date': datetime.now().isoformat(),
        }
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save collection results to files"""
        # Save JSON summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_safe = re.sub(r'[^\w\-]', '_', results['topic'])[:50]
        
        summary_file = self.output_dir / f"collection_summary_{topic_safe}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save bibliography
        bib_file = self.output_dir / f"bibliography_{topic_safe}_{timestamp}.txt"
        with open(bib_file, 'w') as f:
            f.write(f"Bibliography for: {results['topic']}\n")
            f.write(f"Generated: {results['collection_date']}\n")
            f.write(f"Total papers: {results['total_collected']}\n")
            f.write("="*60 + "\n\n")
            
            for paper in results['collected_papers']:
                meta = paper.get('metadata', {})
                f.write(f"Title: {meta.get('title', 'Unknown')}\n")
                f.write(f"Authors: {', '.join(meta.get('authors', []))}\n")
                f.write(f"Journal: {meta.get('journal', 'Unknown')}\n")
                f.write(f"Year: {meta.get('year', 'Unknown')}\n")
                f.write(f"DOI: {paper.get('doi', 'Unknown')}\n")
                f.write(f"PDF: {paper.get('path', 'Unknown')}\n")
                f.write(f"Source: {paper.get('source', 'Unknown')}\n")
                f.write("-"*40 + "\n")
        
        # Save DOIs list
        dois_file = self.output_dir / f"dois_{topic_safe}_{timestamp}.txt"
        with open(dois_file, 'w') as f:
            for paper in results['collected_papers']:
                f.write(f"{paper.get('doi', '')}\n")
        
        logger.info(f"✓ Results saved to:")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Bibliography: {bib_file}")
        logger.info(f"  DOIs: {dois_file}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed report"""
        report = [
            "="*70,
            "INTELLIGENT PAPER COLLECTION REPORT",
            "="*70,
            f"Topic: {results['topic']}",
            f"Collection Date: {results['collection_date']}",
            f"Iterations: {results['iterations']}",
            "",
            "SUMMARY",
            "-"*70,
            f"Total Papers Collected: {results['total_collected']}",
            f"Total Papers Failed: {results['total_failed']}",
            f"Success Rate: {results['success_rate']:.1%}",
            "",
            "SOURCES USED",
            "-"*70,
        ]
        
        for source, count in results['sources_used'].items():
            report.append(f"  {source}: {count} papers ({count/results['total_collected']*100:.1f}%)")
        
        report.append("")
        report.append("TOP 10 PAPERS")
        report.append("-"*70)
        
        # Sort by file size (proxy for completeness)
        sorted_papers = sorted(results['collected_papers'], 
                             key=lambda x: x.get('file_size', 0), 
                             reverse=True)[:10]
        
        for i, paper in enumerate(sorted_papers, 1):
            meta = paper.get('metadata', {})
            title = meta.get('title', 'Unknown')[:60] + "..." if len(meta.get('title', '')) > 60 else meta.get('title', 'Unknown')
            report.append(f"{i}. {title}")
            report.append(f"   Authors: {', '.join(meta.get('authors', ['Unknown'])[:2])}")
            report.append(f"   Source: {paper.get('source', 'Unknown')}")
            report.append(f"   Size: {paper.get('file_size', 0) / 1024 / 1024:.2f} MB")
            report.append("")
        
        if results['failed_dois']:
            report.append("FAILED DOIs (first 20)")
            report.append("-"*70)
            for doi in results['failed_dois'][:20]:
                report.append(f"  - {doi}")
        
        report.append("="*70)
        
        return "\n".join(report)


# Advanced usage with user interaction
class InteractiveResearchAssistant:
    """Interactive CLI for the research pipeline"""
    
    def __init__(self):
        self.pipeline = SmartResearchPipeline()
        
    def run(self):
        """Run interactive assistant"""
        print("\n" + "="*70)
        print("INTELLIGENT RESEARCH PAPER COLLECTOR")
        print("="*70)
        print("This tool will help you collect research papers on any topic.")
        print("It searches Crossref, fetches papers from multiple sources,")
        print("and uses AI to improve success rates.\n")
        
        # Get topic
        topic = input("Enter your research topic: ").strip()
        if not topic:
            print("Topic cannot be empty!")
            return
        
        # Get parameters
        try:
            target_papers = int(input("How many papers do you want? (default: 30): ") or "30")
            min_year = int(input("Minimum publication year? (default: 2015): ") or "2015")
            
            # Optional: Set Unpaywall email
            use_unpaywall = input("Use Unpaywall (legal open access)? (y/n, default: y): ").lower() != 'n'
            if use_unpaywall:
                email = input("Enter your email for Unpaywall API (optional): ").strip()
                if email:
                    self.pipeline.unpaywall_email = email
            
            # Optional: Use semantic search
            use_semantic = input("Use AI semantic search? (y/n, default: y): ").lower() != 'n'
            if not use_semantic:
                global HAS_EMBEDDINGS
                HAS_EMBEDDINGS = False
            
        except ValueError:
            print("Invalid input! Using defaults.")
            target_papers = 30
            min_year = 2015
        
        print("\n" + "="*70)
        print(f"Starting collection for topic: '{topic}'")
        print(f"Target: {target_papers} papers since {min_year}")
        print("="*70 + "\n")
        
        # Start collection
        results = self.pipeline.intelligent_paper_collection(
            topic=topic,
            target_papers=target_papers,
            min_year=min_year
        )
        
        # Show report
        report = self.pipeline.generate_report(results)
        print("\n" + report)
        
        # Save caches
        self.pipeline._save_caches()
        
        print("\n✓ Collection complete!")
        print(f"✓ Papers saved in: {self.pipeline.output_dir}")
        print("="*70)


# Quick start function
def quick_collect(topic: str, target_papers: int = 30, 
                 min_year: int = 2015, email: str = None):
    """
    Quick start function for paper collection
    
    Args:
        topic: Research topic
        target_papers: Number of papers to collect
        min_year: Minimum publication year
        email: Email for Unpaywall API (optional)
    """
    pipeline = SmartResearchPipeline()
    
    if email:
        pipeline.unpaywall_email = email
    
    print(f"Starting quick collection for: '{topic}'")
    print(f"Target: {target_papers} papers since {min_year}")
    print("-" * 50)
    
    results = pipeline.intelligent_paper_collection(
        topic=topic,
        target_papers=target_papers,
        min_year=min_year
    )
    
    # Print summary
    print("\n" + "="*50)
    print("QUICK COLLECTION RESULTS")
    print("="*50)
    print(f"Papers collected: {results['total_collected']}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Iterations: {results['iterations']}")
    
    if results['total_collected'] > 0:
        print(f"\nTop sources:")
        for source, count in results['sources_used'].most_common(3):
            print(f"  {source}: {count} papers")
    
    pipeline._save_caches()
    return results


# Example usage
if __name__ == "__main__":
    # Method 1: Interactive mode
    # assistant = InteractiveResearchAssistant()
    # assistant.run()
    
    # Method 2: Quick collection (uncomment and modify)
    results = quick_collect(
        topic="machine learning fairness",
        target_papers=20,
        min_year=2018,
        email="sarvadnya.bhatlawande2005@gmail.com"  # Optional
    )
    
    # Method 3: Direct API usage
    # pipeline = SmartResearchPipeline()
    # pipeline.unpaywall_email = "your-email@example.com"
    # 
    # results = pipeline.intelligent_paper_collection(
    #     topic="quantum computing",
    #     target_papers=25,
    #     min_year=2020
    # )