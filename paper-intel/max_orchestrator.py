#!/usr/bin/env python3
"""
MAXIMUM COVERAGE ORCHESTRATOR - 10X Search + 100% Delivery
Strategy:
1. CrossRef 10X search (massive paper pool)
2. Semantic filter
3. UNPAYWALL FIRST for ALL DOIs (highest priority)
4. Sci-Hub/Anna's Archive/LibGen for remaining papers
5. If <100% success → BACKPROPAGATE → Expand search → Retry
6. Other sources: OpenAlex, CORE, S2, arXiv
"""

import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import Counter
import json
import re
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.discovery import (
    CrossRefClient, UnpaywallClient, ArxivClient, 
    OpenAlexClient, COREClient, SciHubClient, AnnasArchiveClient,
    SemanticScholarClient
)
from ingestion.models import PaperMetadata
from ingestion.auto_processor import get_auto_processor

# Semantic filtering
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class MaxCoverageOrchestrator:
    """
    Maximum coverage with backpropagation:
    - 10X CrossRef search
    - Unpaywall PRIORITY
    - Shadow libraries properly implemented
    - Auto-retry if not 100%
    """
    
    def __init__(self, output_dir: str = "research_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load API keys
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        email = os.getenv('OPENALEX_EMAIL', 'research@sanshodhak.org')
        core_key = os.getenv('CORE_API_KEY', 'fU7PRKOIE5Nz3roXDSvCwVZ46Wd2TpYi')
        s2_key = os.getenv('SEMANTICSCHOLAR_API_KEY')
        
        print("🔧 Initializing ALL clients...")
        
        # Initialize ALL clients
        self.crossref = CrossRefClient(email=email)
        self.unpaywall = UnpaywallClient(email=email)
        self.scihub = SciHubClient()
        self.annas = AnnasArchiveClient()
        self.openalex = OpenAlexClient(email=email)
        self.core = COREClient(api_key=core_key)
        self.s2 = SemanticScholarClient(api_key=s2_key)
        self.arxiv = ArxivClient()
        
        # Auto-processor for immediate PDF processing
        self.auto_processor = get_auto_processor()
        
        # Embedding model
        self.embedding_model = None
        
        if HAS_EMBEDDINGS:
            print("🧠 Loading semantic model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic model loaded")
            except:
                pass
        
        print("\n✅ ALL CLIENTS READY")
        print("   📚 Primary: CrossRef (10X search)")
        print("   🎯 Priority: Unpaywall (for ALL DOIs)")
        print("   🏴‍☠️ Shadow: Sci-Hub, Anna's Archive, LibGen")
        print("   ✅ Backup: OpenAlex, CORE, S2, arXiv")
        print("   🔄 Auto-retry: Backpropagate if <100%")
        print("   ⚡ Auto-process: Immediate PDF → JSON conversion")
    
    def get_user_input(self):
        """Get user input."""
        print("\n" + "="*80)
        print("🚀 MAX COVERAGE ORCHESTRATOR - 10X SEARCH + 100% DELIVERY")
        print("="*80)
        print("\n📋 Strategy:")
        print("   1. CrossRef 10X search → Massive paper pool")
        print("   2. Semantic filter → Best papers")
        print("   3. UNPAYWALL FIRST → All DOIs tested")
        print("   4. Shadow libraries → Sci-Hub/Anna's/LibGen")
        print("   5. Backpropagate → If <100%, expand & retry")
        print("\n" + "-"*80)
        
        topic = input("\n📋 Research topic: ").strip()
        if not topic:
            print("❌ Topic required!")
            sys.exit(1)
        
        try:
            num = input("📊 Target papers (default: 25): ").strip()
            num_papers = int(num) if num else 25
        except:
            num_papers = 25
        
        try:
            yr = input("📅 Min year (default: 2015): ").strip()
            min_year = int(yr) if yr else 2015
        except:
            min_year = 2015
        
        return topic, num_papers, min_year
    
    def massive_crossref_search(self, topic: str, count: int, min_year: int, multiplier: int = 10):
        """
        PHASE 1: Massive CrossRef 10X search.
        Search for 10x papers to have huge pool for semantic filtering.
        """
        print(f"\n🔍 PHASE 1: Massive CrossRef Search ({multiplier}X)")
        print("-"*80)
        print(f"   Topic: {topic}")
        print(f"   Target: {count} papers")
        print(f"   Searching: {count * multiplier} papers ({multiplier}X)")
        print(f"   Min year: {min_year}")
        
        all_papers = []
        
        # Search in chunks (CrossRef max is 1000)
        chunk_size = 100
        total_needed = count * multiplier
        
        for offset in range(0, total_needed, chunk_size):
            remaining = min(chunk_size, total_needed - offset)
            print(f"   Fetching {offset}-{offset+remaining}...", end='\r')
            
            try:
                result = self.crossref.search(topic, limit=remaining, min_year=min_year)
                if result.success and result.papers:
                    all_papers.extend(result.papers)
            except Exception as e:
                logger.error(f"CrossRef chunk failed: {e}")
            
            if len(all_papers) >= total_needed:
                break
        
        print(f"\n✅ CrossRef: {len(all_papers)} papers")
        
        if all_papers:
            doi_count = sum(1 for p in all_papers if p.doi)
            abstract_count = sum(1 for p in all_papers if p.abstract)
            print(f"   With DOI: {doi_count} ({doi_count/len(all_papers)*100:.0f}%)")
            print(f"   With abstract: {abstract_count}")
            
            years = [p.year for p in all_papers if p.year]
            if years:
                print(f"   Year range: {min(years)}-{max(years)}")
        
        return all_papers
    
    def strong_deduplicate(self, papers: list):
        """Strong deduplication by DOI + title hash."""
        print(f"\n🔄 Deduplication")
        print(f"   Input: {len(papers)} papers")
        
        seen_doi = set()
        seen_title_hash = set()
        unique = []
        
        for paper in papers:
            if paper.doi:
                doi_norm = paper.doi.lower().replace('https://doi.org/', '').replace('http://dx.doi.org/', '').strip()
                if doi_norm in seen_doi:
                    continue
                seen_doi.add(doi_norm)
            
            title_clean = re.sub(r'[^\w\s]', '', paper.title.lower())
            title_clean = ' '.join(title_clean.split())
            title_hash = hashlib.md5(title_clean.encode()).hexdigest()[:16]
            
            if title_hash in seen_title_hash:
                continue
            seen_title_hash.add(title_hash)
            
            unique.append(paper)
        
        print(f"   Output: {len(unique)} unique ({len(papers)-len(unique)} removed)")
        return unique
    
    def adaptive_semantic_filter(self, query: str, papers: list, target: int):
        """Adaptive semantic filtering with multiple thresholds."""
        print(f"\n🧠 PHASE 2: Semantic Filtering")
        print("-"*80)
        print(f"   Input: {len(papers)} papers")
        print(f"   Target: {target} papers")
        
        if not self.embedding_model:
            print(f"   ⚠️  No embeddings, using top {target}")
            return papers[:target]
        
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return papers[:target]
        
        # Score all papers
        print(f"   Computing similarities...")
        scored = []
        for paper in papers:
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract[:500]
            
            paper_emb = self._get_embedding(text)
            if paper_emb is not None:
                sim = float(np.dot(query_emb, paper_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(paper_emb)))
                scored.append((paper, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Try thresholds adaptively
        thresholds = [0.40, 0.30, 0.20, 0.10, 0.05]
        
        for thresh in thresholds:
            filtered = [(p, s) for p, s in scored if s >= thresh]
            if len(filtered) >= target:
                selected = filtered[:target * 2]  # Get 2x for backup
                print(f"   ✅ Threshold {thresh:.2f}: {len(selected)} papers")
                print(f"   Similarity: {selected[-1][1]:.3f} - {selected[0][1]:.3f}")
                return [p for p, _ in selected]
        
        # If still not enough, return all
        print(f"   ⚠️  Using all {len(scored)} papers")
        return [p for p, _ in scored]
    
    def _get_embedding(self, text: str):
        """Get embedding."""
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = ' '.join(text.split())[:1000]
            return self.embedding_model.encode(text, show_progress_bar=False)
        except:
            return None
    
    def download_with_backpropagation(self, papers: list, target: int, topic: str, min_year: int):
        """
        PHASE 3: Download with backpropagation.
        If we don't get 100%, expand search and retry.
        """
        max_rounds = 3
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n📥 PHASE 3: Download (Round {round_num}/{max_rounds})")
            print("-"*80)
            print(f"   Pool: {len(papers)} papers")
            print(f"   Target: {target} papers")
            print("   Priority: UNPAYWALL → SCI-HUB → ANNA'S → OTHERS")
            print("-"*80)
            
            downloaded, failed = self._download_batch(papers[:target * 2])  # Try 2x papers
            
            success_rate = len(downloaded) / target * 100
            
            print(f"\n📊 Round {round_num} Results:")
            print(f"   Downloaded: {len(downloaded)}/{target} ({success_rate:.0f}%)")
            print(f"   Failed: {len(failed)}")
            
            # Check if we reached target
            if len(downloaded) >= target:
                print(f"   ✅ TARGET REACHED!")
                return downloaded[:target], []
            
            # Backpropagate: expand search
            if round_num < max_rounds:
                print(f"\n🔄 BACKPROPAGATION: Expanding search...")
                
                # Get more papers from CrossRef
                more_papers = self.massive_crossref_search(
                    topic, 
                    target * 2, 
                    min_year, 
                    multiplier=5 * round_num  # Increase multiplier each round
                )
                
                more_papers = self.strong_deduplicate(more_papers)
                more_papers = self.adaptive_semantic_filter(topic, more_papers, target * 2)
                
                # Add new papers to pool
                papers.extend(more_papers)
                papers = self.strong_deduplicate(papers)
                
                print(f"   New pool size: {len(papers)} papers")
        
        # Final result
        return downloaded, failed
    
    def _download_batch(self, papers: list):
        """Download batch of papers."""
        downloaded = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for idx, paper in enumerate(papers, 1):
                future = executor.submit(self._download_single, paper, idx, len(papers))
                futures.append((future, paper))
            
            for future, paper in futures:
                try:
                    result = future.result(timeout=120)
                    if result:
                        downloaded.append(result)
                    else:
                        failed.append(paper)
                except Exception as e:
                    failed.append(paper)
        
        return downloaded, failed
    
    def _download_single(self, paper: PaperMetadata, idx: int, total: int):
        """
        Download single paper.
        PRIORITY: UNPAYWALL → SCI-HUB → ANNA'S → OPENALEX → CORE → S2 → ARXIV
        """
        title = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
        print(f"\n[{idx}/{total}] {title}")
        
        filename = f"{idx:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # 1. UNPAYWALL FIRST (highest priority for DOIs)
        if paper.doi:
            result = self._try_unpaywall(paper.doi)
            if result:
                return self._save(result, paper, filepath, 'unpaywall')
        
        # 2. SCI-HUB (shadow library)
        if paper.doi:
            result = self._try_scihub_proper(paper.doi)
            if result:
                return self._save(result, paper, filepath, 'scihub')
        
        # 3. ANNA'S ARCHIVE (shadow library)
        if paper.doi:
            result = self._try_annas_proper(paper.doi, paper.title)
            if result:
                return self._save(result, paper, filepath, 'annas')
        
        # 4. OPENALEX
        result = self._try_openalex(paper)
        if result:
            return self._save(result, paper, filepath, 'openalex')
        
        # 5. CORE
        result = self._try_core(paper)
        if result:
            return self._save(result, paper, filepath, 'core')
        
        # 6. SEMANTIC SCHOLAR
        result = self._try_s2(paper)
        if result:
            return self._save(result, paper, filepath, 's2')
        
        # 7. ARXIV
        result = self._try_arxiv(paper)
        if result:
            return self._save(result, paper, filepath, 'arxiv')
        
        print(f"   ❌ All sources failed")
        return None
    
    def _save(self, pdf_content, paper, filepath, source):
        """Save PDF and trigger auto-processing."""
        if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
            filepath.write_bytes(pdf_content)
            size_mb = len(pdf_content) / (1024 * 1024)
            emoji = "🏴‍☠️" if source in ['scihub', 'annas'] else "✅"
            print(f"   {emoji} {source.upper()} ({size_mb:.2f} MB)")
            
            # Auto-process the PDF immediately
            try:
                self.auto_processor.process_downloaded_pdf(str(filepath))
                print(f"   ⚡ Processed → JSON")
            except Exception as e:
                logger.warning(f"Auto-processing failed: {e}")
            
            return {'paper': paper, 'path': str(filepath), 'source': source}
        return None
    
    def _try_unpaywall(self, doi: str):
        """Try Unpaywall."""
        try:
            info = self.unpaywall.get_pdf_url(doi)
            if info and info.get('pdf_url'):
                import requests
                resp = requests.get(
                    info['pdf_url'], 
                    timeout=30, 
                    allow_redirects=True,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                if resp.status_code == 200 and len(resp.content) > 10000:
                    return resp.content
        except:
            pass
        return None
    
    def _try_scihub_proper(self, doi: str):
        """
        Try Sci-Hub PROPERLY.
        Sci-Hub mirrors: sci-hub.se, sci-hub.st, sci-hub.ru, etc.
        """
        try:
            # Use the SciHubClient
            result = self.scihub.get_pdf(doi)
            if result:
                pdf_content, _ = result
                if len(pdf_content) > 10000:
                    return pdf_content
            
            # Fallback: Direct Sci-Hub URL attempts
            import requests
            
            mirrors = [
                'https://sci-hub.se',
                'https://sci-hub.st', 
                'https://sci-hub.ru',
                'https://sci-hub.mksa.top'
            ]
            
            for mirror in mirrors:
                try:
                    url = f"{mirror}/{doi}"
                    resp = requests.get(
                        url,
                        timeout=20,
                        allow_redirects=True,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    )
                    
                    if resp.status_code == 200:
                        # Parse HTML to find PDF link
                        if b'<iframe' in resp.content or b'<embed' in resp.content:
                            # Extract PDF URL from iframe/embed
                            import re
                            pdf_urls = re.findall(rb'(?:src|href)=["\']([^"\']*\.pdf[^"\']*)["\']', resp.content)
                            
                            for pdf_url in pdf_urls:
                                pdf_url = pdf_url.decode('utf-8')
                                if not pdf_url.startswith('http'):
                                    pdf_url = mirror + pdf_url
                                
                                pdf_resp = requests.get(pdf_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
                                if pdf_resp.status_code == 200 and pdf_resp.content[:4] == b'%PDF':
                                    return pdf_resp.content
                        
                        # Direct PDF response
                        if resp.content[:4] == b'%PDF':
                            return resp.content
                
                except:
                    continue
        
        except:
            pass
        
        return None
    
    def _try_annas_proper(self, doi: str, title: str):
        """
        Try Anna's Archive PROPERLY.
        Anna's Archive is LibGen mirror.
        """
        try:
            # Use AnnasArchiveClient
            result = self.annas.search_by_doi(doi)
            if result and result.get('pdf_url'):
                import requests
                resp = requests.get(
                    result['pdf_url'],
                    timeout=30,
                    allow_redirects=True,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                if resp.status_code == 200 and resp.content[:4] == b'%PDF':
                    return resp.content
            
            # Fallback: LibGen direct
            import requests
            
            # Try LibGen by DOI
            libgen_url = f"http://libgen.rs/scimag/?q={doi}"
            resp = requests.get(libgen_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            
            if resp.status_code == 200:
                # Parse for download links
                import re
                links = re.findall(rb'href=["\']([^"\']*download[^"\']*)["\']', resp.content)
                
                for link in links[:3]:  # Try first 3 links
                    try:
                        link = link.decode('utf-8')
                        if not link.startswith('http'):
                            link = 'http://libgen.rs' + link
                        
                        pdf_resp = requests.get(link, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
                        if pdf_resp.status_code == 200 and pdf_resp.content[:4] == b'%PDF':
                            return pdf_resp.content
                    except:
                        continue
        
        except:
            pass
        
        return None
    
    def _try_openalex(self, paper):
        """Try OpenAlex."""
        try:
            if paper.doi:
                doi_clean = paper.doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                params = {'filter': f'doi:{doi_clean}', 'per-page': 1}
            else:
                params = {'filter': f'title.search:{paper.title[:100]}', 'per-page': 1}
            
            response = self.openalex.get('/works', params=params)
            
            if response and response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results:
                    work = results[0]
                    best_oa = work.get('best_oa_location')
                    if best_oa and best_oa.get('pdf_url'):
                        import requests
                        resp = requests.get(best_oa['pdf_url'], timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                        if resp.status_code == 200:
                            return resp.content
        except:
            pass
        return None
    
    def _try_core(self, paper):
        """Try CORE."""
        try:
            result = self.core.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    return resp.content
        except:
            pass
        return None
    
    def _try_s2(self, paper):
        """Try Semantic Scholar."""
        try:
            if paper.source == 'semanticscholar' and paper.pdf_url:
                import requests
                resp = requests.get(paper.pdf_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    return resp.content
        except:
            pass
        return None
    
    def _try_arxiv(self, paper):
        """Try arXiv."""
        try:
            result = self.arxiv.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    return resp.content
        except:
            pass
        return None
    
    def _safe_filename(self, paper):
        """Safe filename."""
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        author = re.sub(r'[^\w]', '', author)[:15]
        year = paper.year or "0000"
        return f"{author}_{year}.pdf"
    
    def generate_report(self, topic, downloaded, target, topic_dir):
        """Generate report."""
        print(f"\n📝 Generating Report...")
        
        source_counts = Counter([d['source'] for d in downloaded])
        
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'downloaded': len(downloaded),
            'success_rate': f"{len(downloaded)/target*100:.1f}%",
            'sources': dict(source_counts),
            'papers': [{
                'title': d['paper'].title,
                'authors': d['paper'].authors[:3],
                'year': d['paper'].year,
                'doi': d['paper'].doi,
                'file': Path(d['path']).name,
                'source': d['source']
            } for d in downloaded]
        }
        
        (topic_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        report = f"""# {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Target:** {target} papers  
**Downloaded:** {len(downloaded)} ({len(downloaded)/target*100:.0f}%)

## Sources

"""
        for source, count in source_counts.most_common():
            pct = count / len(downloaded) * 100
            emoji = "🏴‍☠️" if source in ['scihub', 'annas'] else "✅"
            report += f"- {emoji} **{source.upper()}**: {count} ({pct:.0f}%)\n"
        
        report += "\n## Papers\n\n"
        for i, d in enumerate(downloaded, 1):
            p = d['paper']
            report += f"{i}. **{p.title}** ({p.year})\n"
            report += f"   - Authors: {', '.join(p.authors[:3])}\n"
            report += f"   - Source: {d['source']}\n"
            report += f"   - File: `{Path(d['path']).name}`\n\n"
        
        (topic_dir / "REPORT.md").write_text(report)
        print(f"✅ Report saved")
    
    def run(self):
        """Main flow."""
        topic, num_papers, min_year = self.get_user_input()
        
        topic_slug = re.sub(r'[^\w\s]', '', topic.lower()).replace(' ', '_')[:50]
        topic_dir = self.output_dir / topic_slug
        topic_dir.mkdir(exist_ok=True)
        self.output_dir = topic_dir
        
        print("\n" + "="*80)
        print("🚀 STARTING MAX COVERAGE COLLECTION")
        print("="*80)
        
        # Phase 1: Massive CrossRef search
        papers = self.massive_crossref_search(topic, num_papers, min_year, multiplier=10)
        
        if not papers:
            print("\n❌ No papers found!")
            return
        
        # Deduplication
        papers = self.strong_deduplicate(papers)
        
        # Phase 2: Semantic filtering
        papers = self.adaptive_semantic_filter(topic, papers, num_papers)
        
        # Phase 3: Download with backpropagation
        downloaded, failed = self.download_with_backpropagation(papers, num_papers, topic, min_year)
        
        # Report
        self.generate_report(topic, downloaded, num_papers, topic_dir)
        
        # Summary
        print("\n" + "="*80)
        print("✅ COLLECTION COMPLETE!")
        print("="*80)
        print(f"\n📁 {topic_dir.absolute()}")
        print(f"📊 {len(downloaded)}/{num_papers} papers ({len(downloaded)/num_papers*100:.0f}%)")
        
        print(f"\n📥 Sources:")
        for src, cnt in Counter([d['source'] for d in downloaded]).most_common():
            pct = cnt / len(downloaded) * 100
            emoji = "🏴‍☠️" if src in ['scihub', 'annas'] else "✅"
            print(f"   {emoji} {src}: {cnt} ({pct:.0f}%)")
        
        if downloaded:
            total_mb = sum(Path(d['path']).stat().st_size for d in downloaded) / (1024*1024)
            print(f"\n💾 Total: {total_mb:.1f} MB")
        
        print("\n" + "="*80 + "\n")


def main():
    try:
        orch = MaxCoverageOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
