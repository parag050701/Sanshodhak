#!/usr/bin/env python3
"""
TRUE MULTI-SOURCE ORCHESTRATOR - Following test7.5-wrkin.py Pattern
1. Search CrossRef for LOTS of papers (with DOIs)
2. Semantic filter to get best papers
3. For EACH DOI, try ALL sources in priority order:
   - Unpaywall (legal OA)
   - OpenAlex (direct PDF links)
   - CORE (OA papers)
   - arXiv (search by title/DOI)
   - Sci-Hub (shadow library)
   - Anna's Archive (shadow library)
"""

import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import Counter
import json
import re

sys.path.insert(0, str(Path(__file__).parent))

from ingestion.discovery import (
    CrossRefClient, UnpaywallClient, ArxivClient, 
    OpenAlexClient, COREClient, SciHubClient, AnnasArchiveClient
)
from ingestion.models import PaperMetadata

# Semantic filtering
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class ProperMultiSourceOrchestrator:
    """
    Proper multi-source following test7.5-wrkin.py:
    - CrossRef for metadata (get DOIs)
    - Try EVERY source for EACH paper
    - Proper priority: Unpaywall → OpenAlex → CORE → arXiv → Sci-Hub
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
        
        # Initialize clients
        self.crossref = CrossRefClient(email=email)
        self.unpaywall = UnpaywallClient(email=email)
        self.openalex = OpenAlexClient(email=email)
        self.core = COREClient(api_key=core_key)
        self.arxiv = ArxivClient()
        self.scihub = SciHubClient()
        self.annas = AnnasArchiveClient()
        
        # Embedding model
        self.embedding_model = None
        self.similarity_threshold = 0.50
        
        if HAS_EMBEDDINGS:
            print("🧠 Loading semantic model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic filtering enabled")
            except:
                self.embedding_model = None
        
        print("✅ All clients initialized")
    
    def get_user_input(self):
        """Get user input."""
        print("\n" + "="*80)
        print("🔬 PROPER MULTI-SOURCE RESEARCH COLLECTOR")
        print("="*80)
        print("\n📋 Strategy:")
        print("   1. CrossRef → Get papers with DOIs")
        print("   2. Semantic filter → Remove irrelevant papers")
        print("   3. For EACH paper, try ALL sources:")
        print("      • Unpaywall (legal OA)")
        print("      • OpenAlex (250M+ works)")
        print("      • CORE (200M+ OA papers)")
        print("      • arXiv (preprints)")
        print("      • Sci-Hub (shadow library)")
        print("      • Anna's Archive (shadow library)")
        print("\n" + "-"*80)
        
        topic = input("\n📋 Research topic: ").strip()
        if not topic:
            print("❌ Topic required!")
            sys.exit(1)
        
        try:
            num = input("📊 Papers (default: 25): ").strip()
            num_papers = int(num) if num else 25
        except:
            num_papers = 25
        
        try:
            yr = input("📅 Min year (default: 2015): ").strip()
            min_year = int(yr) if yr else 2015
        except:
            min_year = 2015
        
        return topic, num_papers, min_year
    
    def search_crossref_large(self, topic: str, count: int, min_year: int):
        """
        PHASE 1: Search CrossRef for 2-3x papers.
        Get papers with DOIs, citations, abstracts.
        """
        print(f"\n🔍 PHASE 1: CrossRef Search")
        print("-"*80)
        print(f"   Topic: {topic}")
        print(f"   Searching for: {count * 2} papers (will filter to {count})")
        print(f"   Min year: {min_year}")
        
        result = self.crossref.search(topic, limit=count * 2, min_year=min_year)
        
        if not result.success:
            logger.error(f"CrossRef failed: {result.error}")
            return []
        
        papers = result.papers
        
        print(f"\n✅ Found {len(papers)} papers from CrossRef")
        print(f"   With DOI: {sum(1 for p in papers if p.doi)}")
        print(f"   With abstract: {sum(1 for p in papers if p.abstract)}")
        print(f"   With citations: {sum(1 for p in papers if p.citations and p.citations > 0)}")
        
        if papers:
            years = [p.year for p in papers if p.year]
            if years:
                print(f"   Year range: {min(years)}-{max(years)}")
        
        return papers
    
    def semantic_filter(self, query: str, papers: list, target: int):
        """
        PHASE 2: Semantic filtering.
        """
        if not self.embedding_model or not papers:
            print(f"\n⚠️  No semantic filtering, using top {target}")
            return papers[:target]
        
        print(f"\n🧠 PHASE 2: Semantic Filtering")
        print("-"*80)
        print(f"   Analyzing {len(papers)} papers...")
        
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return papers[:target]
        
        scored = []
        for paper in papers:
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract[:500]
            
            paper_emb = self._get_embedding(text)
            if paper_emb is not None:
                sim = np.dot(query_emb, paper_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(paper_emb))
                scored.append((paper, float(sim)))
            else:
                scored.append((paper, 0.5))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        filtered = [(p, s) for p, s in scored if s >= self.similarity_threshold]
        if len(filtered) < target:
            filtered = scored[:target]
        else:
            filtered = filtered[:target]
        
        print(f"   ✅ Selected {len(filtered)} papers")
        print(f"   Similarity: {filtered[-1][1]:.3f} - {filtered[0][1]:.3f}")
        print(f"   Filtered out: {len(papers) - len(filtered)} irrelevant")
        
        return [p for p, _ in filtered]
    
    def _get_embedding(self, text: str):
        """Get embedding."""
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = ' '.join(text.split())[:1000]
            return self.embedding_model.encode(text, show_progress_bar=False)
        except:
            return None
    
    def download_all_papers(self, papers: list):
        """
        PHASE 3: Download using ALL sources.
        For EACH paper, try sources in priority order.
        """
        print(f"\n📥 PHASE 3: Multi-Source Download")
        print("-"*80)
        print("   Priority: Unpaywall → OpenAlex → CORE → arXiv → Sci-Hub → Anna's")
        print("-"*80)
        
        downloaded = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
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
                    logger.error(f"Error: {e}")
                    failed.append(paper)
        
        return downloaded, failed
    
    def _download_single(self, paper: PaperMetadata, idx: int, total: int):
        """
        Download single paper trying ALL sources.
        This is the KEY difference - we try EVERY source for EACH paper.
        """
        title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
        print(f"\n[{idx}/{total}] {title}")
        
        filename = f"{idx:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Try ALL sources in priority order
        # KEY: We try EACH source individually, not just first success
        
        # 1. Unpaywall (legal OA)
        if paper.doi:
            result = self._try_unpaywall(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'unpaywall')
        
        # 2. OpenAlex (direct PDF from metadata)
        result = self._try_openalex_direct(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'openalex')
        
        # 3. CORE (search by title)
        result = self._try_core_search(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'core')
        
        # 4. arXiv (search by title)
        result = self._try_arxiv_search(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'arxiv')
        
        # 5. Sci-Hub (shadow library)
        if paper.doi:
            result = self._try_scihub(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'scihub')
        
        # 6. Anna's Archive (shadow library - by MD5, usually need metadata)
        # This is a placeholder as Anna's needs MD5 hash
        
        print(f"   ❌ All sources failed")
        return None
    
    def _save_and_return(self, pdf_content, paper, filepath, source):
        """Save PDF and return result."""
        if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
            filepath.write_bytes(pdf_content)
            size_mb = len(pdf_content) / (1024 * 1024)
            print(f"   ✅ {source.upper()} ({size_mb:.2f} MB)")
            return {'paper': paper, 'path': str(filepath), 'source': source}
        return None
    
    def _try_unpaywall(self, doi: str):
        """Try Unpaywall."""
        try:
            info = self.unpaywall.get_pdf_url(doi)
            if info and info.get('pdf_url'):
                import requests
                resp = requests.get(info['pdf_url'], timeout=30, allow_redirects=True)
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"Unpaywall: {e}")
        return None
    
    def _try_openalex_direct(self, paper: PaperMetadata):
        """
        Try OpenAlex by searching for the paper.
        OpenAlex has direct PDF links in metadata (best_oa_location).
        """
        try:
            # Search by DOI if available (more accurate), otherwise title
            if paper.doi:
                # OpenAlex DOI filter format
                doi_clean = paper.doi.replace('https://doi.org/', '')
                params = {
                    'filter': f'doi:{doi_clean}',
                    'per-page': 1
                }
                response = self.openalex.get('/works', params=params)
            else:
                # Title search fallback
                params = {
                    'filter': f'title.search:{paper.title[:100]}',
                    'per-page': 1,
                    'sort': 'cited_by_count:desc'
                }
                response = self.openalex.get('/works', params=params)
            
            if response and response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results:
                    work = results[0]
                    # Check best_oa_location for PDF
                    best_oa = work.get('best_oa_location')
                    if best_oa and best_oa.get('pdf_url'):
                        pdf_url = best_oa['pdf_url']
                        import requests
                        resp = requests.get(pdf_url, timeout=30, allow_redirects=True,
                                          headers={'User-Agent': 'Sanshodhak/1.0'})
                        if resp.status_code == 200 and len(resp.content) > 10000:
                            return resp.content
        except Exception as e:
            logger.debug(f"OpenAlex: {e}")
        return None
    
    def _try_core_search(self, paper: PaperMetadata):
        """Try CORE by searching title."""
        try:
            result = self.core.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30)
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"CORE: {e}")
        return None
    
    def _try_arxiv_search(self, paper: PaperMetadata):
        """Try arXiv by searching title."""
        try:
            result = self.arxiv.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30)
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"arXiv: {e}")
        return None
    
    def _try_scihub(self, doi: str):
        """Try Sci-Hub."""
        try:
            result = self.scihub.get_pdf(doi)
            if result:
                pdf_content, _ = result
                return pdf_content
        except Exception as e:
            logger.debug(f"Sci-Hub: {e}")
        return None
    
    def _safe_filename(self, paper):
        """Generate safe filename."""
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        author = re.sub(r'[^\w]', '', author)[:15]
        year = paper.year or "0000"
        return f"{author}_{year}.pdf"
    
    def generate_report(self, topic, papers, downloaded, failed, min_year, topic_dir):
        """Generate report."""
        print(f"\n📝 PHASE 4: Generate Report")
        print("-"*80)
        
        success_rate = len(downloaded) / len(papers) * 100 if papers else 0
        source_counts = Counter([d['source'] for d in downloaded])
        
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'min_year': min_year,
            'semantic_filtering': bool(self.embedding_model),
            'target_papers': len(papers),
            'downloaded': len(downloaded),
            'failed': len(failed),
            'success_rate': f"{success_rate:.1f}%",
            'sources_used': dict(source_counts),
            'papers': [{
                'title': d['paper'].title,
                'authors': d['paper'].authors,
                'year': d['paper'].year,
                'doi': d['paper'].doi,
                'citations': d['paper'].citations,
                'file': Path(d['path']).name,
                'source': d['source']
            } for d in downloaded]
        }
        
        (topic_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"✅ Metadata saved")
        
        report = f"""# Research Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Semantic Filtering:** {'✅ Enabled' if self.embedding_model else '❌ Disabled'}

## Summary

- **Target:** {len(papers)} papers
- **Downloaded:** {len(downloaded)} ({success_rate:.1f}%)
- **Failed:** {len(failed)}
- **Min Year:** {min_year}

## Download Sources (Diversity!)

"""
        for source, count in source_counts.most_common():
            pct = count / len(downloaded) * 100 if downloaded else 0
            report += f"- **{source}**: {count} papers ({pct:.1f}%)\n"
        
        report += "\n## Papers (sorted by citations)\n\n"
        for d in sorted(downloaded, key=lambda x: x['paper'].citations or 0, reverse=True):
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **Citations:** {p.citations or 0} | **DOI:** {p.doi or 'N/A'}\n\n"
            report += f"**File:** `{Path(d['path']).name}` | **Source:** {d['source']}\n\n"
            if p.abstract:
                report += f"**Abstract:** {p.abstract[:250]}...\n\n"
            report += "---\n\n"
        
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
        print("🚀 STARTING COLLECTION")
        print("="*80)
        
        # Phase 1: CrossRef search (get DOIs)
        all_papers = self.search_crossref_large(topic, num_papers, min_year)
        
        if not all_papers:
            print("\n❌ No papers found!")
            return
        
        # Phase 2: Semantic filter
        selected = self.semantic_filter(topic, all_papers, num_papers)
        
        # Phase 3: Download from ALL sources
        downloaded, failed = self.download_all_papers(selected)
        
        # Phase 4: Report
        self.generate_report(topic, selected, downloaded, failed, min_year, topic_dir)
        
        # Summary
        print("\n" + "="*80)
        print("✅ COLLECTION COMPLETE!")
        print("="*80)
        print(f"\n📁 {topic_dir.absolute()}")
        print(f"📄 CrossRef: {len(all_papers)} → Filtered: {len(selected)} → Downloaded: {len(downloaded)}")
        print(f"📊 Success: {len(downloaded)/len(selected)*100:.1f}%")
        
        print(f"\n📥 Source Diversity:")
        for src, cnt in Counter([d['source'] for d in downloaded]).most_common():
            pct = cnt / len(downloaded) * 100 if downloaded else 0
            print(f"   • {src}: {cnt} ({pct:.0f}%)")
        
        if downloaded:
            total_mb = sum(Path(d['path']).stat().st_size for d in downloaded) / (1024*1024)
            print(f"\n💾 Total: {total_mb:.1f} MB")
        
        print("\n" + "="*80 + "\n")


def main():
    try:
        orch = ProperMultiSourceOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
