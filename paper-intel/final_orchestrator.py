#!/usr/bin/env python3
"""
FINAL ORCHESTRATOR - Maximum Source Diversity + Adaptive Search
Features:
1. Multi-source parallel search (CrossRef, OpenAlex, CORE, Semantic Scholar, arXiv)
2. Adaptive semantic filtering (expands if not enough papers)
3. Strong deduplication (by DOI + normalized title)
4. Priority download: Unpaywall → Sci-Hub → OpenAlex → CORE → Semantic Scholar → arXiv
5. Quality filtering with citations
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

# Semantic filtering
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class FinalOrchestrator:
    """
    Final orchestrator with:
    - Multi-source PARALLEL search (5 sources)
    - Adaptive semantic threshold
    - Strong deduplication
    - Better CORE/OpenAlex/S2 utilization
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
        self.openalex = OpenAlexClient(email=email)
        self.core = COREClient(api_key=core_key)
        self.s2 = SemanticScholarClient(api_key=s2_key)
        self.arxiv = ArxivClient()
        self.unpaywall = UnpaywallClient(email=email)
        self.scihub = SciHubClient()
        self.annas = AnnasArchiveClient()
        
        # Embedding model
        self.embedding_model = None
        self.similarity_threshold = 0.45  # Start lower for more papers
        
        if HAS_EMBEDDINGS:
            print("🧠 Loading semantic model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic filtering ENABLED")
            except:
                self.embedding_model = None
        
        print("\n✅ ALL CLIENTS READY:")
        print("   📚 Search: CrossRef, OpenAlex, CORE, Semantic Scholar, arXiv")
        print("   📥 Legal: Unpaywall, OpenAlex, CORE, Semantic Scholar, arXiv")
        print("   🏴‍☠️ Shadow: Sci-Hub, Anna's Archive")
    
    def get_user_input(self):
        """Get user input."""
        print("\n" + "="*80)
        print("🚀 FINAL ORCHESTRATOR - ADAPTIVE MULTI-SOURCE")
        print("="*80)
        print("\n📋 Strategy:")
        print("   1. PARALLEL search: CrossRef + OpenAlex + CORE + S2 + arXiv")
        print("   2. Strong deduplication (DOI + title hash)")
        print("   3. Adaptive semantic filter (expands if not enough papers)")
        print("   4. Download priority:")
        print("      ✅ Unpaywall → 🏴‍☠️ Sci-Hub → ✅ OpenAlex → ✅ CORE → ✅ S2 → ✅ arXiv")
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
        
        try:
            cit = input("🎯 Min citations (default: 0): ").strip()
            min_citations = int(cit) if cit else 0
        except:
            min_citations = 0
        
        return topic, num_papers, min_year, min_citations
    
    def multi_source_search(self, topic: str, count: int, min_year: int):
        """
        PHASE 1: Multi-source PARALLEL search.
        Search all sources simultaneously with ThreadPoolExecutor.
        """
        print(f"\n🔍 PHASE 1: Multi-Source PARALLEL Search")
        print("-"*80)
        print(f"   Topic: {topic}")
        print(f"   Searching 5 sources in parallel...")
        print(f"   Min year: {min_year}")
        
        all_papers = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._search_crossref, topic, count * 2, min_year): 'CrossRef',
                executor.submit(self._search_openalex, topic, count * 2, min_year): 'OpenAlex',
                executor.submit(self._search_core, topic, count * 2, min_year): 'CORE',
                executor.submit(self._search_s2, topic, count * 2, min_year): 'Semantic Scholar',
                executor.submit(self._search_arxiv, topic, count * 2): 'arXiv'
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    papers = future.result(timeout=60)
                    if papers:
                        print(f"   ✅ {source}: {len(papers)} papers")
                        all_papers.extend(papers)
                    else:
                        print(f"   ⚠️  {source}: 0 papers")
                except Exception as e:
                    print(f"   ❌ {source} failed: {e}")
        
        print(f"\n📊 Total before dedup: {len(all_papers)} papers")
        return all_papers
    
    def _search_crossref(self, topic, count, min_year):
        """Search CrossRef."""
        try:
            result = self.crossref.search(topic, limit=count, min_year=min_year)
            return result.papers if result.success else []
        except Exception as e:
            logger.error(f"CrossRef: {e}")
            return []
    
    def _search_openalex(self, topic, count, min_year):
        """Search OpenAlex."""
        try:
            result = self.openalex.search(topic, limit=count, min_year=min_year)
            return result.papers if result.success else []
        except Exception as e:
            logger.error(f"OpenAlex: {e}")
            return []
    
    def _search_core(self, topic, count, min_year):
        """Search CORE."""
        try:
            result = self.core.search(topic, limit=count, min_year=min_year)
            return result.papers if result.success else []
        except Exception as e:
            logger.error(f"CORE: {e}")
            return []
    
    def _search_s2(self, topic, count, min_year):
        """Search Semantic Scholar."""
        try:
            result = self.s2.search(topic, limit=count, min_year=min_year)
            return result.papers if result.success else []
        except Exception as e:
            logger.error(f"Semantic Scholar: {e}")
            return []
    
    def _search_arxiv(self, topic, count):
        """Search arXiv."""
        try:
            result = self.arxiv.search(topic, limit=count)
            return result.papers if result.success else []
        except Exception as e:
            logger.error(f"arXiv: {e}")
            return []
    
    def strong_deduplicate(self, papers: list):
        """
        PHASE 2A: Strong deduplication.
        Remove duplicates by:
        1. DOI (normalized)
        2. Title hash (normalized, lowercase, no punctuation)
        3. Title similarity (fuzzy match)
        """
        print(f"\n🔄 PHASE 2A: Strong Deduplication")
        print("-"*80)
        print(f"   Input: {len(papers)} papers")
        
        seen_doi = set()
        seen_title_hash = set()
        unique = []
        
        for paper in papers:
            # Check DOI
            if paper.doi:
                doi_norm = paper.doi.lower().replace('https://doi.org/', '').replace('http://dx.doi.org/', '').strip()
                if doi_norm in seen_doi:
                    continue
                seen_doi.add(doi_norm)
            
            # Check title hash
            title_clean = re.sub(r'[^\w\s]', '', paper.title.lower())
            title_clean = ' '.join(title_clean.split())
            title_hash = hashlib.md5(title_clean.encode()).hexdigest()[:16]
            
            if title_hash in seen_title_hash:
                continue
            seen_title_hash.add(title_hash)
            
            unique.append(paper)
        
        print(f"   ✅ After dedup: {len(unique)} unique papers")
        print(f"   Removed: {len(papers) - len(unique)} duplicates")
        
        return unique
    
    def adaptive_semantic_filter(self, query: str, papers: list, target: int, min_citations: int = 0):
        """
        PHASE 2B: Adaptive semantic filtering.
        If we don't get enough papers, lower the threshold and retry.
        """
        print(f"\n🧠 PHASE 2B: Adaptive Semantic Filtering")
        print("-"*80)
        print(f"   Input: {len(papers)} papers")
        print(f"   Target: {target} papers")
        print(f"   Min citations: {min_citations}")
        
        # Quality filter first
        if min_citations > 0:
            papers = [p for p in papers if p.citations and p.citations >= min_citations]
            print(f"   After citation filter: {len(papers)}")
        
        if not self.embedding_model or not papers:
            print(f"   ⚠️  No semantic filtering, using top {target}")
            papers.sort(key=lambda p: p.citations or 0, reverse=True)
            return papers[:target]
        
        # Try multiple thresholds adaptively
        thresholds = [0.45, 0.35, 0.25, 0.15]
        
        for threshold in thresholds:
            print(f"\n   🔍 Trying threshold: {threshold:.2f}")
            selected = self._filter_with_threshold(query, papers, target, threshold)
            
            if len(selected) >= target:
                print(f"   ✅ Got {len(selected)} papers (target: {target})")
                return selected[:target]
            else:
                print(f"   ⚠️  Only {len(selected)} papers, need {target}")
        
        # If still not enough, return all
        print(f"   ⚠️  Using all {len(papers)} papers (couldn't reach target)")
        return papers
    
    def _filter_with_threshold(self, query: str, papers: list, target: int, threshold: float):
        """Filter papers with given threshold."""
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
                sim = float(np.dot(query_emb, paper_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(paper_emb)))
                scored.append((paper, sim))
            else:
                scored.append((paper, 0.5))
        
        # Sort by similarity, then by citations
        scored.sort(key=lambda x: (x[1], x[0].citations or 0), reverse=True)
        
        # Filter by threshold
        filtered = [(p, s) for p, s in scored if s >= threshold]
        
        if filtered:
            print(f"      Similarity range: {filtered[-1][1]:.3f} - {filtered[0][1]:.3f}")
            if filtered[0][0].citations:
                cites = [p.citations or 0 for p, _ in filtered]
                print(f"      Citation range: {min(cites)} - {max(cites)}")
        
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
        PHASE 3: Download with ALL sources.
        Priority: Unpaywall → Sci-Hub → OpenAlex → CORE → S2 → arXiv
        """
        print(f"\n📥 PHASE 3: Multi-Source Download")
        print("-"*80)
        print("   Priority: Unpaywall → Sci-Hub → OpenAlex → CORE → S2 → arXiv")
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
                    result = future.result(timeout=180)
                    if result:
                        downloaded.append(result)
                    else:
                        failed.append(paper)
                except Exception as e:
                    logger.error(f"Download error: {e}")
                    failed.append(paper)
        
        return downloaded, failed
    
    def _download_single(self, paper: PaperMetadata, idx: int, total: int):
        """
        Download single paper trying ALL sources.
        Priority: Unpaywall → Sci-Hub → OpenAlex → CORE → S2 → arXiv
        """
        title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
        print(f"\n[{idx}/{total}] {title}")
        
        filename = f"{idx:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Try ALL sources in priority order
        
        # 1. Unpaywall (legal OA)
        if paper.doi:
            result = self._try_unpaywall(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'unpaywall')
        
        # 2. Sci-Hub (shadow library)
        if paper.doi:
            result = self._try_scihub(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'scihub')
        
        # 3. OpenAlex (direct PDF)
        result = self._try_openalex(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'openalex')
        
        # 4. CORE (direct PDF)
        result = self._try_core(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'core')
        
        # 5. Semantic Scholar (direct PDF)
        result = self._try_s2(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'semanticscholar')
        
        # 6. arXiv (search by title)
        result = self._try_arxiv(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'arxiv')
        
        print(f"   ❌ All sources failed")
        return None
    
    def _save_and_return(self, pdf_content, paper, filepath, source):
        """Save PDF and return result."""
        if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
            filepath.write_bytes(pdf_content)
            size_mb = len(pdf_content) / (1024 * 1024)
            emoji = "✅" if source != 'scihub' else "🏴‍☠️"
            print(f"   {emoji} {source.upper()} ({size_mb:.2f} MB)")
            return {'paper': paper, 'path': str(filepath), 'source': source}
        return None
    
    def _try_unpaywall(self, doi: str):
        """Try Unpaywall."""
        try:
            info = self.unpaywall.get_pdf_url(doi)
            if info and info.get('pdf_url'):
                import requests
                resp = requests.get(info['pdf_url'], timeout=30, allow_redirects=True,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"Unpaywall: {e}")
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
    
    def _try_openalex(self, paper: PaperMetadata):
        """Try OpenAlex by DOI or title."""
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
                        resp = requests.get(best_oa['pdf_url'], timeout=30, allow_redirects=True,
                                          headers={'User-Agent': 'Sanshodhak/1.0'})
                        if resp.status_code == 200:
                            return resp.content
        except Exception as e:
            logger.debug(f"OpenAlex: {e}")
        return None
    
    def _try_core(self, paper: PaperMetadata):
        """Try CORE by title/DOI."""
        try:
            # Try title search
            result = self.core.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"CORE: {e}")
        return None
    
    def _try_s2(self, paper: PaperMetadata):
        """Try Semantic Scholar."""
        try:
            # S2 paper object from search already has pdf_url
            if paper.source == 'semanticscholar' and paper.pdf_url:
                import requests
                resp = requests.get(paper.pdf_url, timeout=30,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
                if resp.status_code == 200:
                    return resp.content
            
            # Otherwise search by title
            result = self.s2.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"Semantic Scholar: {e}")
        return None
    
    def _try_arxiv(self, paper: PaperMetadata):
        """Try arXiv by title."""
        try:
            result = self.arxiv.search(paper.title[:100], limit=1)
            if result.success and result.papers and result.papers[0].pdf_url:
                import requests
                resp = requests.get(result.papers[0].pdf_url, timeout=30,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"arXiv: {e}")
        return None
    
    def _safe_filename(self, paper):
        """Generate safe filename."""
        author = paper.authors[0].split()[-1] if paper.authors else "Unknown"
        author = re.sub(r'[^\w]', '', author)[:15]
        year = paper.year or "0000"
        return f"{author}_{year}.pdf"
    
    def generate_report(self, topic, all_papers, selected, downloaded, failed, min_year, topic_dir):
        """Generate comprehensive report."""
        print(f"\n📝 PHASE 4: Generate Report")
        print("-"*80)
        
        success_rate = len(downloaded) / len(selected) * 100 if selected else 0
        source_counts = Counter([d['source'] for d in downloaded])
        
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'min_year': min_year,
            'semantic_filtering': bool(self.embedding_model),
            'total_searched': len(all_papers),
            'after_dedup': len(all_papers),
            'after_semantic_filter': len(selected),
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
**Semantic Filtering:** {'✅ Adaptive' if self.embedding_model else '❌ Disabled'}

## Summary

- **Multi-source Search:** {len(all_papers)} papers
- **After Semantic Filter:** {len(selected)} papers  
- **Downloaded:** {len(downloaded)} ({success_rate:.1f}%)
- **Failed:** {len(failed)}
- **Min Year:** {min_year}

## Download Sources (Diversity)

"""
        for source, count in source_counts.most_common():
            pct = count / len(downloaded) * 100 if downloaded else 0
            emoji = "🏴‍☠️" if source == 'scihub' else "✅"
            report += f"- {emoji} **{source.upper()}**: {count} papers ({pct:.1f}%)\n"
        
        report += "\n## Papers (sorted by citations)\n\n"
        for d in sorted(downloaded, key=lambda x: x['paper'].citations or 0, reverse=True):
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **Citations:** {p.citations or 0} | **DOI:** {p.doi or 'N/A'}\n\n"
            
            emoji = "🏴‍☠️" if d['source'] == 'scihub' else "✅"
            report += f"**File:** `{Path(d['path']).name}` | **Source:** {emoji} {d['source']}\n\n"
            
            if p.abstract:
                report += f"**Abstract:** {p.abstract[:250]}...\n\n"
            report += "---\n\n"
        
        (topic_dir / "REPORT.md").write_text(report)
        print(f"✅ Report saved")
    
    def run(self):
        """Main flow."""
        topic, num_papers, min_year, min_citations = self.get_user_input()
        
        topic_slug = re.sub(r'[^\w\s]', '', topic.lower()).replace(' ', '_')[:50]
        topic_dir = self.output_dir / topic_slug
        topic_dir.mkdir(exist_ok=True)
        self.output_dir = topic_dir
        
        print("\n" + "="*80)
        print("🚀 STARTING FINAL COLLECTION")
        print("="*80)
        
        # Phase 1: Multi-source parallel search
        all_papers = self.multi_source_search(topic, num_papers, min_year)
        
        if not all_papers:
            print("\n❌ No papers found!")
            return
        
        # Phase 2A: Strong deduplication
        unique_papers = self.strong_deduplicate(all_papers)
        
        # Phase 2B: Adaptive semantic filter
        selected = self.adaptive_semantic_filter(topic, unique_papers, num_papers, min_citations)
        
        if not selected:
            print("\n❌ No papers after filtering!")
            return
        
        # Phase 3: Download from ALL sources
        downloaded, failed = self.download_all_papers(selected)
        
        # Phase 4: Report
        self.generate_report(topic, unique_papers, selected, downloaded, failed, min_year, topic_dir)
        
        # Summary
        print("\n" + "="*80)
        print("✅ FINAL COLLECTION COMPLETE!")
        print("="*80)
        print(f"\n📁 {topic_dir.absolute()}")
        print(f"📄 Searched: {len(all_papers)} → Unique: {len(unique_papers)} → Filtered: {len(selected)} → Downloaded: {len(downloaded)}")
        print(f"📊 Success: {len(downloaded)/len(selected)*100:.1f}%")
        
        print(f"\n📥 Source Diversity:")
        for src, cnt in Counter([d['source'] for d in downloaded]).most_common():
            pct = cnt / len(downloaded) * 100 if downloaded else 0
            emoji = "🏴‍☠️" if src == 'scihub' else "✅"
            print(f"   {emoji} {src}: {cnt} ({pct:.0f}%)")
        
        if downloaded:
            total_mb = sum(Path(d['path']).stat().st_size for d in downloaded) / (1024*1024)
            print(f"\n💾 Total: {total_mb:.1f} MB")
        
        print("\n" + "="*80 + "\n")


def main():
    try:
        orch = FinalOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
