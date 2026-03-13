#!/usr/bin/env python3
"""
ULTIMATE MULTI-SOURCE ORCHESTRATOR - Maximum Coverage + Quality
Strategy:
1. CrossRef search 3X papers → Get DOIs
2. Semantic filter → Keep best papers
3. For EACH paper with DOI:
   - Unpaywall (legal OA)
   - Sci-Hub (shadow library) 
   - Anna's Archive (shadow library)
   - OpenAlex (250M works)
   - CORE (200M OA)
   - arXiv (preprints)
4. DOAJ search for open access journals
5. Ensure high-quality, well-cited papers
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

# Try DOAJ import
try:
    from ingestion.discovery.doaj_client import DOAJClient
    HAS_DOAJ = True
except ImportError:
    HAS_DOAJ = False
    DOAJClient = None
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


class UltimateOrchestrator:
    """
    ULTIMATE orchestrator with:
    - 3X CrossRef search
    - ALL shadow libraries (Sci-Hub, Anna's)
    - Open access journals (DOAJ)
    - Quality filtering (min citations)
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
        
        print("🔧 Initializing ALL clients...")
        
        # Initialize ALL clients
        self.crossref = CrossRefClient(email=email)
        self.unpaywall = UnpaywallClient(email=email)
        self.openalex = OpenAlexClient(email=email)
        self.core = COREClient(api_key=core_key)
        self.arxiv = ArxivClient()
        self.scihub = SciHubClient()
        self.annas = AnnasArchiveClient()
        
        # Try DOAJ
        self.doaj = None
        if HAS_DOAJ:
            try:
                self.doaj = DOAJClient()
                print("   ✅ DOAJ (open access journals)")
            except Exception as e:
                print(f"   ⚠️  DOAJ unavailable: {e}")
        else:
            print("   ⚠️  DOAJ not available")
        
        # Embedding model
        self.embedding_model = None
        self.similarity_threshold = 0.50
        
        if HAS_EMBEDDINGS:
            print("🧠 Loading semantic model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic filtering ENABLED")
            except:
                self.embedding_model = None
        
        print("\n✅ ALL CLIENTS READY:")
        print("   📚 Search: CrossRef, OpenAlex, arXiv, CORE, DOAJ")
        print("   📥 Legal: Unpaywall, OpenAlex, CORE, arXiv, DOAJ")
        print("   🏴‍☠️ Shadow: Sci-Hub, Anna's Archive")
    
    def get_user_input(self):
        """Get user input."""
        print("\n" + "="*80)
        print("🚀 ULTIMATE RESEARCH ORCHESTRATOR - MAX COVERAGE")
        print("="*80)
        print("\n📋 Strategy:")
        print("   1. CrossRef → 3X papers (get many DOIs)")
        print("   2. Semantic filter → Best papers only")
        print("   3. DOAJ → Open access journals")
        print("   4. For EACH paper, try ALL sources:")
        print("      ✅ Unpaywall (legal OA)")
        print("      🏴‍☠️ Sci-Hub (shadow library)")
        print("      🏴‍☠️ Anna's Archive (shadow library)")
        print("      ✅ OpenAlex (250M+ works)")
        print("      ✅ CORE (200M+ OA papers)")
        print("      ✅ arXiv (preprints)")
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
    
    def search_crossref_3x(self, topic: str, count: int, min_year: int):
        """
        PHASE 1A: CrossRef 3X search.
        Get 3x papers to ensure we have enough after filtering.
        """
        print(f"\n🔍 PHASE 1A: CrossRef 3X Search")
        print("-"*80)
        print(f"   Topic: {topic}")
        print(f"   Searching: {count * 3} papers → Will filter to {count}")
        print(f"   Min year: {min_year}")
        
        result = self.crossref.search(topic, limit=count * 3, min_year=min_year)
        
        if not result.success:
            logger.error(f"CrossRef failed: {result.error}")
            return []
        
        papers = result.papers
        
        print(f"\n✅ CrossRef: {len(papers)} papers")
        print(f"   With DOI: {sum(1 for p in papers if p.doi)}")
        print(f"   With abstract: {sum(1 for p in papers if p.abstract)}")
        print(f"   With citations: {sum(1 for p in papers if p.citations and p.citations > 0)}")
        
        if papers:
            years = [p.year for p in papers if p.year]
            cites = [p.citations for p in papers if p.citations and p.citations > 0]
            if years:
                print(f"   Year range: {min(years)}-{max(years)}")
            if cites:
                print(f"   Citations: {min(cites)}-{max(cites)} (avg: {sum(cites)//len(cites)})")
        
        return papers
    
    def search_doaj(self, topic: str, count: int):
        """
        PHASE 1B: DOAJ for open access journals.
        """
        if not self.doaj:
            return []
        
        print(f"\n🔍 PHASE 1B: DOAJ (Open Access Journals)")
        print("-"*80)
        
        try:
            result = self.doaj.search(topic, limit=count)
            if result.success and result.papers:
                print(f"✅ DOAJ: {len(result.papers)} OA journal papers")
                return result.papers
        except Exception as e:
            logger.debug(f"DOAJ failed: {e}")
        
        print("⚠️  DOAJ: No results")
        return []
    
    def deduplicate(self, papers: list):
        """Remove duplicates by DOI/title."""
        seen_doi = set()
        seen_title = set()
        unique = []
        
        for paper in papers:
            if paper.doi:
                if paper.doi.lower() not in seen_doi:
                    seen_doi.add(paper.doi.lower())
                    unique.append(paper)
            else:
                title_clean = re.sub(r'[^\w]', '', paper.title.lower())[:100]
                if title_clean not in seen_title:
                    seen_title.add(title_clean)
                    unique.append(paper)
        
        return unique
    
    def semantic_filter(self, query: str, papers: list, target: int, min_citations: int = 0):
        """
        PHASE 2: Semantic filtering + quality filter.
        """
        print(f"\n🧠 PHASE 2: Semantic + Quality Filtering")
        print("-"*80)
        print(f"   Input: {len(papers)} papers")
        print(f"   Target: {target} papers")
        print(f"   Min citations: {min_citations}")
        
        # Quality filter
        if min_citations > 0:
            papers = [p for p in papers if p.citations and p.citations >= min_citations]
            print(f"   After citation filter: {len(papers)}")
        
        if not self.embedding_model or not papers:
            print(f"   ⚠️  No semantic filtering, using top {target}")
            # Sort by citations
            papers.sort(key=lambda p: p.citations or 0, reverse=True)
            return papers[:target]
        
        print(f"   🧠 Computing semantic similarity...")
        
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
        
        # Sort by similarity, then by citations
        scored.sort(key=lambda x: (x[1], x[0].citations or 0), reverse=True)
        
        filtered = [(p, s) for p, s in scored if s >= self.similarity_threshold]
        if len(filtered) < target:
            filtered = scored[:target]
        else:
            filtered = filtered[:target]
        
        print(f"   ✅ Selected: {len(filtered)} papers")
        print(f"   Similarity: {filtered[-1][1]:.3f} - {filtered[0][1]:.3f}")
        if filtered[0][0].citations:
            print(f"   Citations: {min(p.citations or 0 for p, _ in filtered)} - {max(p.citations or 0 for p, _ in filtered)}")
        print(f"   Filtered out: {len(papers) - len(filtered)} papers")
        
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
        PHASE 3: Download using ALL sources INCLUDING SHADOW LIBS.
        """
        print(f"\n📥 PHASE 3: Multi-Source Download (ALL SOURCES)")
        print("-"*80)
        print("   Priority: Unpaywall → Sci-Hub → Anna's → OpenAlex → CORE → arXiv")
        print("   🏴‍☠️ SHADOW LIBRARIES ENABLED")
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
                    result = future.result(timeout=180)  # 3 min timeout
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
        Priority: Unpaywall → Sci-Hub → Anna's → OpenAlex → CORE → arXiv
        """
        title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
        print(f"\n[{idx}/{total}] {title}")
        
        filename = f"{idx:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Try ALL sources in priority order
        
        # 1. Unpaywall (legal OA) - HIGHEST PRIORITY
        if paper.doi:
            result = self._try_unpaywall(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'unpaywall')
        
        # 2. Sci-Hub (shadow library) - SECOND PRIORITY
        if paper.doi:
            result = self._try_scihub(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'scihub')
        
        # 3. Anna's Archive (shadow library)
        if paper.doi:
            result = self._try_annas_archive(paper.doi)
            if result:
                return self._save_and_return(result, paper, filepath, 'annas_archive')
        
        # 4. OpenAlex (direct PDF from metadata)
        result = self._try_openalex_direct(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'openalex')
        
        # 5. CORE (search by title/DOI)
        result = self._try_core_search(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'core')
        
        # 6. arXiv (search by title)
        result = self._try_arxiv_search(paper)
        if result:
            return self._save_and_return(result, paper, filepath, 'arxiv')
        
        print(f"   ❌ All sources failed")
        return None
    
    def _save_and_return(self, pdf_content, paper, filepath, source):
        """Save PDF and return result."""
        if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
            filepath.write_bytes(pdf_content)
            size_mb = len(pdf_content) / (1024 * 1024)
            
            # Emoji for source type
            emoji = "✅" if source in ['unpaywall', 'openalex', 'core', 'arxiv'] else "🏴‍☠️"
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
    
    def _try_annas_archive(self, doi: str):
        """Try Anna's Archive by DOI."""
        try:
            result = self.annas.search_by_doi(doi)
            if result and result.get('pdf_url'):
                import requests
                resp = requests.get(result['pdf_url'], timeout=30, allow_redirects=True,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"Anna's: {e}")
        return None
    
    def _try_openalex_direct(self, paper: PaperMetadata):
        """Try OpenAlex with DOI or title."""
        try:
            if paper.doi:
                doi_clean = paper.doi.replace('https://doi.org/', '')
                params = {
                    'filter': f'doi:{doi_clean}',
                    'per-page': 1
                }
                response = self.openalex.get('/works', params=params)
            else:
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
                resp = requests.get(result.papers[0].pdf_url, timeout=30,
                                  headers={'User-Agent': 'Sanshodhak/1.0'})
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
        
        # Count shadow libs
        shadow_count = source_counts.get('scihub', 0) + source_counts.get('annas_archive', 0)
        legal_count = len(downloaded) - shadow_count
        
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'min_year': min_year,
            'semantic_filtering': bool(self.embedding_model),
            'crossref_papers': len(all_papers),
            'after_semantic_filter': len(selected),
            'downloaded': len(downloaded),
            'failed': len(failed),
            'success_rate': f"{success_rate:.1f}%",
            'sources_used': dict(source_counts),
            'legal_sources': legal_count,
            'shadow_sources': shadow_count,
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

- **CrossRef Search (3X):** {len(all_papers)} papers
- **After Semantic Filter:** {len(selected)} papers  
- **Downloaded:** {len(downloaded)} ({success_rate:.1f}%)
- **Failed:** {len(failed)}
- **Min Year:** {min_year}

## Download Sources

**Legal Sources ({legal_count}):**
"""
        legal_sources = ['unpaywall', 'openalex', 'core', 'arxiv', 'doaj', 'cache']
        for source in legal_sources:
            count = source_counts.get(source, 0)
            if count > 0:
                pct = count / len(downloaded) * 100 if downloaded else 0
                report += f"- ✅ **{source.upper()}**: {count} papers ({pct:.1f}%)\n"
        
        report += f"\n**Shadow Libraries ({shadow_count}):**\n"
        shadow_sources = ['scihub', 'annas_archive']
        for source in shadow_sources:
            count = source_counts.get(source, 0)
            if count > 0:
                pct = count / len(downloaded) * 100 if downloaded else 0
                report += f"- 🏴‍☠️ **{source.upper()}**: {count} papers ({pct:.1f}%)\n"
        
        report += "\n## Papers (sorted by citations)\n\n"
        for d in sorted(downloaded, key=lambda x: x['paper'].citations or 0, reverse=True):
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **Citations:** {p.citations or 0} | **DOI:** {p.doi or 'N/A'}\n\n"
            
            # Emoji for source
            emoji = "✅" if d['source'] in legal_sources else "🏴‍☠️"
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
        print("🚀 STARTING ULTIMATE COLLECTION")
        print("="*80)
        
        # Phase 1A: CrossRef 3X search
        crossref_papers = self.search_crossref_3x(topic, num_papers, min_year)
        
        # Phase 1B: DOAJ for OA journals
        doaj_papers = self.search_doaj(topic, num_papers // 2)
        
        # Combine and deduplicate
        all_papers = crossref_papers + doaj_papers
        all_papers = self.deduplicate(all_papers)
        
        print(f"\n📊 Combined: {len(all_papers)} unique papers")
        
        if not all_papers:
            print("\n❌ No papers found!")
            return
        
        # Phase 2: Semantic filter
        selected = self.semantic_filter(topic, all_papers, num_papers, min_citations)
        
        # Phase 3: Download from ALL sources
        downloaded, failed = self.download_all_papers(selected)
        
        # Phase 4: Report
        self.generate_report(topic, all_papers, selected, downloaded, failed, min_year, topic_dir)
        
        # Summary
        print("\n" + "="*80)
        print("✅ ULTIMATE COLLECTION COMPLETE!")
        print("="*80)
        print(f"\n📁 {topic_dir.absolute()}")
        print(f"📄 CrossRef 3X: {len(crossref_papers)} → Combined: {len(all_papers)} → Filtered: {len(selected)} → Downloaded: {len(downloaded)}")
        print(f"📊 Success: {len(downloaded)/len(selected)*100:.1f}%")
        
        print(f"\n📥 Source Diversity:")
        source_counts = Counter([d['source'] for d in downloaded])
        
        print("\n   ✅ LEGAL SOURCES:")
        for src in ['unpaywall', 'openalex', 'core', 'arxiv', 'doaj', 'cache']:
            cnt = source_counts.get(src, 0)
            if cnt > 0:
                pct = cnt / len(downloaded) * 100 if downloaded else 0
                print(f"      • {src}: {cnt} ({pct:.0f}%)")
        
        shadow_count = source_counts.get('scihub', 0) + source_counts.get('annas_archive', 0)
        if shadow_count > 0:
            print("\n   🏴‍☠️ SHADOW LIBRARIES:")
            for src in ['scihub', 'annas_archive']:
                cnt = source_counts.get(src, 0)
                if cnt > 0:
                    pct = cnt / len(downloaded) * 100 if downloaded else 0
                    print(f"      • {src}: {cnt} ({pct:.0f}%)")
        
        if downloaded:
            total_mb = sum(Path(d['path']).stat().st_size for d in downloaded) / (1024*1024)
            print(f"\n💾 Total: {total_mb:.1f} MB")
        
        print("\n" + "="*80 + "\n")


def main():
    try:
        orch = UltimateOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
