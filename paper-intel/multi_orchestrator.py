#!/usr/bin/env python3
"""
MULTI-SOURCE SEMANTIC ORCHESTRATOR
- Searches ALL sources simultaneously: CrossRef, OpenAlex, CORE, Semantic Scholar, arXiv
- Uses embeddings to filter irrelevant papers
- Downloads from all available sources
- Fast and comprehensive
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
    OpenAlexClient, COREClient, SemanticScholarClient,
    SciHubClient, deduplicate_papers
)
from ingestion.models import PaperMetadata

# Semantic similarity
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  pip install sentence-transformers")

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class MultiSourceOrchestrator:
    """
    Multi-source orchestrator with semantic filtering.
    Searches 5+ sources simultaneously, filters with embeddings.
    """
    
    def __init__(self, output_dir: str = "research_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load API keys from .env
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        email = os.getenv('OPENALEX_EMAIL', 'research@sanshodhak.org')
        s2_key = os.getenv('SEMANTICSCHOLAR_API_KEY')
        core_key = os.getenv('CORE_API_KEY', 'fU7PRKOIE5Nz3roXDSvCwVZ46Wd2TpYi')
        
        # Initialize ALL search clients
        self.search_clients = {
            'crossref': CrossRefClient(email=email),
            'openalex': OpenAlexClient(email=email),
            'core': COREClient(api_key=core_key),
            'semanticscholar': SemanticScholarClient(api_key=s2_key),
            'arxiv': ArxivClient()
        }
        
        # Download clients
        self.unpaywall = UnpaywallClient(email=email)
        self.scihub = SciHubClient()
        
        # Embedding model
        self.embedding_model = None
        self.similarity_threshold = 0.50  # Balanced threshold
        
        if HAS_EMBEDDINGS:
            print("🧠 Loading semantic model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic filtering enabled")
            except Exception as e:
                print(f"⚠️  Model load failed: {e}")
                self.embedding_model = None
        
        print(f"✅ {len(self.search_clients)} search sources initialized")
    
    def get_user_input(self):
        """Get user input."""
        print("\n" + "="*80)
        print("🔬 MULTI-SOURCE SEMANTIC RESEARCH COLLECTOR")
        print("="*80)
        print("\n🌐 Sources: CrossRef, OpenAlex, CORE, Semantic Scholar, arXiv")
        print("🧠 Semantic filtering removes irrelevant papers")
        print("⚡ Parallel search & download")
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
    
    def multi_search(self, topic: str, count_per_source: int, min_year: int):
        """
        PHASE 1: Search ALL sources simultaneously.
        """
        print(f"\n🔍 PHASE 1: Multi-Source Parallel Search")
        print("-"*80)
        print(f"   Topic: {topic}")
        print(f"   Searching {len(self.search_clients)} sources simultaneously...")
        
        all_papers = []
        source_results = {}
        
        # Search all sources in parallel
        with ThreadPoolExecutor(max_workers=len(self.search_clients)) as executor:
            future_to_source = {}
            
            for source_name, client in self.search_clients.items():
                future = executor.submit(self._search_single_source, source_name, client, topic, count_per_source, min_year)
                future_to_source[future] = source_name
            
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    papers = future.result(timeout=60)
                    if papers:
                        all_papers.extend(papers)
                        source_results[source_name] = len(papers)
                        print(f"   ✅ {source_name}: {len(papers)} papers")
                    else:
                        print(f"   ⚠️  {source_name}: 0 papers")
                except Exception as e:
                    print(f"   ❌ {source_name}: {e}")
                    source_results[source_name] = 0
        
        print(f"\n📊 Raw Results:")
        for source, count in source_results.items():
            print(f"   • {source}: {count}")
        
        # Deduplicate by DOI/title
        print(f"\n🔄 Deduplicating...")
        unique_papers = deduplicate_papers(all_papers)
        print(f"   {len(all_papers)} → {len(unique_papers)} unique papers")
        
        return unique_papers
    
    def _search_single_source(self, source_name: str, client, topic: str, count: int, min_year: int):
        """Search a single source."""
        try:
            result = client.search(topic, limit=count, min_year=min_year)
            if result.success:
                return result.papers
        except Exception as e:
            logger.error(f"{source_name} error: {e}")
        return []
    
    def semantic_filter(self, query: str, papers: list, target_count: int):
        """
        PHASE 2: Semantic filtering using embeddings.
        """
        if not self.embedding_model or not papers:
            print(f"\n⚠️  Semantic filtering disabled, using top {target_count}")
            return papers[:target_count]
        
        print(f"\n🧠 PHASE 2: Semantic Filtering")
        print("-"*80)
        print(f"   Analyzing {len(papers)} papers...")
        
        # Get query embedding
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return papers[:target_count]
        
        # Score all papers
        scored = []
        for paper in papers:
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract[:500]
            
            paper_emb = self._get_embedding(text)
            if paper_emb is not None:
                # Cosine similarity
                sim = np.dot(query_emb, paper_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(paper_emb))
                scored.append((paper, float(sim)))
            else:
                scored.append((paper, 0.5))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        filtered = [(p, s) for p, s in scored if s >= self.similarity_threshold]
        
        if len(filtered) < target_count:
            print(f"   ⚠️  Only {len(filtered)} above threshold {self.similarity_threshold}")
            filtered = scored[:target_count]
        else:
            filtered = filtered[:target_count]
        
        print(f"   ✅ Selected {len(filtered)} most relevant papers")
        print(f"   Similarity: {filtered[-1][1]:.3f} - {filtered[0][1]:.3f}")
        print(f"   Removed: {len(papers) - len(filtered)} irrelevant papers")
        
        return [p for p, _ in filtered]
    
    def _get_embedding(self, text: str):
        """Get embedding for text."""
        try:
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
            text = ' '.join(text.split())[:1000]  # Truncate
            return self.embedding_model.encode(text, show_progress_bar=False)
        except:
            return None
    
    def multi_download(self, papers: list):
        """
        PHASE 3: Download PDFs from multiple sources.
        """
        print(f"\n📥 PHASE 3: Multi-Source PDF Download")
        print("-"*80)
        print(f"   Sources: Unpaywall → arXiv/CORE/OpenAlex → Sci-Hub")
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
                    logger.error(f"Download error: {e}")
                    failed.append(paper)
        
        return downloaded, failed
    
    def _download_single(self, paper: PaperMetadata, idx: int, total: int):
        """Download single paper from multiple sources."""
        title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title
        print(f"\n[{idx}/{total}] {title}")
        
        filename = f"{idx:03d}_{self._safe_filename(paper)}"
        filepath = self.output_dir / filename
        
        if filepath.exists() and filepath.stat().st_size > 10000:
            print(f"   ✓ Cached")
            return {'paper': paper, 'path': str(filepath), 'source': 'cache'}
        
        # Try multiple sources
        sources = [
            ('unpaywall', lambda: self._try_unpaywall(paper)),
            ('arxiv', lambda: self._try_arxiv(paper)),
            ('core', lambda: self._try_core(paper)),
            ('openalex', lambda: self._try_openalex(paper)),
            ('scihub', lambda: self._try_scihub(paper))
        ]
        
        for source_name, fetch_func in sources:
            try:
                result = fetch_func()
                if result:
                    pdf_content = result
                    if len(pdf_content) > 10000 and pdf_content[:4] == b'%PDF':
                        filepath.write_bytes(pdf_content)
                        size_mb = len(pdf_content) / (1024 * 1024)
                        print(f"   ✅ {source_name.upper()} ({size_mb:.2f} MB)")
                        return {'paper': paper, 'path': str(filepath), 'source': source_name}
            except Exception as e:
                logger.debug(f"{source_name} failed: {e}")
        
        print(f"   ❌ All sources failed")
        return None
    
    def _try_unpaywall(self, paper):
        if not paper.doi:
            return None
        info = self.unpaywall.get_pdf_url(paper.doi)
        if info and info.get('pdf_url'):
            import requests
            resp = requests.get(info['pdf_url'], timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                return resp.content
        return None
    
    def _try_arxiv(self, paper):
        result = self.search_clients['arxiv'].search(paper.title[:100], limit=1)
        if result.success and result.papers and result.papers[0].pdf_url:
            import requests
            resp = requests.get(result.papers[0].pdf_url, timeout=30)
            if resp.status_code == 200:
                return resp.content
        return None
    
    def _try_core(self, paper):
        result = self.search_clients['core'].search(paper.title[:100], limit=1)
        if result.success and result.papers and result.papers[0].pdf_url:
            import requests
            resp = requests.get(result.papers[0].pdf_url, timeout=30)
            if resp.status_code == 200:
                return resp.content
        return None
    
    def _try_openalex(self, paper):
        query = f"doi:{paper.doi}" if paper.doi else paper.title[:100]
        result = self.search_clients['openalex'].search(query, limit=1)
        if result.success and result.papers and result.papers[0].pdf_url:
            import requests
            resp = requests.get(result.papers[0].pdf_url, timeout=30)
            if resp.status_code == 200:
                return resp.content
        return None
    
    def _try_scihub(self, paper):
        if not paper.doi:
            return None
        result = self.scihub.get_pdf(paper.doi)
        if result:
            pdf_content, _ = result
            return pdf_content
        return None
    
    def _safe_filename(self, paper):
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
        
        # Metadata
        metadata = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'min_year': min_year,
            'semantic_filtering': bool(self.embedding_model),
            'sources_searched': list(self.search_clients.keys()),
            'target_papers': len(papers),
            'downloaded': len(downloaded),
            'failed': len(failed),
            'success_rate': f"{success_rate:.1f}%",
            'download_sources': dict(source_counts),
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
        
        # Markdown report
        report = f"""# Multi-Source Research Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Semantic Filtering:** {'✅ Enabled' if self.embedding_model else '❌ Disabled'}  
**Sources Searched:** {', '.join(self.search_clients.keys())}

## Summary

- **Target Papers:** {len(papers)}
- **Downloaded:** {len(downloaded)} ({success_rate:.1f}%)
- **Failed:** {len(failed)}
- **Min Year:** {min_year}

## Download Sources

"""
        for source, count in source_counts.most_common():
            pct = count / len(downloaded) * 100 if downloaded else 0
            report += f"- **{source}**: {count} ({pct:.1f}%)\n"
        
        report += "\n## Papers\n\n"
        for d in sorted(downloaded, key=lambda x: x['paper'].citations or 0, reverse=True):
            p = d['paper']
            report += f"### {p.title}\n\n"
            report += f"**Authors:** {', '.join(p.authors[:3])}\n\n"
            report += f"**Year:** {p.year} | **Cites:** {p.citations or 0} | **DOI:** {p.doi or 'N/A'}\n\n"
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
        print("🚀 STARTING MULTI-SOURCE COLLECTION")
        print("="*80)
        
        # Phase 1: Multi-source search
        all_papers = self.multi_search(topic, num_papers, min_year)
        
        if not all_papers:
            print("\n❌ No papers found!")
            return
        
        # Phase 2: Semantic filtering
        selected = self.semantic_filter(topic, all_papers, num_papers)
        
        # Phase 3: Download
        downloaded, failed = self.multi_download(selected)
        
        # Phase 4: Report
        self.generate_report(topic, selected, downloaded, failed, min_year, topic_dir)
        
        # Summary
        print("\n" + "="*80)
        print("✅ COLLECTION COMPLETE!")
        print("="*80)
        print(f"\n📁 {topic_dir.absolute()}")
        print(f"📄 Found: {len(all_papers)} → Selected: {len(selected)} → Downloaded: {len(downloaded)}")
        print(f"📊 Success: {len(downloaded)/len(selected)*100:.1f}%")
        print(f"\n📥 Sources:")
        for src, cnt in Counter([d['source'] for d in downloaded]).most_common():
            print(f"   • {src}: {cnt}")
        if downloaded:
            total_mb = sum(Path(d['path']).stat().st_size for d in downloaded) / (1024*1024)
            print(f"\n💾 Total: {total_mb:.1f} MB")
        print("\n" + "="*80 + "\n")


def main():
    try:
        orch = MultiSourceOrchestrator()
        orch.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
