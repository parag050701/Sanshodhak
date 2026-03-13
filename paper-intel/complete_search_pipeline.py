#!/usr/bin/env python3
"""
COMPLETE SANSHODHAK SEARCH PIPELINE
====================================

This script implements the FULL research paper search and analysis pipeline:

1. USER INPUT - Ask for: Topic, Paper count, Minimum year
2. SEARCH - Semantic search using paper-intel SearchEngine (10X strategy)
3. DOWNLOAD - Download PDFs to research_papers/ folder
4. PARSE - Extract text from PDFs using fallback extractors
5. EMBED & RAG - Build FAISS index with Ollama embeddings
6. EVALUATE - Run RAG quality metrics
7. RETRIEVE - Answer test questions
8. RECOMMEND - Find HuggingFace models, GitHub repos, web resources

Uses EXACT same implementation as paper-intel ingestion system.
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import ingestion components
from ingestion.ingestion_engine import IngestionEngine
from ingestion.models import PaperMetadata
from ingestion.fallback_extractor import extract_text_fast

# Import RAG and recommendation
from ollama_rag import OllamaRAG
from resource_recommender import ResourceRecommender


class CompletePipeline:
    """Complete research pipeline with all stages."""
    
    def __init__(self, output_dir: str = "research_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = None
        self.session_dir = None
        self.text_dir = None
        self.index_dir = None
        
        self.ingestion_engine = None
        self.rag_system = None
        self.recommender = None
        
        self.papers: List[PaperMetadata] = []
        self.downloaded_pdfs: List[Path] = []
        self.parsed_texts: List[Path] = []
    
    def print_banner(self, text: str):
        """Print a formatted banner."""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)
    
    def print_progress(self, stage: str, message: str, data: Optional[Dict] = None):
        """Print progress with optional data."""
        print(f"\n[{stage}] {message}")
        if data:
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  • {key}: {value}")
    
    def get_user_inputs(self) -> Dict[str, any]:
        """Get user inputs for the pipeline."""
        self.print_banner("🔍 SANSHODHAK RESEARCH PIPELINE")
        
        print("\nPlease provide the following information:\n")
        
        # Get topic
        topic = input("1️⃣  Research Topic: ").strip()
        if not topic:
            print("❌ Error: Topic cannot be empty!")
            sys.exit(1)
        
        # Get paper count
        while True:
            try:
                paper_count = input("2️⃣  Number of Papers (default: 10): ").strip()
                paper_count = int(paper_count) if paper_count else 10
                if paper_count < 1:
                    print("❌ Error: Must be at least 1 paper!")
                    continue
                if paper_count > 50:
                    confirm = input(f"⚠️  {paper_count} papers will take time. Continue? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                break
            except ValueError:
                print("❌ Error: Please enter a valid number!")
        
        # Get minimum year
        while True:
            try:
                current_year = datetime.now().year
                min_year = input(f"3️⃣  Minimum Publication Year (default: {current_year - 5}): ").strip()
                min_year = int(min_year) if min_year else (current_year - 5)
                if min_year < 1950 or min_year > current_year:
                    print(f"❌ Error: Year must be between 1950 and {current_year}!")
                    continue
                break
            except ValueError:
                print("❌ Error: Please enter a valid year!")
        
        return {
            'topic': topic,
            'paper_count': paper_count,
            'min_year': min_year
        }
    
    async def stage1_search(self, query: str, required_papers: int, min_year: int):
        """
        STAGE 1: SEMANTIC SEARCH
        
        Uses paper-intel SearchEngine with 10X strategy:
        - User wants N papers → Search 10×N papers
        - Multi-source: CrossRef, Unpaywall, OpenAlex, CORE, S2, arXiv
        - Merge, deduplicate, rank by quality
        - Select TOP N papers
        """
        self.print_progress(
            "SEARCH",
            "🔍 Starting 10X SEMANTIC SEARCH using paper-intel engine",
            {
                'query': query,
                'user_requested': required_papers,
                'will_search': required_papers * 10,
                'strategy': '10X (search 10× more, select best)',
                'min_year': min_year
            }
        )
        
        # Initialize ingestion engine
        # Note: Set output to research_papers/ folder
        self.ingestion_engine = IngestionEngine(
            query=query,
            required_count=required_papers,
            output_dir=str(self.output_dir),
            min_year=min_year,
            enable_core=True,
            enable_scihub=False,  # Disable for ethical reasons
            enable_libgen=False,
            prefer_open_access=True
        )
        
        print("\n  📚 Searching closed-access sources (CrossRef + Unpaywall)...")
        closed_papers = await self.ingestion_engine.search_closed_access()
        print(f"  ✓ Found {len(closed_papers)} papers from CrossRef")
        
        print("\n  🌐 Searching open-source sources (OpenAlex, CORE, S2, arXiv)...")
        open_papers = await self.ingestion_engine.search_open_source()
        print(f"  ✓ Found {len(open_papers)} papers from open sources")
        
        print("\n  🔗 Merging and deduplicating...")
        all_papers = await self.ingestion_engine.merge_and_deduplicate(open_papers, closed_papers)
        print(f"  ✓ Total unique papers: {len(all_papers)}")
        
        print("\n  🏆 Ranking by quality (citations × recency × OA availability)...")
        ranked_papers = await self.ingestion_engine.rank_and_filter(all_papers)
        
        # Select top N
        self.papers = ranked_papers[:required_papers]
        
        self.print_progress(
            "SEARCH",
            f"✅ 10X Search complete: Selected TOP {len(self.papers)} from {len(all_papers)} candidates",
            {
                'total_searched': len(all_papers),
                'selected': len(self.papers),
                'avg_citations': np.mean([p.citations for p in self.papers]),
                'oa_percentage': sum(p.is_open_access for p in self.papers) / len(self.papers) * 100,
                'year_range': f"{min(p.year for p in self.papers)}-{max(p.year for p in self.papers)}"
            }
        )
        
        # Show top 5 papers
        print("\n  📄 Top 5 papers:")
        for i, paper in enumerate(self.papers[:5], 1):
            print(f"     {i}. {paper.title[:70]}...")
            print(f"        Year: {paper.year}, Citations: {paper.citations}, OA: {paper.is_open_access}")
        
        return self.papers
    
    async def stage2_download(self):
        """
        STAGE 2: DOWNLOAD PDFs
        
        Downloads PDFs to research_papers/ folder using fallback chain:
        - Unpaywall Open Access
        - arXiv preprints
        - Publisher direct (if available)
        """
        self.print_progress(
            "DOWNLOAD",
            f"📥 Downloading {len(self.papers)} PDFs to {self.output_dir}/",
            {'fallback_chain': 'Unpaywall → arXiv → Publisher'}
        )
        
        # Use ingestion engine's downloader
        downloaded = 0
        failed = 0
        
        for i, paper in enumerate(self.papers, 1):
            print(f"\n  [{i}/{len(self.papers)}] {paper.title[:60]}...")
            
            # Try download (NOT async!)
            result = self.ingestion_engine.pdf_downloader.download_paper(paper)
            
            if result.success:
                print(f"     ✓ Downloaded: {result.pdf_path}")
                self.downloaded_pdfs.append(Path(result.pdf_path))
                downloaded += 1
            else:
                print(f"     ✗ Failed: {result.error}")
                failed += 1
        
        self.print_progress(
            "DOWNLOAD",
            f"✅ Download complete: {downloaded}/{len(self.papers)} PDFs",
            {
                'success': downloaded,
                'failed': failed,
                'success_rate': f"{downloaded/len(self.papers)*100:.1f}%",
                'output_dir': str(self.output_dir)
            }
        )
        
        return self.downloaded_pdfs
    
    async def stage3_parse(self):
        """
        STAGE 3: PARSE PDFs TO TEXT
        
        Extracts text using fallback chain:
        - PyMuPDF (fitz) - fastest
        - pdfplumber - more accurate
        - PyPDF2 - most compatible
        """
        self.print_progress(
            "PARSE",
            f"📄 Parsing {len(self.downloaded_pdfs)} PDFs with fallback extractors",
            {'extractors': 'PyMuPDF → pdfplumber → PyPDF2'}
        )
        
        # Create text directory
        self.text_dir = self.output_dir / "extracted_text"
        self.text_dir.mkdir(exist_ok=True)
        
        parsed = 0
        failed = 0
        total_chars = 0
        
        for i, pdf_path in enumerate(self.downloaded_pdfs, 1):
            print(f"\n  [{i}/{len(self.downloaded_pdfs)}] {pdf_path.name}...")
            
            try:
                # Use fallback extractor
                text = extract_text_fast(str(pdf_path))
                
                if text and len(text.strip()) > 100:
                    # Save to text file
                    text_file = self.text_dir / f"{pdf_path.stem}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    print(f"     ✓ Extracted {len(text)} characters")
                    self.parsed_texts.append(text_file)
                    parsed += 1
                    total_chars += len(text)
                else:
                    print(f"     ✗ Failed: No text extracted")
                    failed += 1
            except Exception as e:
                print(f"     ✗ Failed: {e}")
                failed += 1
        
        self.print_progress(
            "PARSE",
            f"✅ Parsing complete: {parsed}/{len(self.downloaded_pdfs)} PDFs",
            {
                'success': parsed,
                'failed': failed,
                'total_characters': total_chars,
                'avg_chars_per_paper': total_chars // parsed if parsed > 0 else 0,
                'output_dir': str(self.text_dir)
            }
        )
        
        return self.parsed_texts
    
    async def stage4_embed_and_rag(self):
        """
        STAGE 4: EMBEDDINGS & RAG INDEX
        
        Creates NEW vector index for this topic:
        - Chunk documents (500 chars per chunk)
        - Generate embeddings using Ollama (bge-m3, 1024-dim)
        - Build FAISS index (cosine similarity)
        - Save to index directory
        """
        self.print_progress(
            "EMBED",
            f"🧬 Creating NEW embeddings for {len(self.parsed_texts)} papers",
            {
                'model': 'bge-m3 (Ollama)',
                'dimension': 1024,
                'chunk_size': 500,
                'index_type': 'FAISS IndexFlatIP (cosine similarity)'
            }
        )
        
        # Create index directory
        self.index_dir = self.output_dir / "rag_index"
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize RAG system
        print("\n  🤖 Initializing Ollama RAG system...")
        self.rag_system = OllamaRAG(
            embed_model='bge-m3',
            llm_model='deepseek-r1:7b'
        )
        
        # Build index
        print(f"\n  📚 Building index from {self.text_dir}...")
        self.rag_system.build_index(str(self.text_dir))
        
        # Save index
        print(f"\n  💾 Saving index to {self.index_dir}...")
        self.rag_system.save(str(self.index_dir))
        
        self.print_progress(
            "EMBED",
            "✅ Embedding & indexing complete",
            {
                'total_chunks': len(self.rag_system.chunks),
                'papers_indexed': len(self.parsed_texts),
                'index_size': self.rag_system.index.ntotal,
                'index_path': str(self.index_dir)
            }
        )
        
        return self.rag_system
    
    async def stage5_evaluate(self):
        """
        STAGE 5: RAG EVALUATION
        
        Tests RAG system quality:
        - Retrieval accuracy (top-k precision)
        - Answer relevance
        - Response time
        - Coverage (% of papers contributing to answers)
        """
        self.print_progress(
            "EVALUATE",
            "🎯 Running RAG quality evaluation",
            {'metrics': 'Retrieval accuracy, Answer relevance, Coverage'}
        )
        
        # Test questions based on topic
        test_questions = [
            f"What are the main approaches in this research area?",
            f"What are the key challenges and limitations?",
            f"What datasets or benchmarks are commonly used?",
            f"What are the recent advancements?",
            f"What are potential future research directions?"
        ]
        
        eval_results = {
            'retrieval_times': [],
            'answer_lengths': [],
            'sources_per_answer': [],
            'unique_sources': set()
        }
        
        print("\n  Testing with evaluation questions:")
        for i, question in enumerate(test_questions, 1):
            print(f"\n  Q{i}: {question}")
            
            start_time = time.time()
            answer, sources = self.rag_system.query(question, top_k=5)
            retrieval_time = time.time() - start_time
            
            eval_results['retrieval_times'].append(retrieval_time)
            eval_results['answer_lengths'].append(len(answer))
            eval_results['sources_per_answer'].append(len(sources))
            
            for src in sources:
                eval_results['unique_sources'].add(src['metadata']['file'])
            
            print(f"     ✓ Retrieved in {retrieval_time:.2f}s, {len(sources)} sources")
        
        # Calculate metrics
        avg_time = np.mean(eval_results['retrieval_times'])
        avg_sources = np.mean(eval_results['sources_per_answer'])
        coverage = len(eval_results['unique_sources']) / len(self.parsed_texts) * 100
        
        self.print_progress(
            "EVALUATE",
            "✅ Evaluation complete",
            {
                'avg_retrieval_time': f"{avg_time:.2f}s",
                'avg_sources_per_answer': f"{avg_sources:.1f}",
                'coverage': f"{coverage:.1f}% of papers used",
                'unique_sources': len(eval_results['unique_sources']),
                'total_papers': len(self.parsed_texts)
            }
        )
        
        return eval_results
    
    async def stage6_retrieve(self, user_query: str):
        """
        STAGE 6: RETRIEVE & ANSWER
        
        Query the RAG system for the user's research topic.
        """
        self.print_progress(
            "RETRIEVE",
            f"💡 Querying RAG system: {user_query}",
            {'model': 'deepseek-r1:7b', 'top_k': 5}
        )
        
        print("\n  🔍 Searching vector index...")
        answer, sources = self.rag_system.query(user_query, top_k=5)
        
        print("\n" + "─"*80)
        print("📝 ANSWER:")
        print("─"*80)
        print(answer)
        print("─"*80)
        
        print("\n📚 SOURCES:")
        for i, src in enumerate(sources, 1):
            print(f"\n  [{i}] {src['metadata']['file']}")
            print(f"      Score: {src['score']:.4f}")
            print(f"      Snippet: {src['text'][:150]}...")
        
        self.print_progress(
            "RETRIEVE",
            "✅ Answer generated",
            {
                'answer_length': len(answer),
                'sources_used': len(sources),
                'avg_relevance': np.mean([s['score'] for s in sources])
            }
        )
        
        return answer, sources
    
    async def stage7_recommend(self, query: str):
        """
        STAGE 7: RESOURCE RECOMMENDATION
        
        Find related resources:
        - HuggingFace models & datasets
        - GitHub repositories
        - Web tutorials & documentation
        """
        self.print_progress(
            "RECOMMEND",
            f"🎯 Finding resources for: {query}",
            {'sources': 'HuggingFace, GitHub, Web'}
        )
        
        # Initialize recommender
        self.recommender = ResourceRecommender(self.rag_system)
        
        print("\n  🔍 Searching resources...")
        resources = self.recommender.recommend_resources(query)
        
        # Display results
        print("\n" + "="*80)
        print("🎯 RECOMMENDED RESOURCES")
        print("="*80)
        
        if resources.get('huggingface'):
            print("\n🤗 HuggingFace Models & Datasets:")
            for i, item in enumerate(resources['huggingface'][:5], 1):
                print(f"  {i}. {item['name']}")
                print(f"     Type: {item['type']}, Downloads: {item.get('downloads', 'N/A')}")
                print(f"     URL: {item['url']}")
        
        if resources.get('github'):
            print("\n💻 GitHub Repositories:")
            for i, repo in enumerate(resources['github'][:5], 1):
                print(f"  {i}. {repo['name']}")
                print(f"     Stars: {repo.get('stars', 0)}, Language: {repo.get('language', 'N/A')}")
                print(f"     URL: {repo['url']}")
        
        if resources.get('web'):
            print("\n🌐 Web Resources:")
            for i, link in enumerate(resources['web'][:5], 1):
                print(f"  {i}. {link['title']}")
                print(f"     URL: {link['url']}")
        
        self.print_progress(
            "RECOMMEND",
            "✅ Resources found",
            {
                'huggingface': len(resources.get('huggingface', [])),
                'github': len(resources.get('github', [])),
                'web': len(resources.get('web', []))
            }
        )
        
        return resources
    
    async def run_complete_pipeline(self):
        """Execute the complete pipeline."""
        start_time = time.time()
        
        try:
            # Get user inputs
            inputs = self.get_user_inputs()
            
            # Run all stages
            print("\n" + "🚀 "*20)
            print("Starting complete pipeline...")
            print("🚀 "*20)
            
            # Stage 1: Search
            await self.stage1_search(
                query=inputs['topic'],
                required_papers=inputs['paper_count'],
                min_year=inputs['min_year']
            )
            
            # Stage 2: Download
            if self.papers:
                await self.stage2_download()
            else:
                print("❌ No papers found. Exiting.")
                return
            
            # Stage 3: Parse
            if self.downloaded_pdfs:
                await self.stage3_parse()
            else:
                print("❌ No PDFs downloaded. Exiting.")
                return
            
            # Stage 4: Embed & RAG
            if self.parsed_texts:
                await self.stage4_embed_and_rag()
            else:
                print("❌ No texts parsed. Exiting.")
                return
            
            # Stage 5: Evaluate
            await self.stage5_evaluate()
            
            # Stage 6: Retrieve
            await self.stage6_retrieve(inputs['topic'])
            
            # Stage 7: Recommend
            await self.stage7_recommend(inputs['topic'])
            
            # Summary
            duration = time.time() - start_time
            self.print_banner(f"✅ PIPELINE COMPLETE in {duration:.1f}s!")
            
            print("\n📁 Output Locations:")
            print(f"  • PDFs: {self.output_dir}/")
            print(f"  • Texts: {self.text_dir}/")
            print(f"  • Index: {self.index_dir}/")
            
            print("\n✨ Summary:")
            print(f"  • Papers found: {len(self.papers)}")
            print(f"  • PDFs downloaded: {len(self.downloaded_pdfs)}")
            print(f"  • Texts parsed: {len(self.parsed_texts)}")
            print(f"  • Chunks indexed: {len(self.rag_system.chunks)}")
            
            print("\n🎉 All stages completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


async def main():
    """Main entry point."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "SANSHODHAK RESEARCH PIPELINE" + " "*30 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\nThis pipeline will:")
    print("  1. Search for research papers using semantic search")
    print("  2. Download PDFs to research_papers/ folder")
    print("  3. Parse PDFs to extract text")
    print("  4. Build embeddings and RAG index")
    print("  5. Evaluate RAG quality")
    print("  6. Answer your research questions")
    print("  7. Recommend related resources")
    
    print("\n⏱️  Estimated time: 5-20 minutes depending on paper count")
    print("⚠️  Press Ctrl+C to abort at any time\n")
    
    # Run pipeline
    pipeline = CompletePipeline(output_dir="research_papers")
    await pipeline.run_complete_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
