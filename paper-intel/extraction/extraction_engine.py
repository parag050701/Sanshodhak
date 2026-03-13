"""
Extraction Engine - Main pipeline for Phase 2.

Orchestrates extraction using Graffiti (primary) and LLM fallback.
Processes Phase 1 outputs into Neo4j-ready triples.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

from .graffiti_client import GraffitiClient
from .llm_fallback import LLMFallbackExtractor
from .schema import normalize_entity, normalize_relation, validate_schema
from .triple_builder import TripleBuilder

logger = logging.getLogger(__name__)


class ExtractionEngine:
    """
    Phase 2 extraction engine.
    
    Pipeline:
    1. Load paper text + metadata from Phase 1
    2. Try Graffiti extraction
    3. If fail → LLM fallback
    4. Normalize via schema
    5. Build triples via triple_builder
    6. Save outputs (entities, relations, triples)
    """
    
    def __init__(
        self,
        json_dir: Path = Path("ingestion/output_json"),
        text_dir: Path = Path("ingestion/raw_text"),
        output_dir: Path = Path("extraction"),
        use_graffiti: bool = True,
        use_llm_fallback: bool = True,
        workers: int = 4
    ):
        """
        Initialize extraction engine.
        
        Args:
            json_dir: Phase 1 JSON output directory
            text_dir: Phase 1 text output directory
            output_dir: Phase 2 output directory
            use_graffiti: Enable Graffiti extraction
            use_llm_fallback: Enable LLM fallback
            workers: Number of parallel workers
        """
        self.json_dir = Path(json_dir)
        self.text_dir = Path(text_dir)
        self.output_dir = Path(output_dir)
        self.workers = workers
        
        # Output subdirectories
        self.entities_dir = self.output_dir / 'output_entities'
        self.relations_dir = self.output_dir / 'output_relations'
        self.triples_dir = self.output_dir / 'output_triples'
        
        # Create output dirs
        for dir_path in [self.entities_dir, self.relations_dir, self.triples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractors
        self.use_graffiti = use_graffiti
        self.use_llm_fallback = use_llm_fallback
        
        self.graffiti = None
        self.llm_fallback = None
        
        if use_graffiti:
            try:
                self.graffiti = GraffitiClient.from_config()
                logger.info("Graffiti client initialized")
            except Exception as e:
                logger.warning(f"Graffiti init failed: {e}")
                self.use_graffiti = False
        
        if use_llm_fallback:
            try:
                self.llm_fallback = LLMFallbackExtractor.from_config()
                logger.info("LLM fallback initialized")
            except Exception as e:
                logger.warning(f"LLM fallback init failed: {e}")
                self.use_llm_fallback = False
        
        logger.info(f"Extraction engine ready: workers={workers}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.graffiti:
            await self.graffiti.__aenter__()
        if self.llm_fallback:
            await self.llm_fallback.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.graffiti:
            await self.graffiti.__aexit__(exc_type, exc_val, exc_tb)
        if self.llm_fallback:
            await self.llm_fallback.__aexit__(exc_type, exc_val, exc_tb)
    
    def _load_paper_data(self, paper_id: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Load paper text and metadata.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Tuple of (text, metadata_dict)
        """
        # Load text
        text_file = self.text_dir / f"{paper_id}.txt"
        if not text_file.exists():
            logger.error(f"Text file not found: {text_file}")
            return None, None
        
        text = text_file.read_text(encoding='utf-8')
        
        # Load metadata
        json_file = self.json_dir / f"{paper_id}.json"
        metadata = None
        
        if json_file.exists():
            try:
                with open(json_file) as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return text, metadata
    
    async def _extract_with_graffiti(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract using Graffiti.
        
        Args:
            text: Paper text
            
        Returns:
            Tuple of (success, result, error)
        """
        if not self.use_graffiti or not self.graffiti:
            return False, None, "Graffiti not available"
        
        try:
            return await self.graffiti.extract(text)
        except Exception as e:
            logger.error(f"Graffiti extraction error: {e}")
            return False, None, str(e)
    
    async def _extract_with_llm(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract using LLM fallback.
        
        Args:
            text: Paper text
            
        Returns:
            Tuple of (success, result, error)
        """
        if not self.use_llm_fallback or not self.llm_fallback:
            return False, None, "LLM fallback not available"
        
        try:
            return await self.llm_fallback.extract(text)
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return False, None, str(e)
    
    def _save_outputs(
        self,
        paper_id: str,
        entities: List[Dict],
        relations: List[Dict],
        triples: List[List[str]],
        metadata: Optional[Dict] = None
    ):
        """
        Save extraction outputs.
        
        Args:
            paper_id: Paper identifier
            entities: Entity list
            relations: Relation list
            triples: Triple list
            metadata: Optional paper metadata
        """
        # Save entities
        entities_file = self.entities_dir / f"{paper_id}.json"
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
        
        # Save relations
        relations_file = self.relations_dir / f"{paper_id}.json"
        with open(relations_file, 'w', encoding='utf-8') as f:
            json.dump(relations, f, indent=2, ensure_ascii=False)
        
        # Save triples
        triples_file = self.triples_dir / f"{paper_id}.json"
        triples_output = {
            'paper_id': paper_id,
            'triples': triples,
            'metadata': {
                'entity_count': len(entities),
                'relation_count': len(relations),
                'triple_count': len(triples),
                'extracted_at': datetime.now().isoformat()
            }
        }
        
        if metadata:
            triples_output['paper_metadata'] = {
                'title': metadata.get('title', ''),
                'authors': metadata.get('authors', []),
                'year': metadata.get('metadata', {}).get('year', '')
            }
        
        with open(triples_file, 'w', encoding='utf-8') as f:
            json.dump(triples_output, f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Saved {paper_id}: {len(entities)} entities, "
            f"{len(relations)} relations, {len(triples)} triples"
        )
    
    async def extract_single(self, paper_id: str) -> Tuple[bool, Optional[str]]:
        """
        Extract single paper.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Tuple of (success, error_message)
        """
        logger.info(f"Extracting: {paper_id}")
        
        try:
            # Load paper data
            text, metadata = self._load_paper_data(paper_id)
            
            if not text:
                return False, "Failed to load paper text"
            
            # Try Graffiti first
            success, result, error = await self._extract_with_graffiti(text)
            
            extraction_method = 'graffiti'
            
            # Fallback to LLM if Graffiti fails
            if not success:
                logger.warning(f"Graffiti failed for {paper_id}: {error}")
                logger.info(f"Trying LLM fallback for {paper_id}")
                
                success, result, error = await self._extract_with_llm(text)
                extraction_method = 'llm'
                
                if not success:
                    return False, f"Both extraction methods failed: {error}"
            
            logger.info(f"Extraction successful via {extraction_method}")
            
            # Normalize schema
            raw_entities = result.get('entities', [])
            raw_relations = result.get('relations', [])
            
            # Normalize entities
            normalized_entities = []
            for entity in raw_entities:
                norm = normalize_entity(entity)
                if norm:
                    normalized_entities.append(norm)
            
            # Normalize relations
            normalized_relations = []
            for relation in raw_relations:
                norm = normalize_relation(relation)
                if norm:
                    normalized_relations.append(norm)
            
            # Validate
            clean_entities, clean_relations = validate_schema(
                normalized_entities,
                normalized_relations
            )
            
            # Add claims back (already normalized in extraction)
            result['entities'] = clean_entities
            result['relations'] = clean_relations
            
            # Build triples
            builder = TripleBuilder(paper_id)
            entities, relations, triples = builder.build(result)
            
            # Validate triples
            valid, validation_error = TripleBuilder.validate_triples(
                entities, relations, triples
            )
            
            if not valid:
                logger.error(f"Triple validation failed: {validation_error}")
                return False, f"Triple validation failed: {validation_error}"
            
            # Save outputs
            self._save_outputs(paper_id, entities, relations, triples, metadata)
            
            logger.info(f"✅ Success: {paper_id}")
            return True, None
            
        except Exception as e:
            error_msg = f"Extraction failed: {e}"
            logger.error(f"❌ {paper_id}: {error_msg}")
            return False, error_msg
    
    def _find_papers(self) -> List[str]:
        """
        Find all papers in JSON directory.
        
        Returns:
            List of paper IDs
        """
        if not self.json_dir.exists():
            logger.error(f"JSON directory not found: {self.json_dir}")
            return []
        
        json_files = list(self.json_dir.glob('*.json'))
        paper_ids = [f.stem for f in json_files]
        
        logger.info(f"Found {len(paper_ids)} papers")
        return paper_ids
    
    async def extract_all(self, paper_ids: Optional[List[str]] = None) -> Dict:
        """
        Extract all papers in parallel.
        
        Args:
            paper_ids: Optional list of paper IDs (defaults to all)
            
        Returns:
            Summary statistics
        """
        if paper_ids is None:
            paper_ids = self._find_papers()
        
        if not paper_ids:
            logger.warning("No papers to extract")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        logger.info(f"Extracting {len(paper_ids)} papers with {self.workers} workers")
        
        # Process in parallel with semaphore
        semaphore = asyncio.Semaphore(self.workers)
        
        async def extract_with_semaphore(paper_id):
            async with semaphore:
                return await self.extract_single(paper_id)
        
        # Run all extractions
        results = await asyncio.gather(
            *[extract_with_semaphore(pid) for pid in paper_ids],
            return_exceptions=True
        )
        
        # Collect statistics
        success_count = 0
        failed_count = 0
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                errors.append(f"{paper_ids[i]}: {result}")
            elif result[0]:  # success
                success_count += 1
            else:
                failed_count += 1
                errors.append(f"{paper_ids[i]}: {result[1]}")
        
        stats = {
            'total': len(paper_ids),
            'success': success_count,
            'failed': failed_count,
            'success_rate': success_count / len(paper_ids) * 100 if paper_ids else 0,
            'errors': errors[:10]  # First 10 errors
        }
        
        logger.info(
            f"\n{'='*80}\n"
            f"✅ Extraction complete!\n"
            f"   Total: {stats['total']}\n"
            f"   Success: {stats['success']}\n"
            f"   Failed: {stats['failed']}\n"
            f"   Success rate: {stats['success_rate']:.1f}%\n"
            f"{'='*80}"
        )
        
        return stats


# CLI entry point
async def main():
    """Main CLI function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 2: Extract entities, relations, and triples from papers'
    )
    parser.add_argument(
        '--json-dir',
        type=Path,
        default=Path('ingestion/output_json'),
        help='Phase 1 JSON directory'
    )
    parser.add_argument(
        '--text-dir',
        type=Path,
        default=Path('ingestion/raw_text'),
        help='Phase 1 text directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('extraction'),
        help='Output directory'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--paper-id',
        type=str,
        help='Extract single paper by ID'
    )
    parser.add_argument(
        '--no-graffiti',
        action='store_true',
        help='Disable Graffiti extraction'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM fallback'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create engine
    engine = ExtractionEngine(
        json_dir=args.json_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        use_graffiti=not args.no_graffiti,
        use_llm_fallback=not args.no_llm,
        workers=args.workers
    )
    
    async with engine:
        if args.paper_id:
            # Extract single paper
            success, error = await engine.extract_single(args.paper_id)
            if success:
                print(f"✅ Successfully extracted: {args.paper_id}")
            else:
                print(f"❌ Extraction failed: {error}")
        else:
            # Extract all papers
            stats = await engine.extract_all()
            print(f"\n{'='*80}")
            print(f"📊 EXTRACTION SUMMARY")
            print(f"{'='*80}")
            print(f"Total papers: {stats['total']}")
            print(f"Successful: {stats['success']}")
            print(f"Failed: {stats['failed']}")
            print(f"Success rate: {stats['success_rate']:.1f}%")
            
            if stats['errors']:
                print(f"\n❌ First {len(stats['errors'])} errors:")
                for error in stats['errors']:
                    print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
