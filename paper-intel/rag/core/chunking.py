"""
Semantic chunking with overlap and metadata preservation.
"""
import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Document chunk with metadata."""
    text: str
    paper_id: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class SemanticChunker:
    """Semantic chunking based on sentence boundaries."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e)\.',
                     lambda m: m.group().replace('.', '<DOT>'), text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_document(
        self,
        text: str,
        paper_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        Chunk document semantically.
        
        Args:
            text: Document text
            paper_id: Paper identifier
            metadata: Additional metadata
            
        Returns:
            List of chunks
        """
        if metadata is None:
            metadata = {}
        
        sentences = self._split_sentences(text)
        chunks = []
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding sentence exceeds max size
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        paper_id=paper_id,
                        chunk_id=chunk_id,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={**metadata, 'num_sentences': len(current_chunk)}
                    ))
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_tokens = []
                overlap_length = 0
                
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.chunk_overlap:
                        overlap_tokens.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_tokens
                current_length = overlap_length
                start_char = end_char - overlap_length if chunks else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
            
            # Check if chunk is complete
            if current_length >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        paper_id=paper_id,
                        chunk_id=chunk_id,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={**metadata, 'num_sentences': len(current_chunk)}
                    ))
                    chunk_id += 1
                
                # Reset with overlap
                overlap_tokens = []
                overlap_length = 0
                
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.chunk_overlap:
                        overlap_tokens.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_tokens
                current_length = overlap_length
                start_char = end_char - overlap_length if chunks else 0
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                end_char = start_char + len(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={**metadata, 'num_sentences': len(current_chunk)}
                ))
        
        logger.debug(f"Chunked {paper_id}: {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
