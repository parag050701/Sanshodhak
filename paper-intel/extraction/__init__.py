"""
Phase 2: Semantic Extraction Layer

Transforms Phase 1 outputs (JSON + text) into knowledge graph triples.
"""

from .extraction_engine import ExtractionEngine
from .schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    normalize_entity,
    normalize_relation
)

__all__ = [
    'ExtractionEngine',
    'ENTITY_TYPES',
    'RELATION_TYPES',
    'normalize_entity',
    'normalize_relation'
]

__version__ = '1.0.0'
