"""
Knowledge Graph Schema - Canonical entity and relation types.

Defines allowed entity types, relation types, and normalization functions.
"""

from typing import Dict, Optional, Set
import re
import logging

logger = logging.getLogger(__name__)

# Canonical entity types
ENTITY_TYPES: Set[str] = {
    'Model',
    'Method',
    'Task',
    'Dataset',
    'Metric',
    'Result',
    'Paper',
    'Author'
}

# Canonical relation types
RELATION_TYPES: Set[str] = {
    'USES_MODEL',
    'USES_METHOD',
    'USES_DATASET',
    'EVALUATES_ON',
    'REPORTS_METRIC',
    'ACHIEVES_RESULT',
    'COMPARES_WITH',
    'CITES',
    'AUTHORED_BY',
    'STUDIES_TASK'
}

# Entity type aliases/mappings
ENTITY_ALIASES: Dict[str, str] = {
    'algorithm': 'Method',
    'technique': 'Method',
    'approach': 'Method',
    'framework': 'Method',
    'architecture': 'Model',
    'neural_network': 'Model',
    'neural network': 'Model',
    'transformer': 'Model',
    'problem': 'Task',
    'application': 'Task',
    'benchmark': 'Dataset',
    'corpus': 'Dataset',
    'evaluation': 'Metric',
    'score': 'Metric',
    'performance': 'Result',
    'finding': 'Result',
    'outcome': 'Result',
    'publication': 'Paper',
    'article': 'Paper',
    'researcher': 'Author',
    'scientist': 'Author'
}

# Relation type aliases
RELATION_ALIASES: Dict[str, str] = {
    'uses': 'USES_METHOD',
    'applies': 'USES_METHOD',
    'employs': 'USES_METHOD',
    'implements': 'USES_METHOD',
    'evaluates': 'EVALUATES_ON',
    'tests': 'EVALUATES_ON',
    'benchmarks': 'EVALUATES_ON',
    'reports': 'REPORTS_METRIC',
    'measures': 'REPORTS_METRIC',
    'achieves': 'ACHIEVES_RESULT',
    'obtains': 'ACHIEVES_RESULT',
    'compares': 'COMPARES_WITH',
    'cites': 'CITES',
    'references': 'CITES',
    'written_by': 'AUTHORED_BY',
    'authored': 'AUTHORED_BY',
    'addresses': 'STUDIES_TASK',
    'solves': 'STUDIES_TASK',
    'tackles': 'STUDIES_TASK'
}


def normalize_entity_type(raw_type: str) -> Optional[str]:
    """
    Normalize entity type to canonical form.
    
    Args:
        raw_type: Raw entity type string
        
    Returns:
        Canonical entity type or None if invalid
    """
    if not raw_type:
        return None
    
    # Clean and normalize
    clean = raw_type.strip().lower()
    clean = re.sub(r'[_\-\s]+', '_', clean)
    
    # Check if already canonical
    for entity_type in ENTITY_TYPES:
        if clean == entity_type.lower():
            return entity_type
    
    # Check aliases
    if clean in ENTITY_ALIASES:
        return ENTITY_ALIASES[clean]
    
    # Try partial match
    for alias, canonical in ENTITY_ALIASES.items():
        if alias in clean or clean in alias:
            return canonical
    
    logger.warning(f"Unknown entity type: {raw_type}")
    return None


def normalize_relation_type(raw_relation: str) -> Optional[str]:
    """
    Normalize relation type to canonical form.
    
    Args:
        raw_relation: Raw relation type string
        
    Returns:
        Canonical relation type or None if invalid
    """
    if not raw_relation:
        return None
    
    # Clean and normalize
    clean = raw_relation.strip().lower()
    clean = re.sub(r'[_\-\s]+', '_', clean)
    
    # Check if already canonical
    for relation_type in RELATION_TYPES:
        if clean == relation_type.lower():
            return relation_type
    
    # Check aliases
    if clean in RELATION_ALIASES:
        return RELATION_ALIASES[clean]
    
    # Try partial match
    for alias, canonical in RELATION_ALIASES.items():
        if alias in clean or clean in alias:
            return canonical
    
    logger.warning(f"Unknown relation type: {raw_relation}")
    return None


def normalize_entity_name(raw_name: str) -> str:
    """
    Normalize entity name for consistency.
    
    Args:
        raw_name: Raw entity name
        
    Returns:
        Cleaned entity name
    """
    if not raw_name:
        return ""
    
    # Remove extra whitespace
    clean = ' '.join(raw_name.split())
    
    # Remove special characters but keep alphanumeric, spaces, hyphens
    clean = re.sub(r'[^\w\s\-\.\+]', '', clean)
    
    # Trim
    clean = clean.strip()
    
    return clean


def normalize_entity(raw_entity: Dict) -> Optional[Dict]:
    """
    Normalize entity to canonical schema.
    
    Args:
        raw_entity: Raw entity dict with at least 'type' and 'name'
        
    Returns:
        Normalized entity dict or None if invalid
    """
    if not raw_entity or not isinstance(raw_entity, dict):
        return None
    
    # Extract and normalize type
    raw_type = raw_entity.get('type', '')
    entity_type = normalize_entity_type(raw_type)
    
    if not entity_type:
        logger.warning(f"Invalid entity type: {raw_type}")
        return None
    
    # Extract and normalize name
    raw_name = raw_entity.get('name', '')
    entity_name = normalize_entity_name(raw_name)
    
    if not entity_name:
        logger.warning(f"Empty entity name for type: {entity_type}")
        return None
    
    # Build normalized entity
    normalized = {
        'type': entity_type,
        'name': entity_name
    }
    
    # Optional fields
    if 'id' in raw_entity:
        normalized['id'] = raw_entity['id']
    
    if 'source_paper_id' in raw_entity:
        normalized['source_paper_id'] = raw_entity['source_paper_id']
    
    # Preserve additional metadata
    for key in ['description', 'aliases', 'properties', 'metadata']:
        if key in raw_entity:
            normalized[key] = raw_entity[key]
    
    return normalized


def normalize_relation(raw_relation: Dict) -> Optional[Dict]:
    """
    Normalize relation to canonical schema.
    
    Args:
        raw_relation: Raw relation dict with 'source', 'type', 'target'
        
    Returns:
        Normalized relation dict or None if invalid
    """
    if not raw_relation or not isinstance(raw_relation, dict):
        return None
    
    # Extract relation type
    raw_type = raw_relation.get('type', '')
    relation_type = normalize_relation_type(raw_type)
    
    if not relation_type:
        logger.warning(f"Invalid relation type: {raw_type}")
        return None
    
    # Extract source and target
    source = raw_relation.get('source')
    target = raw_relation.get('target')
    
    if not source or not target:
        logger.warning(f"Missing source or target in relation: {raw_relation}")
        return None
    
    # Build normalized relation
    normalized = {
        'source': source,
        'type': relation_type,
        'target': target
    }
    
    # Optional fields
    if 'properties' in raw_relation:
        normalized['properties'] = raw_relation['properties']
    
    if 'confidence' in raw_relation:
        normalized['confidence'] = raw_relation['confidence']
    
    if 'source_paper_id' in raw_relation:
        normalized['source_paper_id'] = raw_relation['source_paper_id']
    
    return normalized


def validate_schema(entities: list, relations: list) -> tuple[list, list]:
    """
    Validate and clean entities and relations.
    
    Args:
        entities: List of entity dicts
        relations: List of relation dicts
        
    Returns:
        Tuple of (cleaned_entities, cleaned_relations)
    """
    # Normalize entities
    cleaned_entities = []
    for entity in entities:
        normalized = normalize_entity(entity)
        if normalized:
            cleaned_entities.append(normalized)
    
    # Build entity ID set for validation
    entity_ids = {e.get('id') for e in cleaned_entities if e.get('id')}
    
    # Normalize relations and validate references
    cleaned_relations = []
    for relation in relations:
        normalized = normalize_relation(relation)
        if normalized:
            # Validate that source and target exist (if IDs are used)
            source = normalized['source']
            target = normalized['target']
            
            # If using IDs, check they exist
            if isinstance(source, str) and source.startswith('E_'):
                if source not in entity_ids:
                    logger.warning(f"Relation references non-existent source: {source}")
                    continue
            
            if isinstance(target, str) and target.startswith('E_'):
                if target not in entity_ids:
                    logger.warning(f"Relation references non-existent target: {target}")
                    continue
            
            cleaned_relations.append(normalized)
    
    logger.info(f"Validated: {len(cleaned_entities)} entities, {len(cleaned_relations)} relations")
    
    return cleaned_entities, cleaned_relations


# Export schema info
def get_schema_info() -> Dict:
    """Get schema information."""
    return {
        'entity_types': list(ENTITY_TYPES),
        'relation_types': list(RELATION_TYPES),
        'entity_aliases': ENTITY_ALIASES,
        'relation_aliases': RELATION_ALIASES
    }
