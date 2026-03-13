"""
Triple Builder - Convert entities + relations + claims into Neo4j-ready triples.

Handles deduplication, ID assignment, and triple construction.
"""

import logging
from typing import Dict, List, Tuple, Set
import hashlib

logger = logging.getLogger(__name__)


class TripleBuilder:
    """
    Builds knowledge graph triples from extracted entities and relations.
    
    Responsibilities:
    - Deduplicate entities per paper
    - Assign unique IDs (E_<paper_id>_<counter>)
    - Create relation edges with correct entity IDs
    - Convert claims into Result nodes + relations
    - Generate triples (entity) -[relation]-> (entity)
    """
    
    def __init__(self, paper_id: str):
        """
        Initialize triple builder for a paper.
        
        Args:
            paper_id: Unique paper identifier
        """
        self.paper_id = paper_id
        self.entity_counter = 0
        self.entity_map = {}  # name+type -> entity_id
        
    def _generate_entity_id(self) -> str:
        """
        Generate unique entity ID.
        
        Returns:
            Entity ID in format E_<paper_id>_<counter>
        """
        self.entity_counter += 1
        return f"E_{self.paper_id}_{self.entity_counter}"
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Deduplicate entities by name+type, assign IDs.
        
        Args:
            entities: List of entity dicts
            
        Returns:
            List of unique entities with IDs
        """
        unique_entities = []
        
        for entity in entities:
            name = entity.get('name', '').strip()
            entity_type = entity.get('type', '')
            
            if not name or not entity_type:
                continue
            
            # Create unique key
            key = (name.lower(), entity_type)
            
            # Skip if already seen
            if key in self.entity_map:
                continue
            
            # Assign new ID
            entity_id = self._generate_entity_id()
            self.entity_map[key] = entity_id
            
            # Create entity with ID
            unique_entity = {
                'id': entity_id,
                'type': entity_type,
                'name': name,
                'source_paper_id': self.paper_id
            }
            
            # Preserve additional fields
            for field in ['description', 'aliases', 'properties', 'metadata']:
                if field in entity:
                    unique_entity[field] = entity[field]
            
            unique_entities.append(unique_entity)
        
        logger.info(f"Deduplicated {len(entities)} → {len(unique_entities)} entities")
        return unique_entities
    
    def _resolve_entity_id(self, entity_ref: any) -> str:
        """
        Resolve entity reference to ID.
        
        Args:
            entity_ref: Entity name or dict
            
        Returns:
            Entity ID or original reference
        """
        if isinstance(entity_ref, dict):
            # Extract name and type
            name = entity_ref.get('name', '').strip().lower()
            entity_type = entity_ref.get('type', '')
            key = (name, entity_type)
            
            return self.entity_map.get(key, name)
        
        elif isinstance(entity_ref, str):
            # Try to find by name in any type
            ref_lower = entity_ref.strip().lower()
            
            for (name, _), entity_id in self.entity_map.items():
                if name == ref_lower:
                    return entity_id
            
            return entity_ref
        
        return str(entity_ref)
    
    def _build_relations(
        self,
        relations: List[Dict],
        entities: List[Dict]
    ) -> List[Dict]:
        """
        Build relations with resolved entity IDs.
        
        Args:
            relations: List of relation dicts
            entities: List of deduplicated entities
            
        Returns:
            List of relations with entity IDs
        """
        built_relations = []
        
        for relation in relations:
            source = relation.get('source')
            target = relation.get('target')
            rel_type = relation.get('type')
            
            if not source or not target or not rel_type:
                continue
            
            # Resolve entity IDs
            source_id = self._resolve_entity_id(source)
            target_id = self._resolve_entity_id(target)
            
            # Create relation
            built_relation = {
                'source': source_id,
                'type': rel_type,
                'target': target_id,
                'source_paper_id': self.paper_id
            }
            
            # Preserve properties
            if 'properties' in relation:
                built_relation['properties'] = relation['properties']
            
            if 'confidence' in relation:
                built_relation['confidence'] = relation['confidence']
            
            built_relations.append(built_relation)
        
        logger.info(f"Built {len(built_relations)} relations")
        return built_relations
    
    def _claims_to_entities_and_relations(
        self,
        claims: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert claims into Result entities and relations.
        
        Args:
            claims: List of claim dicts
            
        Returns:
            Tuple of (result_entities, metric_relations)
        """
        result_entities = []
        metric_relations = []
        
        for claim in claims:
            metric_name = claim.get('metric_name')
            metric_value = claim.get('metric_value')
            dataset = claim.get('dataset')
            
            if not metric_name or not metric_value:
                continue
            
            # Create Result entity
            result_name = f"{metric_name}={metric_value}"
            if dataset:
                result_name += f" on {dataset}"
            
            result_id = self._generate_entity_id()
            
            result_entity = {
                'id': result_id,
                'type': 'Result',
                'name': result_name,
                'source_paper_id': self.paper_id,
                'properties': {
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'dataset': dataset,
                    'experiment_description': claim.get('experiment_description', ''),
                    'improvement_over_baseline': claim.get('improvement_over_baseline', '')
                }
            }
            
            result_entities.append(result_entity)
            
            # Add to entity map
            key = (result_name.lower(), 'Result')
            self.entity_map[key] = result_id
            
            # Create REPORTS_METRIC relation to Metric entity
            metric_key = (metric_name.lower(), 'Metric')
            if metric_key in self.entity_map:
                metric_id = self.entity_map[metric_key]
                
                metric_relations.append({
                    'source': result_id,
                    'type': 'REPORTS_METRIC',
                    'target': metric_id,
                    'source_paper_id': self.paper_id
                })
            
            # Create EVALUATES_ON relation to Dataset if present
            if dataset:
                dataset_key = (dataset.lower(), 'Dataset')
                if dataset_key in self.entity_map:
                    dataset_id = self.entity_map[dataset_key]
                    
                    metric_relations.append({
                        'source': result_id,
                        'type': 'EVALUATES_ON',
                        'target': dataset_id,
                        'source_paper_id': self.paper_id
                    })
        
        logger.info(f"Created {len(result_entities)} result entities from claims")
        return result_entities, metric_relations
    
    def _generate_triples(
        self,
        entities: List[Dict],
        relations: List[Dict]
    ) -> List[List[str]]:
        """
        Generate simple triples: [entity_name, relation_type, entity_name].
        
        Args:
            entities: List of entity dicts with IDs
            relations: List of relation dicts with IDs
            
        Returns:
            List of triples
        """
        # Build ID -> name mapping
        id_to_name = {e['id']: e['name'] for e in entities}
        
        triples = []
        
        for relation in relations:
            source_id = relation['source']
            target_id = relation['target']
            rel_type = relation['type']
            
            # Resolve names
            source_name = id_to_name.get(source_id, source_id)
            target_name = id_to_name.get(target_id, target_id)
            
            triple = [source_name, rel_type, target_name]
            triples.append(triple)
        
        logger.info(f"Generated {len(triples)} triples")
        return triples
    
    def build(
        self,
        extraction_result: Dict
    ) -> Tuple[List[Dict], List[Dict], List[List[str]]]:
        """
        Build complete knowledge graph components from extraction.
        
        Args:
            extraction_result: Dict with 'entities', 'relations', 'claims'
            
        Returns:
            Tuple of (entities, relations, triples)
        """
        logger.info(f"Building triples for paper: {self.paper_id}")
        
        # Extract components
        raw_entities = extraction_result.get('entities', [])
        raw_relations = extraction_result.get('relations', [])
        claims = extraction_result.get('claims', [])
        
        # Step 1: Deduplicate and assign IDs to entities
        entities = self._deduplicate_entities(raw_entities)
        
        # Step 2: Convert claims to entities and relations
        result_entities, claim_relations = self._claims_to_entities_and_relations(claims)
        entities.extend(result_entities)
        
        # Step 3: Build relations with resolved IDs
        relations = self._build_relations(raw_relations, entities)
        relations.extend(claim_relations)
        
        # Step 4: Generate simple triples
        triples = self._generate_triples(entities, relations)
        
        logger.info(
            f"Built: {len(entities)} entities, "
            f"{len(relations)} relations, "
            f"{len(triples)} triples"
        )
        
        return entities, relations, triples
    
    @staticmethod
    def validate_triples(
        entities: List[Dict],
        relations: List[Dict],
        triples: List[List[str]]
    ) -> Tuple[bool, str]:
        """
        Validate triple structure.
        
        Args:
            entities: Entity list
            relations: Relation list
            triples: Triple list
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check entities have required fields
        for entity in entities:
            if 'id' not in entity:
                return False, "Entity missing 'id'"
            if 'type' not in entity:
                return False, "Entity missing 'type'"
            if 'name' not in entity:
                return False, "Entity missing 'name'"
        
        # Check relations have required fields
        entity_ids = {e['id'] for e in entities}
        
        for relation in relations:
            if 'source' not in relation:
                return False, "Relation missing 'source'"
            if 'target' not in relation:
                return False, "Relation missing 'target'"
            if 'type' not in relation:
                return False, "Relation missing 'type'"
            
            # Validate entity references
            if relation['source'] not in entity_ids:
                logger.warning(f"Relation references unknown source: {relation['source']}")
            
            if relation['target'] not in entity_ids:
                logger.warning(f"Relation references unknown target: {relation['target']}")
        
        # Check triples are valid
        for triple in triples:
            if len(triple) != 3:
                return False, f"Invalid triple format: {triple}"
        
        return True, ""


# Test
def test_triple_builder():
    """Test triple builder."""
    extraction = {
        'entities': [
            {'type': 'Model', 'name': 'BERT'},
            {'type': 'Dataset', 'name': 'GLUE'},
            {'type': 'Metric', 'name': 'Accuracy'},
            {'type': 'Model', 'name': 'BERT'}  # Duplicate
        ],
        'relations': [
            {'source': 'BERT', 'type': 'EVALUATES_ON', 'target': 'GLUE'},
            {'source': {'name': 'BERT', 'type': 'Model'}, 
             'type': 'REPORTS_METRIC', 
             'target': {'name': 'Accuracy', 'type': 'Metric'}}
        ],
        'claims': [
            {
                'metric_name': 'Accuracy',
                'metric_value': '92.4%',
                'dataset': 'GLUE',
                'experiment_description': 'BERT-base on GLUE benchmark'
            }
        ]
    }
    
    builder = TripleBuilder('test_paper_123')
    entities, relations, triples = builder.build(extraction)
    
    print(f"\n✅ Built {len(entities)} entities")
    for entity in entities[:5]:
        print(f"  {entity['id']}: {entity['type']} - {entity['name']}")
    
    print(f"\n✅ Built {len(relations)} relations")
    for relation in relations[:5]:
        print(f"  {relation['source']} -[{relation['type']}]-> {relation['target']}")
    
    print(f"\n✅ Built {len(triples)} triples")
    for triple in triples[:5]:
        print(f"  {triple}")
    
    # Validate
    valid, error = TripleBuilder.validate_triples(entities, relations, triples)
    print(f"\n{'✅' if valid else '❌'} Validation: {error or 'OK'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_triple_builder()
