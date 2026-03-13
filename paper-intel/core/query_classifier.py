"""
Lightweight Query Difficulty Classifier.

Routes queries to the optimal retrieval configuration based on linguistic features.
"""

import re
from typing import Dict, Union, Set, Optional


def classify_query(
    query: str, vocabulary: Optional[Set[str]] = None
) -> Dict[str, Union[str, float]]:
    """
    Classify a query into a category and recommend retrieval adjustments.
    
    Categories:
      - 'keyword_focused' : short, dense, many entities/nouns
      - 'conceptual'      : long, abstract words, question-like
      - 'mixed'           : average queries
    
    Returns:
       dict with recommended weights: {
           'class': str,
           'dense_weight': float,
           'bm25_weight': float,
           'graph_budget_multiplier': float
       }
    """
    # Extract alpha-numeric tokens
    tokens = [t.lower() for t in re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', query)]
    length = len(tokens)
    
    # Simple heuristics
    question_words = {"what", "how", "why", "where", "when", "does", "is", "are"}
    has_q_word = any(w in question_words for w in tokens)
    
    oov_ratio = 0.0
    if vocabulary:
        oov_ratio = sum(1 for t in tokens if t not in vocabulary) / max(1, length)
        
    is_keyword = length <= 4 and not has_q_word
    is_conceptual = length >= 8 or has_q_word or oov_ratio > 0.3
    
    if is_keyword:
        return {
            "class": "keyword_focused",
            "dense_weight": 0.4,
            "bm25_weight": 0.6,
            "graph_budget_multiplier": 0.5
        }
    elif is_conceptual:
        return {
            "class": "conceptual",
            "dense_weight": 0.7,
            "bm25_weight": 0.3,
            "graph_budget_multiplier": 1.5
        }
    else:
        return {
            "class": "mixed",
            "dense_weight": 0.5,
            "bm25_weight": 0.5,
            "graph_budget_multiplier": 1.0
        }
