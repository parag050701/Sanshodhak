"""
Simulated Inter-Annotator Agreement (IAA) demonstrator.
Generates two annotation JSON files for demonstration purposes
with a controlled agreement level (Cohen's Kappa ~ 0.85).
"""

import json
import random
from pathlib import Path

def generate_simulated_annotations(output_dir: str):
    random.seed(42)  # For reproducibility
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    annotator1_data = []
    annotator2_data = []
    
    topics = ["RAG architecture", "Graph Expansion", "Information Retrieval", "BM25 tuning"]
    
    for i in range(1, 101): # 100 simulated annotations
        topic = random.choice(topics)
        # Base truth
        relevance1 = random.choice(["Relevant", "Not Relevant", "Partially Relevant"])
        
        # ~85% agreement
        if random.random() < 0.85:
            relevance2 = relevance1
        else:
            options = ["Relevant", "Not Relevant", "Partially Relevant"]
            options.remove(relevance1)
            relevance2 = random.choice(options)
            
        annotator1_data.append({
            "doc_id": f"doc_{i:03d}",
            "topic": topic,
            "relevance": relevance1
        })
        
        annotator2_data.append({
            "doc_id": f"doc_{i:03d}",
            "topic": topic,
            "relevance": relevance2
        })
        
    with open(path / "annotations_annotator1.json", "w") as f:
        json.dump(annotator1_data, f, indent=2)
        
    with open(path / "annotations_annotator2.json", "w") as f:
        json.dump(annotator2_data, f, indent=2)
        
    print(f"✅ Generated 100 simulated annotations at {path}/annotations_annotator*.json")
    print("These files demonstrate the IAA framework functionality with an expected agreement of ~85%.")

if __name__ == "__main__":
    # Ensure it's run from the right place, typically paper-intel 
    generate_simulated_annotations("/home/admin-/Desktop/Sanshodhak/paper-intel")
