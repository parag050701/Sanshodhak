import json
import re

def evaluate_factual_grounding(demo_file):
    print("Evaluating factual grounding rate...")
    try:
        with open(demo_file, 'r') as f:
            data = json.load(f)
        
        total_sentences = 0
        supported_sentences = 0
        
        for qa in data:
            if 'answer' in qa:
                # Basic proxy for sentence splitting and citation checking
                sentences = [s.strip() for s in qa['answer'].split('.') if len(s.strip()) > 10]
                for sentence in sentences:
                    total_sentences += 1
                    # Check if sentence contains a citation marker like [1], [2]
                    if re.search(r'\[\d+\]', sentence):
                        supported_sentences += 1
        
        rate = (supported_sentences / total_sentences) if total_sentences > 0 else 0
        return rate
    except Exception as e:
        print(f"Error evaluating grounding: {e}")
        return 0.0

if __name__ == "__main__":
    rate = evaluate_factual_grounding('demo_answers.json')
    print(f"Factual Grounding Rate: {rate:.2%}")
    with open('grounding_results.json', 'w') as f:
        json.dump({"factual_grounding_rate": rate}, f)
