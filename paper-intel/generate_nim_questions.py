#!/usr/bin/env python3
"""
NIM-based large-scale question candidate generation.

Generates diverse candidate questions from corpus papers using NIM or OpenRouter,
filters by resolvability heuristics, and exports ranked candidates for sampling.

Usage:
  python generate_nim_questions.py --corpus data/corpus_merged.json \
    --output data/candidate_questions.json \
    --num_questions 1000 \
    --batch_size 8
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Dict

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NIM-based questions from corpus")
    parser.add_argument("--corpus", required=True, help="Path to merged corpus JSON")
    parser.add_argument("--output", required=True, help="Output candidate questions JSON")
    parser.add_argument("--num_questions", type=int, default=1000, help="Target number of questions")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--api_base", default=None, help="NIM API base URL (e.g., https://integrate.api.nvidia.com/v1)")
    parser.add_argument("--api_key", default=None, help="NIM API key (nvapi-*)")
    parser.add_argument("--model", default=None, help="Model ID (e.g., qwen/qwen2.5-coder-32b-instruct)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--timeout", type=int, default=60, help="API request timeout")
    return parser.parse_args()


def load_corpus(path: str) -> List[dict]:
    """Load merged corpus JSON."""
    with open(path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    if isinstance(corpus_data, list):
        return corpus_data
    if isinstance(corpus_data, dict) and "papers" in corpus_data:
        return corpus_data["papers"]
    raise ValueError(f"Unexpected corpus format in {path}")


def sample_papers(papers: List[dict], num_samples: int) -> List[dict]:
    """Sample papers proportional to citation count."""
    if len(papers) <= num_samples:
        return papers

    weights = [max(1, p.get("citations", 0)) for p in papers]
    return random.choices(papers, weights=weights, k=num_samples)


def generate_questions_nim(
    papers: List[dict],
    api_base: str,
    api_key: str,
    model: str,
    batch_size: int,
    temperature: float,
    timeout: int,
) -> List[dict]:
    """Generate questions using OpenAI-compatible NIM API (NVIDIA)."""
    if not _OPENAI_AVAILABLE:
        logger.error("openai library required; install: pip install openai")
        return []

    questions = []
    sampled = sample_papers(papers, min(len(papers), 500))  # Cap at 500 to avoid API cost

    prompt_template = """Given this research paper:
Title: {title}
Abstract: {abstract}

Generate 2 diverse, specific research questions that could be answered by this paper.
Format as JSON array: ["question 1", "question 2"]
Questions should be:
- Specific and measurable
- Related to retrieval, RAG, or information systems
- Not already obvious from the title/abstract
Only output JSON, no explanation."""

    logger.info(f"Using API endpoint: {api_base}")
    logger.info(f"Model: {model}")

    client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)

    for paper in tqdm(sampled, desc="Generating questions"):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")[:500]  # Truncate long abstracts

        if not title or not abstract:
            continue

        prompt = prompt_template.format(title=title, abstract=abstract)

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=0.9,
                max_tokens=200,
            )
            content = completion.choices[0].message.content.strip()

            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for q in parsed:
                        if isinstance(q, str) and len(q) > 20 and len(q) < 500:
                            questions.append({
                                "question": q,
                                "paper_title": title,
                                "paper_doi": paper.get("doi"),
                                "source": "nim_generated",
                                "quality": "candidate",
                            })
            except json.JSONDecodeError:
                logger.debug(f"Could not parse JSON from API response: {content[:100]}")

        except Exception as e:
            logger.debug(f"Error generating question: {e}")

    return questions


def generate_questions_template(papers: List[dict], num_questions: int) -> List[dict]:
    """Fallback: generate questions using simple templates (no API)."""
    questions = []
    templates = [
        "How does {topic} compare to {baseline}?",
        "What are the limitations of {topic}?",
        "Can {topic} be combined with {related}?",
        "How does {topic} scale to {scenario}?",
        "What novel applications could {topic} enable?",
        "How does domain knowledge improve {topic}?",
        "What evaluation metrics are most suitable for {topic}?",
    ]

    sampled = sample_papers(papers, min(len(papers), 200))
    if not sampled:
        return []

    variant = 0
    while len(questions) < num_questions:
        for paper in sampled:
            title = paper.get("title", "").split()[0:5]  # First few words
            keywords = paper.get("keywords", ["retrieval"])[:3]

            for template in templates:
                q = template.format(
                    topic=" ".join(title),
                    baseline=keywords[0] if keywords else "baseline",
                    related=keywords[1] if len(keywords) > 1 else "related methods",
                    scenario="large-scale retrieval",
                )
                if variant > 0:
                    q = f"{q} (variant {variant})"

                questions.append({
                    "question": q,
                    "paper_title": paper.get("title"),
                    "paper_doi": paper.get("doi"),
                    "source": "template_generated",
                    "quality": "candidate",
                })

                if len(questions) >= num_questions:
                    break

            if len(questions) >= num_questions:
                break

        variant += 1

    return questions[:num_questions]


def rank_questions(questions: List[dict]) -> List[dict]:
    """Rank questions by resolvability heuristics."""
    for q in questions:
        score = 0.0
        text = q["question"].lower()

        # Prefer how/what/why (open-ended, question-driven)
        if any(text.startswith(w) for w in ["how", "what", "why", "can"]):
            score += 2.0

        # Prefer specific technical terms
        tech_terms = ["retrieve", "rank", "embed", "index", "graph", "sparse", "dense", "fusion", "hybrid"]
        score += sum(1.0 for t in tech_terms if t in text)

        # Penalize very long questions
        if len(q["question"]) > 200:
            score -= 1.0

        q["resolvability_score"] = score

    return sorted(questions, key=lambda x: x["resolvability_score"], reverse=True)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading corpus from {args.corpus}...")
    papers = load_corpus(args.corpus)
    logger.info(f"Loaded {len(papers)} papers")

    # Load from environment, with defaults for NVIDIA NIM
    api_base = args.api_base or os.getenv("NIM_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key = args.api_key or os.getenv("NIM_API_KEY", "")
    model = args.model or os.getenv("NIM_MODEL", "qwen/qwen2.5-coder-32b-instruct")

    # Try API-based generation first
    questions = []
    if api_key and api_base:
        logger.info(f"Generating questions via NVIDIA NIM...")
        logger.info(f"Endpoint: {api_base}")
        try:
            questions = generate_questions_nim(
                papers, api_base, api_key, model, args.batch_size, args.temperature, args.timeout
            )
            logger.info(f"API generated {len(questions)} questions")
        except Exception as e:
            logger.warning(f"API generation failed: {e}; falling back to templates")

    # Fallback to template-based if API didn't produce enough
    if len(questions) < args.num_questions * 0.5:
        logger.info("Supplementing with template-based questions...")
        template_questions = generate_questions_template(papers, args.num_questions - len(questions))
        questions.extend(template_questions)
        logger.info(f"Template generated {len(template_questions)} questions")

    # Rank and select top questions
    ranked = rank_questions(questions)
    selected = ranked[: args.num_questions]

    logger.info(f"Generated {len(selected)} candidate questions")

    # Export
    output_data = {
        "source_corpus": args.corpus,
        "num_papers": len(papers),
        "num_papers_used": len(set(q.get("paper_doi") or q.get("paper_title", "unknown") for q in selected)),
        "questions": selected,
        "generated_at": time.time(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Candidate questions saved to {output_path}")
    logger.info(f"   - Total candidates: {len(selected)}")
    
    # Count by source
    source_counts = {}
    for q in selected:
        src = q.get('source', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1
    logger.info(f"   - Source distribution: {source_counts}")


if __name__ == "__main__":
    main()
