"""
Rebuild the retrieval test set so each question is grounded in a specific paper
that exists in the current corpus. Eliminates the corpus-test-set mismatch that
caused the prior 100-Q eval to score ~0.066 NDCG@10.

For each sampled paper, pick a mid-document chunk and prompt an LLM to generate
ONE specific factual question + answer. Output schema matches
rag_test_questions_100.json so eval_compare.py can drop it in unchanged.
"""
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Force stdout to UTF-8 with replacement for unprintable chars (Greek letters,
# math symbols, etc. show up in scholarly text and crash cp1252 on Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

NIM_KEY = os.getenv("NVIDIA_NIM_API_KEY")
NIM_BASE = os.getenv("NIM_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
assert NIM_KEY, "NVIDIA_NIM_API_KEY missing from .env"

LLM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
MAX_PAPERS = 100
MIN_CHUNK_CHARS = 800
MAX_CHUNK_CHARS = 2500

INDEX_DIR = SCRIPT_DIR / "rag_index"
META_PATH = INDEX_DIR / "metadata.pkl"
OUT_PATH = SCRIPT_DIR / "rag_test_questions_corpus.json"

random.seed(42)

PROMPT = """You are generating an evaluation question for a scholarly retrieval benchmark.

Read the passage below from a research paper. Output ONE specific factual question that:
1. Can only be answered confidently by reading THIS passage (not a generic textbook fact).
2. Mentions at least one concrete entity, method name, dataset name, metric, or specific number from the passage.
3. Avoids generic openers like "What is X?" or "How does Y work?".

Also produce a 1–2 sentence answer grounded in the passage.

Output strict JSON ONLY (no preamble, no code fences):
{{"question": "...", "answer": "..."}}

PASSAGE:
{passage}
"""

client = OpenAI(api_key=NIM_KEY, base_url=NIM_BASE)


def pick_chunk(chunks_for_paper):
    """Pick a representative mid-document chunk; skip very short ones."""
    candidates = [
        (idx, text) for idx, text in chunks_for_paper
        if MIN_CHUNK_CHARS <= len(text) <= MAX_CHUNK_CHARS
    ]
    if not candidates:
        candidates = chunks_for_paper
    if len(candidates) >= 3:
        return candidates[len(candidates) // 2]
    return candidates[0]


def generate_qa(passage: str, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": PROMPT.format(passage=passage)}],
                temperature=0.4,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON found in: {text[:200]}")
            obj = json.loads(m.group(0))
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not q or not a:
                raise ValueError("empty question or answer")
            return q, a
        except Exception as e:
            if attempt == retries:
                return None, f"gen_failed: {e}"
            time.sleep(2 ** attempt)
    return None, "exhausted"


def main():
    import pickle
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    chunks = meta["chunks"]
    metas = meta["metadata"]

    by_file = defaultdict(list)
    for i, m in enumerate(metas):
        by_file[m["file"]].append((m.get("chunk_index", i), chunks[i]))
    for f in by_file:
        by_file[f].sort()

    files = sorted(by_file.keys())
    random.shuffle(files)
    sampled = files[:MAX_PAPERS]
    print(f"corpus: {len(files)} papers, sampling {len(sampled)}")

    out = []
    skips = 0
    t0 = time.time()
    for i, fname in enumerate(sampled, 1):
        chunk_idx, passage = pick_chunk(by_file[fname])
        q, a = generate_qa(passage)
        if q is None:
            skips += 1
            print(f"[{i:3d}/{len(sampled)}] SKIP {fname}  ({a})")
            continue
        paper_id = fname.replace(".txt", "")
        out.append({
            "id": len(out) + 1,
            "question": q,
            "answer": a,
            "type": "factual",
            "difficulty": "easy",
            "paper_id": paper_id,
            "relevant_docs": [fname],
            "stratum": "corpus-grounded",
            "_source_chunk_index": chunk_idx,
        })
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (len(sampled) - i) / rate if rate > 0 else 0
        print(f"[{i:3d}/{len(sampled)}] OK   {fname[:50]:50s}  Q: {q[:80]}  ({eta:.0f}s left)")
        # Save incrementally so a crash doesn't lose work
        if i % 10 == 0 or i == len(sampled):
            OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    total = time.time() - t0
    print(f"\nWROTE {len(out)} questions ({skips} skipped) in {total:.0f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
