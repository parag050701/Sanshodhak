import arxiv
import requests
from datetime import datetime
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
import numpy as np


# ------------------------------------------------------------
# Embedding Model (Fast + Accurate)
# ------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-" \
"L6-v2")


def embed(text: str):
    return model.encode(text, convert_to_numpy=True)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------------------------------------
# Arxiv Adapter
# ------------------------------------------------------------
class ArxivAdapter:
    """Enhanced one-file ArXiv searcher and PDF downloader."""

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        results = []

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for result in search.results():
            year = (
                result.published.year
                if isinstance(result.published, datetime)
                else None
            )

            results.append({
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.strip(),
                "year": year,
                "arxiv_id": result.get_short_id(),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "page_url": result.entry_id,
                "source": "arxiv",
            })

        return results

    # --------------------------------------------------------

    def fetch_pdf(self, paper: Dict) -> Optional[bytes]:
        """Download PDF from paper['pdf_url']."""
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            print("No PDF URL found.")
            return None

        try:
            response = requests.get(pdf_url, timeout=20)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"[ERROR] PDF download error: {e}")
            return None


# ------------------------------------------------------------
# Filtering (Fast + Efficient)
# ------------------------------------------------------------

ALLOWED_CATEGORIES = {"cs.CV", "cs.CL", "cs.AI", "cs.LG", "stat.ML"}


def filter_papers(papers: List[Dict], min_year=2020, keywords=None):
    if keywords is None:
        keywords = []

    filtered = []

    for p in papers:
        # Category filter
        if not any(c in ALLOWED_CATEGORIES for c in p["categories"]):
            continue

        # Year filter
        if p["year"] is not None and p["year"] < min_year:
            continue

        # Keyword filter
        text = (p["title"] + " " + p["abstract"]).lower()
        if keywords:
            if not any(k.lower() in text for k in keywords):
                continue

        filtered.append(p)

    return filtered


# ------------------------------------------------------------
# Embedding-based Ranking
# ------------------------------------------------------------

def paper_text(paper: Dict, mode="balanced"):
    """
    mode options:
    - ultra_fast: only title
    - balanced: title + first 300 chars of abstract
    - accurate: full abstract
    """
    title = paper["title"]
    abstract = paper["abstract"]

    if mode == "ultra_fast":
        return title

    if mode == "balanced":
        return title + " " + abstract[:300]

    return title + " " + abstract


def rank_papers(query: str, papers: List[Dict], mode="balanced"):
    q_emb = embed(query)
    ranked = []

    for p in papers:
        text = paper_text(p, mode=mode)
        sim = cosine_sim(q_emb, embed(text))
        ranked.append((sim, p))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked


# ------------------------------------------------------------
# Pretty Printing
# ------------------------------------------------------------

def print_paper_info(p: Dict, score=None):
    print("------------------------------------------------")
    if score is not None:
        print(f"Score: {round(score, 4)}")

    print("Title:", p["title"])
    print("Authors:", ", ".join(p["authors"]))
    print("Year:", p["year"])
    print("Categories:", p["categories"])
    print("arXiv ID:", p["arxiv_id"])
    print("PDF URL:", p["pdf_url"])
    print("------------------------------------------------\n")


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    adapter = ArxivAdapter()
    query = "vision transformer"

    # Step 1: Search
    papers = adapter.search(query, max_results=25)
    print(f"\nRaw results: {len(papers)} papers\n")

    # Step 2: Filter
    papers = filter_papers(papers, min_year=2020, keywords=["transformer", "vision"])
    print(f"After filtering: {len(papers)} papers\n")

    if not papers:
        print("No papers passed filtering.")
        exit()

    # Step 3: Rank
    ranked = rank_papers(query, papers, mode="balanced")

    print("Top Ranked Papers:\n")
    for score, p in ranked[:5]:
        print_paper_info(p, score)

    # Step 4: Download Top 3 PDFs
    print("\nDownloading Top 3 PDFs...\n")

    for score, p in ranked[:3]:
        pdf = adapter.fetch_pdf(p)
        if not pdf:
            print(f"❌ Failed: {p['title']}\n")
            continue

        filename = f"{p['arxiv_id'].replace('/', '_')}.pdf"
        with open(filename, "wb") as f:
            f.write(pdf)

        print(f"✅ Saved: {filename}\n")
