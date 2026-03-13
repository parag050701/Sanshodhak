import requests
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SEMANTIC_SCHOLAR_API_KEY = "YVVAKNpeyS7URIYZKCcjkam4v7qKpZJo2m2K794w"
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return MODEL.encode(text, convert_to_numpy=True)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------------------------------------
# Semantic Scholar Adapter
# ------------------------------------------------------------
class SemanticScholarAdapter:

    def __init__(self, api_key: str = SEMANTIC_SCHOLAR_API_KEY):
        self.api_key = api_key

    def search(self, query: str, limit: int = 20) -> List[Dict]:

        headers = {
            "x-api-key": self.api_key
        }

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,externalIds,fieldsOfStudy,url,openAccessPdf"
        }

        try:
            resp = requests.get(BASE_URL, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print("[ERROR] Semantic Scholar request failed:", e)
            try:
                print("Response:", resp.text)
            except:
                pass
            return []

        items = data.get("data", [])
        papers = []

        for item in items:

            # Extract DOI if present
            external_ids = item.get("externalIds", {})
            doi = external_ids.get("DOI")

            # OpenAccess PDF link
            pdf_url = None
            if item.get("openAccessPdf"):
                pdf_url = item["openAccessPdf"].get("url")

            # Build paper dict
            papers.append({
                "title": item.get("title"),
                "abstract": item.get("abstract") or "",
                "authors": [a["name"] for a in item.get("authors", [])],
                "year": item.get("year"),
                "doi": doi,
                "pdf_url": pdf_url,
                "page_url": item.get("url"),
                "source": "semantic_scholar"
            })

        return papers

    # ------------------------------------------------------------
    def fetch_pdf(self, paper: Dict):
        """Download open-access PDF if available."""

        url = paper.get("pdf_url")
        if not url:
            print("❌ No PDF available (not open access).")
            return None

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.content
        except Exception as e:
            print("[ERROR] PDF download failed:", e)
            return None


# ------------------------------------------------------------
# Filtering
# ------------------------------------------------------------
def filter_papers(papers, min_year=2018):
    out = []
    for p in papers:
        if p["year"] and p["year"] < min_year:
            continue
        out.append(p)
    return out


# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------
def rank(query, papers):
    q_emb = embed(query)
    ranked = []

    for p in papers:
        text = p["title"] + " " + p["abstract"][:300]
        sim = cosine(q_emb, embed(text))
        ranked.append((sim, p))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked


# ------------------------------------------------------------
# Pretty Output
# ------------------------------------------------------------
def print_paper(p, score=None):
    print("------------------------------------------------")
    if score:
        print("Score:", round(score, 4))
    print("Title:", p["title"])
    print("Authors:", p["authors"])
    print("Year:", p["year"])
    print("DOI:", p["doi"])
    print("PDF:", p["pdf_url"])
    print("Source:", p["source"])
    print("------------------------------------------------")


# ------------------------------------------------------------
# MAIN Test
# ------------------------------------------------------------
if __name__ == "__main__":
    adapter = SemanticScholarAdapter()

    query = "Graph Retreival Augmented Generation"

    print("Searching Semantic Scholar...\n")
    papers = adapter.search(query, limit=20)

    print("Raw Results:", len(papers))
    papers = filter_papers(papers)
    print("Filtered:", len(papers))

    ranked = rank(query, papers)

    print("\nTop Papers:\n")
    for score, p in ranked[:5]:
        print_paper(p, score)

    # Save top PDF if available
    if ranked:
        pdf = adapter.fetch_pdf(ranked[0][1])
        if pdf:
            with open("semantic_scholar_top.pdf", "wb") as f:
                f.write(pdf)
            print("\nSaved: semantic_scholar_top.pdf")
