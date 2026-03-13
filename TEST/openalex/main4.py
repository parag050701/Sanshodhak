import requests
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    return MODEL.encode(text, convert_to_numpy=True)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------------------------------------
# Helper – OpenAlex inverted index → normal text
# ------------------------------------------------------------
def invert_abstract(index_dict: Dict):
    if not index_dict:
        return ""

    words = []
    for word, positions in index_dict.items():
        for pos in positions:
            if pos >= len(words):
                words.extend([""] * (pos - len(words) + 1))
            words[pos] = word

    return " ".join(words)


# ------------------------------------------------------------
# Fixed OpenAlex Adapter
# ------------------------------------------------------------
class OpenAlexAdapter:
    BASE_URL = "https://api.openalex.org/works"

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Correct OpenAlex search:
        1. Full-text search → search=...
        2. Title-only fallback → filter=title.search:...
        """

        # Primary search (best match)
        params = {
            "search": query,
            "per-page": limit,
            "sort": "relevance_score:desc"
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json().get("results", [])
        except Exception as e:
            print("[ERROR] OpenAlex request failed:", e)
            return []

        # Fallback if empty
        if len(data) == 0:
            print("⚠️ Primary OpenAlex search empty. Trying title.search fallback...")
            params = {
                "filter": f"title.search:{query}",
                "per-page": limit,
                "sort": "relevance_score:desc"
            }
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json().get("results", [])
            except:
                return []

        papers = []
        for item in data:
            title = item.get("title")

            # Decode abstract
            abstract_raw = item.get("abstract_inverted_index")
            abstract = invert_abstract(abstract_raw)

            # Authors
            authors = [
                a.get("author", {}).get("display_name")
                for a in item.get("authorships", [])
            ]

            # Year
            year = item.get("publication_year")

            # DOI
            doi = item.get("doi")

            # PDF URL (if Open Access)
            pdf_url = None
            oa_data = item.get("open_access", {})
            if oa_data.get("is_oa"):
                pdf_url = oa_data.get("oa_url")

            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "doi": doi,
                "pdf_url": pdf_url,
                "page_url": item.get("id"),
                "source": "openalex"
            })

        return papers


# ------------------------------------------------------------
# Filtering
# ------------------------------------------------------------
def filter_papers(papers, min_year=2018):
    return [p for p in papers if p["year"] and p["year"] >= min_year]


# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------
def rank(query, papers):
    q_emb = embed(query)
    ranked = []
    for p in papers:
        text = (p["title"] or "") + " " + (p["abstract"] or "")[:300]
        sim = cosine(q_emb, embed(text))
        ranked.append((sim, p))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked


# ------------------------------------------------------------
# Pretty Print
# ------------------------------------------------------------
def print_paper(p, score=None):
    print("------------------------------------------------")
    if score is not None:
        print("Score:", round(score, 4))
    print("Title:", p["title"])
    print("Authors:", p["authors"])
    print("Year:", p["year"])
    print("DOI:", p["doi"])
    print("PDF:", p["pdf_url"])
    print("------------------------------------------------")


# ------------------------------------------------------------
# MAIN TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    query = "vision transformer"

    adapter = OpenAlexAdapter()
    print("\nSearching OpenAlex...\n")

    papers = adapter.search(query, limit=25)
    print("\nRaw Results:", len(papers))

    papers = filter_papers(papers, min_year=2019)
    print("Filtered:", len(papers))

    ranked = rank(query, papers)

    print("\nTop Papers:\n")
    for score, p in ranked[:5]:
        print_paper(p, score)
