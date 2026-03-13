import requests
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return MODEL.encode(text, convert_to_numpy=True)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------------------------------------
# DOAJ Adapter (FIXED)
# ------------------------------------------------------------
class DOAJAdapter:
    """
    DOAJ API v3 Search for OA Articles
    Correct endpoint:
        https://doaj.org/api/v3/search/articles?q=YOUR_QUERY
    """

    BASE_URL = "https://doaj.org/api/v3/search/articles"

    def search(self, query: str, limit: int = 10) -> List[Dict]:

        params = {
            "q": query,
            "page": 1,
            "pageSize": limit
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print("[ERROR] DOAJ request failed:", e)
            print("Response text:", getattr(e.response, "text", "No body"))
            return []

        data = resp.json()
        items = data.get("results", [])

        papers = []
        for item in items:
            bib = item.get("bibjson", {})
            title = bib.get("title")
            year = bib.get("year")
            identifier_list = bib.get("identifier", [])

            # Extract DOI
            doi = None
            for ident in identifier_list:
                if ident.get("type") == "doi":
                    doi = ident.get("id")

            # Extract first PDF
            pdf_url = None
            for link in bib.get("link", []):
                if link.get("type") == "fulltext":
                    if link.get("url", "").endswith(".pdf"):
                        pdf_url = link.get("url")

            authors = [a.get("name") for a in bib.get("author", [])]

            papers.append({
                "title": title,
                "abstract": "",
                "authors": authors,
                "year": year,
                "doi": doi,
                "pdf_url": pdf_url,
                "source": "doaj"
            })

        return papers


# ------------------------------------------------------------
# Filter
# ------------------------------------------------------------
def filter_papers(papers, min_year=2018):
    out = []
    for p in papers:
        if p["year"] and p["year"] < min_year:
            continue
        out.append(p)
    return out


# ------------------------------------------------------------
# Rank
# ------------------------------------------------------------
def rank(query, papers):
    q_emb = embed(query)
    ranked = []

    for p in papers:
        text = (p["title"] or "")
        sim = cosine(q_emb, embed(text))
        ranked.append((sim, p))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------
def print_paper(p, score=None):
    print("------------------------------------------------")
    if score:
        print("Score:", round(score, 4))
    print("Title:", p["title"])
    print("Authors:", p["authors"])
    print("Year:", p["year"])
    print("DOI:", p["doi"])
    print("PDF URL:", p["pdf_url"])
    print("Source:", p["source"])
    print("------------------------------------------------\n")


# ------------------------------------------------------------
# MAIN TEST
# ------------------------------------------------------------
if __name__ == "__main__":

    adapter = DOAJAdapter()

    print("\nSearching DOAJ...\n")
    papers = adapter.search("vision transformer", limit=15)

    print("\nRaw Results:", len(papers))
    papers = filter_papers(papers)
    print("Filtered:", len(papers))

    ranked = rank("vision transformer", papers)

    print("\nTop Papers:\n")
    for score, p in ranked[:5]:
        print_paper(p, score)
