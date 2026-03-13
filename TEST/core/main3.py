import requests
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# PUT YOUR API KEY HERE
# ------------------------------------------------------------
CORE_API_KEY = "fU7PRKOIE5Nz3roXDSvCwVZ46Wd2TpYi"


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return MODEL.encode(text, convert_to_numpy=True)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ------------------------------------------------------------
# CORE WORKS SEARCH API  (THIS ONE WORKS)
# ------------------------------------------------------------
class CoreSearch:

    URL = "https://api.core.ac.uk/v3/search/works"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or CORE_API_KEY
        if not self.api_key:
            raise ValueError("❌ CORE API key missing.")

    def search(self, query: str, limit: int = 10) -> List[Dict]:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "limit": limit,
        }

        try:
            resp = requests.post(self.URL, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print("[ERROR] CORE works search failed:", e)
            print("Response text:", getattr(e.response, "text", "No body"))
            return []

        data = resp.json()

        # API returns "results" inside "results"
        results = data.get("results", [])

        papers = []
        for item in results:
            papers.append({
                "title": item.get("title"),
                "abstract": item.get("abstract") or "",
                "authors": item.get("authors", []),
                "year": item.get("yearPublished"),
                "doi": item.get("doi"),
                "pdf_url": item.get("downloadUrl"),
                "source": "core"
            })

        return papers


    # ------------------------------------------------------------
    def download_pdf(self, paper: Dict):
        url = paper.get("pdf_url")
        if not url:
            print("❌ No PDF URL.")
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
    if score:
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

    core = CoreSearch()

    print("\nSearching CORE...\n")
    papers = core.search(query, limit=20)

    print("\nRaw Results:", len(papers))
    papers = filter_papers(papers)
    print("Filtered:", len(papers))

    ranked = rank(query, papers)

    print("\nTop Papers:\n")
    for score, p in ranked[:5]:
        print_paper(p, score)

    # Download top PDF
    if ranked:
        pdf = core.download_pdf(ranked[0][1])
        if pdf:
            with open("core_top.pdf", "wb") as f:
                f.write(pdf)
            print("\nSaved core_top.pdf")
