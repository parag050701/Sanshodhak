import os
import requests
import feedparser
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

ARXIV_API = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}"

# --- Step 1: Search arXiv and download PDFs ---
def search_arxiv(query, max_results=5):
    url = ARXIV_API.format(requests.utils.quote(query), max_results)
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        pdf_url = None
        for link in entry.links:
            if link.type == 'application/pdf':
                pdf_url = link.href
                break
        if pdf_url:
            papers.append({
                'title': entry.title,
                'summary': entry.summary,
                'pdf_url': pdf_url
            })
    return papers

def download_pdf(pdf_url, out_dir):
    local_name = pdf_url.split('/')[-1] + '.pdf'
    out_path = Path(out_dir) / local_name
    if out_path.exists():
        return str(out_path)
    r = requests.get(pdf_url, timeout=30)
    if r.status_code == 200:
        with open(out_path, 'wb') as f:
            f.write(r.content)
        return str(out_path)
    return None

# --- Step 2: Extract text (simple, just summary for demo) ---
# For real use, add PDF text extraction here

def main():
    topic = input("Enter arXiv topic: ").strip()
    num_papers = int(input("How many papers? (default 3): ") or 3)
    out_dir = "arxiv_papers"
    Path(out_dir).mkdir(exist_ok=True)
    print(f"\nSearching arXiv for '{topic}'...")
    papers = search_arxiv(topic, num_papers)
    if not papers:
        print("No papers found.")
        return
    print(f"Found {len(papers)} papers. Downloading PDFs...")
    for i, paper in enumerate(papers, 1):
        pdf_path = download_pdf(paper['pdf_url'], out_dir)
        print(f"[{i}] {paper['title'][:80]}\n    PDF: {pdf_path if pdf_path else 'Download failed!'}")
    # --- Step 3: Embed summaries ---
    print("\nEmbedding paper summaries with bge-m3...")
    model = SentenceTransformer("BAAI/bge-m3")
    summaries = [p['summary'] for p in papers]
    embeddings = model.encode(summaries, normalize_embeddings=True)
    # --- Step 4: Simple Q&A ---
    print("\nYou can now search the summaries. Type 'quit' to exit.")
    while True:
        q = input("Your question: ").strip()
        if q.lower() == 'quit':
            break
        q_emb = model.encode([q], normalize_embeddings=True)[0]
        sims = np.dot(embeddings, q_emb)
        idx = int(np.argmax(sims))
        print(f"\nBest match: {papers[idx]['title'][:80]}\nSummary: {papers[idx]['summary'][:500]}\n")

if __name__ == "__main__":
    main()
