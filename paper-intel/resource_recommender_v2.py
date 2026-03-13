"""
Enhanced Resource Recommendation System with LLM-powered search
Recommends Papers with Code, GitHub repos, HuggingFace models, and web resources
"""

import json
import requests
import os
import re
import time
from typing import List, Dict, Optional, Any
from urllib.parse import quote
from pathlib import Path
from datetime import datetime


class EnhancedResourceRecommender:
    """Recommends code, models, and resources with LLM-powered keyword generation."""
    
    def __init__(self, rag_system=None, llm_api="ollama", openrouter_key=None):
        """
        Args:
            rag_system: RAG system for paper search
            llm_api: "ollama" or "openrouter" 
            openrouter_key: OpenRouter API key (optional)
        """
        self.rag = rag_system
        self.llm_api = llm_api
        self.openrouter_key = openrouter_key or os.getenv('OPENROUTER_API_KEY')
        
        # Session for API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Assistant Bot)'
        })
        
        # Tech keywords for fallback extraction
        self.tech_keywords = {
            'models': ['transformer', 'bert', 'gpt', 'llama', 'mistral', 'roberta', 
                      'vit', 'resnet', 't5', 'bart', 'clip', 'whisper', 'stable-diffusion'],
            'frameworks': ['pytorch', 'tensorflow', 'jax', 'keras', 'transformers',
                          'langchain', 'llamaindex', 'haystack', 'sentence-transformers'],
            'tasks': ['classification', 'generation', 'retrieval', 'embedding',
                     'question-answering', 'summarization', 'translation', 'rag',
                     'text-generation', 'feature-extraction'],
        }
    
    def generate_search_keywords(self, query: str) -> Dict[str, List[str]]:
        """Use LLM to generate optimized search keywords for different platforms."""
        
        prompt = f"""Given this research query: "{query}"

Generate specific search keywords for finding relevant resources.

For HuggingFace: List model names, architectures, and tasks (e.g., "bert-base", "text-classification")
For GitHub: List library names, frameworks, and tools (e.g., "langchain", "transformers")
For Web: List tutorial topics and guides (e.g., "RAG tutorial", "transformer guide")

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "huggingface": ["keyword1", "keyword2", "keyword3"],
  "github": ["keyword1", "keyword2", "keyword3"],
  "web": ["keyword1", "keyword2"]
}}"""

        try:
            # Try OpenRouter first
            if self.llm_api == "openrouter" and self.openrouter_key:
                print("  🤖 Using OpenRouter for keyword generation...")
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "meta-llama/llama-3.1-8b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    content = self._extract_json(content)
                    keywords = json.loads(content)
                    print(f"  ✅ Generated keywords via OpenRouter")
                    return keywords
                else:
                    print(f"  ⚠️ OpenRouter failed: {response.status_code}")
            
            # Fallback to Ollama
            print("  🤖 Using Ollama for keyword generation...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.json()['response']
                content = self._extract_json(content)
                keywords = json.loads(content)
                print(f"  ✅ Generated keywords via Ollama")
                return keywords
                
        except Exception as e:
            print(f"  ⚠️ LLM keyword generation failed: {e}")
        
        # Ultimate fallback: basic keyword extraction
        print("  ⚠️ Using fallback keyword extraction")
        words = [w.strip() for w in query.lower().split() if len(w.strip()) > 3]
        return {
            'huggingface': words[:3],
            'github': words[:3],
            'web': [query]
        }
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from markdown code blocks or mixed text."""
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        return text.strip()
    
    def extract_technical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract models, frameworks, and tasks from text."""
        text_lower = text.lower()
        
        extracted = {
            'models': [],
            'frameworks': [],
            'tasks': []
        }
        
        for category, terms in self.tech_keywords.items():
            for term in terms:
                if term in text_lower:
                    extracted[category].append(term)
        
        return {k: list(set(v)) for k, v in extracted.items()}
    
    @staticmethod
    def _hf_safe_terms(keywords: List[str], query: str) -> List[str]:
        """Return short HuggingFace-compatible search terms (≤3 words each)."""
        # HuggingFace API works best with single or hyphenated terms
        terms = []
        for kw in (keywords or []):
            # Take only the first meaningful word from multi-word phrases
            parts = kw.strip().split()
            if parts:
                terms.append(parts[0].lower().rstrip(',.'))
        # Always add top query words as fallback
        for w in query.lower().split():
            if len(w) > 3 and w not in terms:
                terms.append(w)
        # Deduplicate while preserving order, skip trivial words
        _skip = {'the', 'and', 'for', 'with', 'using', 'based', 'model', 'from'}
        seen, clean = set(), []
        for t in terms:
            t = t.strip('-_.,')
            if t and t not in seen and t not in _skip:
                seen.add(t)
                clean.append(t)
        return clean[:5]  # max 5 search terms

    def search_huggingface(self, query: str, keywords: List[str] = None, limit: int = 10) -> Dict[str, List[Dict]]:
        """Search HuggingFace with improved API handling."""

        results = {'models': [], 'datasets': []}
        search_terms = self._hf_safe_terms(keywords or [], query)
        
        print(f"  🔍 Searching HuggingFace with {len(search_terms)} keywords...")
        
        for term in search_terms[:3]:  # Limit to 3 keywords
            try:
                # Search models
                url = f"https://huggingface.co/api/models?search={quote(term)}&limit={limit}"
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    models = response.json()
                    if isinstance(models, list):
                        for model in models:
                            if len(results['models']) >= limit:
                                break
                            
                            # Handle different ID field names
                            model_id = model.get('id') or model.get('modelId') or model.get('_id')
                            if not model_id:
                                continue
                            
                            # Skip if already added
                            if any(m['name'] == model_id for m in results['models']):
                                continue
                            
                            results['models'].append({
                                'name': model_id,
                                'task': model.get('pipeline_tag') or model.get('task') or 'unknown',
                                'downloads': model.get('downloads', 0),
                                'likes': model.get('likes', 0),
                                'library': model.get('library_name', 'transformers'),
                                'tags': model.get('tags', [])[:5],
                                'url': f"https://huggingface.co/{model_id}"
                            })
                        
                        print(f"    ✓ Found {len(results['models'])} models for '{term}'")
                
                time.sleep(0.5)  # Rate limiting
                
                # Search datasets
                url = f"https://huggingface.co/api/datasets?search={quote(term)}&limit={limit}"
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    datasets = response.json()
                    if isinstance(datasets, list):
                        for dataset in datasets:
                            if len(results['datasets']) >= limit:
                                break
                            
                            dataset_id = dataset.get('id') or dataset.get('_id')
                            if not dataset_id:
                                continue
                            
                            if any(d['name'] == dataset_id for d in results['datasets']):
                                continue
                            
                            results['datasets'].append({
                                'name': dataset_id,
                                'downloads': dataset.get('downloads', 0),
                                'likes': dataset.get('likes', 0),
                                'tags': dataset.get('tags', [])[:5],
                                'url': f"https://huggingface.co/datasets/{dataset_id}"
                            })
                        
                        print(f"    ✓ Found {len(results['datasets'])} datasets for '{term}'")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ✗ Error for '{term}': {e}")
                continue
        
        # Sort by popularity
        results['models'].sort(key=lambda x: (x['likes'] + x['downloads']/1000), reverse=True)
        results['datasets'].sort(key=lambda x: (x['likes'] + x['downloads']/1000), reverse=True)
        
        return results
    
    def search_github(self, query: str, keywords: List[str] = None, limit: int = 10) -> List[Dict]:
        """Search GitHub repositories."""
        
        repos = []
        search_terms = keywords if keywords else [query]
        
        print(f"  🔍 Searching GitHub with {len(search_terms)} keywords...")
        
        for term in search_terms[:2]:  # Limit to 2 keywords
            try:
                url = f"https://api.github.com/search/repositories?q={quote(term)}&sort=stars&order=desc&per_page={limit}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    for repo in data.get('items', []):
                        if len(repos) >= limit:
                            break
                        
                        repo_url = repo['html_url']
                        if any(r['url'] == repo_url for r in repos):
                            continue
                        
                        repos.append({
                            'name': repo['full_name'],
                            'url': repo_url,
                            'description': repo.get('description', 'No description'),
                            'stars': repo['stargazers_count'],
                            'language': repo.get('language', 'Unknown'),
                            'topics': repo.get('topics', [])[:5]
                        })
                    
                    print(f"    ✓ Found {len(repos)} repos for '{term}'")
                
                time.sleep(1)  # GitHub rate limiting
                
            except Exception as e:
                print(f"    ✗ Error for '{term}': {e}")
                continue
        
        repos.sort(key=lambda x: x['stars'], reverse=True)
        return repos[:limit]
    
    def search_web(self, query: str, keywords: List[str] = None, limit: int = 5) -> List[Dict]:
        """Search web for tutorials and guides."""
        
        results = []
        search_terms = keywords if keywords else [query]
        
        print(f"  🔍 Searching web with {len(search_terms)} keywords...")
        
        # Add curated resources first
        curated = self._get_curated_resources(query)
        results.extend(curated)
        
        # Try DuckDuckGo API
        for term in search_terms[:2]:
            try:
                url = f"https://api.duckduckgo.com/?q={quote(term + ' tutorial guide')}&format=json"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for topic in data.get('RelatedTopics', [])[:5]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append({
                                'title': topic.get('Text', '')[:100],
                                'url': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', '')[:200],
                                'source': 'DuckDuckGo'
                            })
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ✗ Error for '{term}': {e}")
        
        return results[:limit]
    
    def _get_curated_resources(self, query: str) -> List[Dict]:
        """Get curated high-quality resources based on query."""
        
        resources = []
        q = query.lower()
        
        # RAG resources
        if any(term in q for term in ['rag', 'retrieval', 'augmented']):
            resources.extend([
                {
                    'title': 'LangChain RAG Tutorial - Building RAG Applications',
                    'url': 'https://python.langchain.com/docs/tutorials/rag/',
                    'snippet': 'Official LangChain tutorial for building RAG applications',
                    'source': 'Curated'
                },
                {
                    'title': 'LlamaIndex RAG Guide - Getting Started',
                    'url': 'https://docs.llamaindex.ai/en/stable/',
                    'snippet': 'Comprehensive guide to RAG with LlamaIndex',
                    'source': 'Curated'
                },
                {
                    'title': 'Pinecone RAG Guide',
                    'url': 'https://www.pinecone.io/learn/retrieval-augmented-generation/',
                    'snippet': 'Understanding Retrieval-Augmented Generation',
                    'source': 'Curated'
                }
            ])
        
        # Transformers/LLMs
        if any(term in q for term in ['transformer', 'bert', 'gpt', 'llm', 'language model']):
            resources.extend([
                {
                    'title': 'HuggingFace NLP Course',
                    'url': 'https://huggingface.co/learn/nlp-course/',
                    'snippet': 'Free course on transformers and NLP',
                    'source': 'Curated'
                },
                {
                    'title': 'The Illustrated Transformer',
                    'url': 'https://jalammar.github.io/illustrated-transformer/',
                    'snippet': 'Visual guide to understanding transformers',
                    'source': 'Curated'
                },
                {
                    'title': 'Attention Is All You Need (Paper)',
                    'url': 'https://arxiv.org/abs/1706.03762',
                    'snippet': 'Original transformer architecture paper',
                    'source': 'Curated'
                }
            ])
        
        # Embeddings/Search
        if any(term in q for term in ['embedding', 'vector', 'semantic', 'search']):
            resources.extend([
                {
                    'title': 'Sentence Transformers Documentation',
                    'url': 'https://www.sbert.net/',
                    'snippet': 'Sentence embeddings using transformers',
                    'source': 'Curated'
                },
                {
                    'title': 'Weaviate Vector Search Guide',
                    'url': 'https://weaviate.io/developers/weaviate',
                    'snippet': 'Building vector search applications',
                    'source': 'Curated'
                }
            ])
        
        return resources
    
    def search_papers_with_code(self, query: str) -> List[Dict]:
        """Search Papers with Code via their official API."""
        try:
            # Official PwC search API
            url = f"https://paperswithcode.com/api/v1/papers/?q={quote(query)}&ordering=-github_link"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get('results', [])[:5]:
                    results.append({
                        'title': item.get('title', 'Unknown'),
                        'paper_url': item.get('url_pdf') or item.get('url_abs', ''),
                        'arxiv_id': item.get('arxiv_id', ''),
                        'github_url': item.get('repository', {}).get('url', '') if item.get('repository') else '',
                        'stars': item.get('repository', {}).get('stars', 0) if item.get('repository') else 0,
                    })
                return results

            # Fallback: search methods endpoint
            url2 = f"https://paperswithcode.com/api/v1/methods/?q={quote(query)}"
            response2 = self.session.get(url2, timeout=10)
            if response2.status_code == 200:
                data2 = response2.json()
                return [
                    {'title': item.get('name', ''), 'paper_url': item.get('paper', ''), 'github_url': ''}
                    for item in data2.get('results', [])[:5]
                ]
        except Exception as e:
            logger.debug(f"Papers with Code search failed: {e}") if 'logger' in dir() else None

        return []
    
    def recommend_resources(self, query: str, top_k_papers: int = 3) -> Dict[str, Any]:
        """Main recommendation pipeline with LLM-powered search."""
        
        print("\n" + "="*80)
        print("🔍 RESOURCE RECOMMENDATION")
        print("="*80)
        print(f"\n📝 Query: {query}")
        
        # Step 1: Search papers
        print("\n📚 Step 1: Searching research papers...")
        relevant_papers = []
        if self.rag:
            try:
                search_results = self.rag.search(query, top_k=top_k_papers)
                for i, result in enumerate(search_results, 1):
                    relevant_papers.append({
                        'rank': i,
                        'file': result['file'],
                        'score': result['score'],
                        'snippet': result['text'][:200]
                    })
                    print(f"  [{i}] {result['file']} (score: {result['score']:.3f})")
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
        
        # Step 2: Generate keywords with LLM
        print("\n🤖 Step 2: Generating optimized search keywords...")
        search_keywords = self.generate_search_keywords(query)
        print(f"  HuggingFace: {', '.join(search_keywords.get('huggingface', [])[:3])}")
        print(f"  GitHub: {', '.join(search_keywords.get('github', [])[:3])}")
        print(f"  Web: {', '.join(search_keywords.get('web', [])[:2])}")
        
        # Step 3: Extract technical terms
        print("\n🔧 Step 3: Extracting technical terms...")
        all_text = query + " " + " ".join([p['snippet'] for p in relevant_papers])
        technical_terms = self.extract_technical_terms(all_text)
        for category, terms in technical_terms.items():
            if terms:
                print(f"  {category.capitalize()}: {', '.join(terms[:5])}")
        
        # Step 4: Search HuggingFace
        print("\n🤗 Step 4: Searching HuggingFace...")
        hf_results = self.search_huggingface(
            query, 
            keywords=search_keywords.get('huggingface'),
            limit=10
        )
        print(f"  ✅ Total: {len(hf_results['models'])} models, {len(hf_results['datasets'])} datasets")
        
        # Step 5: Search GitHub
        print("\n💻 Step 5: Searching GitHub...")
        github_repos = self.search_github(
            query,
            keywords=search_keywords.get('github'),
            limit=10
        )
        print(f"  ✅ Total: {len(github_repos)} repositories")
        
        # Step 6: Search web
        print("\n🌐 Step 6: Searching web resources...")
        web_results = self.search_web(
            query,
            keywords=search_keywords.get('web'),
            limit=10
        )
        print(f"  ✅ Total: {len(web_results)} web resources")
        
        # Step 7: Papers with Code
        print("\n📄 Step 7: Searching Papers with Code...")
        pwc_results = self.search_papers_with_code(query)
        print(f"  ✅ Total: {len(pwc_results)} papers")
        
        return {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'search_keywords': search_keywords,
            'relevant_papers': relevant_papers,
            'technical_terms': technical_terms,
            'huggingface': hf_results,
            'github_repositories': github_repos,
            'web_resources': web_results,
            'papers_with_code': pwc_results,
            'summary': {
                'total_papers': len(relevant_papers),
                'total_hf_models': len(hf_results['models']),
                'total_hf_datasets': len(hf_results['datasets']),
                'total_github_repos': len(github_repos),
                'total_web_resources': len(web_results),
                'total_pwc_papers': len(pwc_results)
            }
        }
    
    def print_recommendations(self, recs: Dict[str, Any]):
        """Print recommendations in formatted output."""
        
        print("\n" + "="*80)
        print("📋 RECOMMENDATIONS SUMMARY")
        print("="*80)
        
        s = recs['summary']
        print(f"\n📊 Statistics:")
        print(f"  • Research Papers: {s['total_papers']}")
        print(f"  • HuggingFace Models: {s['total_hf_models']}")
        print(f"  • HuggingFace Datasets: {s['total_hf_datasets']}")
        print(f"  • GitHub Repos: {s['total_github_repos']}")
        print(f"  • Web Resources: {s['total_web_resources']}")
        print(f"  • Papers with Code: {s['total_pwc_papers']}")
        
        # Top HuggingFace models
        if recs['huggingface']['models']:
            print(f"\n🤗 Top HuggingFace Models:")
            for model in recs['huggingface']['models'][:5]:
                print(f"  • {model['name']}")
                print(f"    Task: {model['task']} | 💙 {model['likes']} | ⬇️ {model['downloads']:,}")
                print(f"    {model['url']}")
        
        # Top GitHub repos
        if recs['github_repositories']:
            print(f"\n⭐ Top GitHub Repositories:")
            for repo in recs['github_repositories'][:5]:
                print(f"  • {repo['name']} ({repo['stars']:,}⭐)")
                print(f"    {repo['description'][:100]}")
                print(f"    {repo['url']}")
        
        # Web resources
        if recs['web_resources']:
            print(f"\n🌐 Top Web Resources:")
            for resource in recs['web_resources'][:5]:
                print(f"  • {resource['title']}")
                print(f"    {resource['url']}")
    
    def save_recommendations(self, recs: Dict[str, Any], filename: str):
        """Save recommendations to JSON file."""
        with open(filename, 'w') as f:
            json.dump(recs, f, indent=2)
        print(f"\n💾 Saved to {filename}")


if __name__ == "__main__":
    from enhanced_rag import EnhancedRAG

    print("Loading RAG system...")
    rag = EnhancedRAG(use_ollama_fallback=True)
    rag.load('rag_index')
    
    print("Creating recommender...")
    recommender = EnhancedResourceRecommender(rag, llm_api="ollama")
    
    # Test query
    query = "Retrieval-Augmented Generation with transformers and vector databases"
    recs = recommender.recommend_resources(query, top_k_papers=3)
    
    recommender.print_recommendations(recs)
    recommender.save_recommendations(recs, "test_recommendations.json")
