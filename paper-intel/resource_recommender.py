"""
Resource Recommendation System for Research Papers
Recommends Papers with Code, GitHub repos, HuggingFace models, and other resources
based on user queries and paper content.
"""

import json
import requests
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import re
from datetime import datetime


class ResourceRecommender:
    """Recommends code, models, and implementation resources for research papers."""
    
    def __init__(self, rag_system):
        self.rag = rag_system
        
        # API endpoints
        self.paperswithcode_api = "https://paperswithcode.com/api/v1"
        self.github_api = "https://api.github.com"
        self.huggingface_api = "https://huggingface.co/api"
        
        # Common ML/AI keywords for extraction
        self.tech_keywords = {
            'models': ['transformer', 'bert', 'gpt', 'llama', 'mistral', 'roberta', 
                      'vit', 'resnet', 'unet', 'vae', 'gan', 'diffusion'],
            'frameworks': ['pytorch', 'tensorflow', 'jax', 'keras', 'scikit-learn',
                          'transformers', 'langchain', 'llamaindex'],
            'tasks': ['classification', 'generation', 'retrieval', 'embedding',
                     'question answering', 'summarization', 'translation', 'rag'],
            'techniques': ['fine-tuning', 'prompting', 'few-shot', 'zero-shot',
                          'attention', 'self-attention', 'cross-attention']
        }
    
    def extract_technical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract models, frameworks, and techniques from text."""
        text_lower = text.lower()
        
        extracted = {
            'models': [],
            'frameworks': [],
            'tasks': [],
            'techniques': []
        }
        
        for category, terms in self.tech_keywords.items():
            for term in terms:
                if term in text_lower:
                    extracted[category].append(term)
        
        # Extract version numbers (e.g., GPT-4, BERT-base)
        versions = re.findall(r'\b([A-Z][a-z]+[-_]?\w*[-_]?\d+[\w]*)\b', text)
        extracted['models'].extend(versions)
        
        # Remove duplicates
        for key in extracted:
            extracted[key] = list(set(extracted[key]))
        
        return extracted
    
    def search_papers_with_code(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Papers with Code for implementations."""
        print(f"  🔍 Searching Papers with Code...")
        
        results = []
        
        try:
            # Search papers
            response = requests.get(
                f"{self.paperswithcode_api}/papers/",
                params={"q": query, "items_per_page": limit},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for paper in data.get('results', [])[:limit]:
                    paper_id = paper.get('id')
                    
                    # Get implementations for this paper
                    impl_response = requests.get(
                        f"{self.paperswithcode_api}/papers/{paper_id}/repositories/",
                        timeout=10
                    )
                    
                    implementations = []
                    if impl_response.status_code == 200:
                        impl_data = impl_response.json()
                        implementations = [
                            {
                                'url': repo.get('url'),
                                'framework': repo.get('framework'),
                                'stars': repo.get('stars', 0)
                            }
                            for repo in impl_data.get('results', [])[:3]
                        ]
                    
                    results.append({
                        'title': paper.get('title'),
                        'paper_url': paper.get('url_abs'),
                        'arxiv_id': paper.get('arxiv_id'),
                        'implementations': implementations,
                        'source': 'Papers with Code'
                    })
        
        except Exception as e:
            print(f"    ⚠️  Papers with Code error: {e}")
        
        return results
    
    def search_github(self, query: str, language: str = None, limit: int = 5) -> List[Dict]:
        """Search GitHub for relevant repositories."""
        print(f"  🔍 Searching GitHub...")
        
        results = []
        
        try:
            # Build search query
            search_query = query
            if language:
                search_query += f" language:{language}"
            
            # Add ML/AI related terms
            search_query += " machine learning OR deep learning OR AI"
            
            response = requests.get(
                f"{self.github_api}/search/repositories",
                params={
                    "q": search_query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": limit
                },
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for repo in data.get('items', [])[:limit]:
                    results.append({
                        'name': repo.get('full_name'),
                        'description': repo.get('description', 'No description'),
                        'url': repo.get('html_url'),
                        'stars': repo.get('stargazers_count', 0),
                        'language': repo.get('language'),
                        'topics': repo.get('topics', []),
                        'last_updated': repo.get('updated_at'),
                        'source': 'GitHub'
                    })
        
        except Exception as e:
            print(f"    ⚠️  GitHub error: {e}")
        
        return results
    
    def search_huggingface(self, query: str, task: str = None, limit: int = 5) -> List[Dict]:
        """Search HuggingFace for models and datasets."""
        print(f"  🔍 Searching HuggingFace...")
        
        results = {
            'models': [],
            'datasets': []
        }
        
        try:
            # Search models
            model_params = {"search": query, "limit": limit}
            if task:
                model_params["filter"] = task
            
            response = requests.get(
                f"{self.huggingface_api}/models",
                params=model_params,
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json()
                
                for model in models[:limit]:
                    results['models'].append({
                        'name': model.get('id'),
                        'task': model.get('pipeline_tag'),
                        'downloads': model.get('downloads', 0),
                        'likes': model.get('likes', 0),
                        'url': f"https://huggingface.co/{model.get('id')}",
                        'library': model.get('library_name'),
                        'source': 'HuggingFace Models'
                    })
            
            # Search datasets
            dataset_response = requests.get(
                f"{self.huggingface_api}/datasets",
                params={"search": query, "limit": limit},
                timeout=10
            )
            
            if dataset_response.status_code == 200:
                datasets = dataset_response.json()
                
                for dataset in datasets[:limit]:
                    results['datasets'].append({
                        'name': dataset.get('id'),
                        'downloads': dataset.get('downloads', 0),
                        'likes': dataset.get('likes', 0),
                        'url': f"https://huggingface.co/datasets/{dataset.get('id')}",
                        'source': 'HuggingFace Datasets'
                    })
        
        except Exception as e:
            print(f"    ⚠️  HuggingFace error: {e}")
        
        return results
    
    def search_arxiv_code(self, arxiv_id: str) -> List[Dict]:
        """Search for code associated with ArXiv papers."""
        print(f"  🔍 Searching ArXiv code links...")
        
        results = []
        
        # Check Papers with Code
        try:
            response = requests.get(
                f"https://paperswithcode.com/api/v1/papers/{arxiv_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'official' in data and data['official']:
                    results.append({
                        'type': 'Official Implementation',
                        'url': data['official'].get('url'),
                        'framework': data['official'].get('framework'),
                        'source': 'Papers with Code'
                    })
        
        except Exception as e:
            print(f"    ⚠️  ArXiv code search error: {e}")
        
        return results
    
    def search_web(self, query: str, keywords: List[str] = None, num_results: int = 5) -> List[Dict]:
        """Search the web for tutorials, guides, and resources.
        
        Args:
            query: Search query
            keywords: Specific keywords to search
            num_results: Number of results to return
            
        Returns:
            List of web resources
        """
        results = []
        search_queries = keywords if keywords else [query]
        
        # Use DuckDuckGo HTML search (no API key needed)
        for search_term in search_queries[:2]:  # Limit to 2 queries
            try:
                # DuckDuckGo Instant Answer API
                url = f"https://api.duckduckgo.com/?q={quote(search_term + ' tutorial guide')}&format=json"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Related topics
                    for topic in data.get('RelatedTopics', [])[:num_results]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append({
                                'title': topic.get('Text', '')[:100],
                                'url': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', '')[:200],
                                'source': 'DuckDuckGo'
                            })
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ⚠️  Web search error for '{search_term}': {e}")
        
        # Add curated resources based on query terms
        curated = self._get_curated_resources(query)
        results.extend(curated)
        
        return results[:num_results]
    
    def _get_curated_resources(self, query: str) -> List[Dict]:
        """Get curated resources based on query keywords."""
        resources = []
        query_lower = query.lower()
        
        # RAG resources
        if any(term in query_lower for term in ['rag', 'retrieval', 'augmented']):
            resources.extend([
                {
                    'title': 'LangChain RAG Tutorial',
                    'url': 'https://python.langchain.com/docs/tutorials/rag/',
                    'snippet': 'Build RAG applications with LangChain',
                    'source': 'Curated'
                },
                {
                    'title': 'LlamaIndex RAG Guide',
                    'url': 'https://docs.llamaindex.ai/en/stable/getting_started/starter_example/',
                    'snippet': 'Getting started with RAG using LlamaIndex',
                    'source': 'Curated'
                },
                {
                    'title': 'Pinecone RAG Guide',
                    'url': 'https://www.pinecone.io/learn/retrieval-augmented-generation/',
                    'snippet': 'Understanding and implementing RAG systems',
                    'source': 'Curated'
                }
            ])
        
        # Transformers/NLP
        if any(term in query_lower for term in ['transformer', 'bert', 'gpt', 'llm', 'language model']):
            resources.extend([
                {
                    'title': 'HuggingFace Transformers Course',
                    'url': 'https://huggingface.co/learn/nlp-course/',
                    'snippet': 'Free course on transformers and NLP',
                    'source': 'Curated'
                },
                {
                    'title': 'The Illustrated Transformer',
                    'url': 'https://jalammar.github.io/illustrated-transformer/',
                    'snippet': 'Visual guide to understanding transformers',
                    'source': 'Curated'
                }
            ])
        
        # Embeddings/Search
        if any(term in query_lower for term in ['embedding', 'vector', 'semantic search']):
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
    
    def recommend_resources(self, query: str, top_k_papers: int = 5) -> Dict[str, Any]:
        """
        Main recommendation function.
        1. Search papers in RAG system
        2. Extract technical terms
        3. Search for implementations and resources
        """
        
        print(f"\n{'='*80}")
        print(f"🔍 RESOURCE RECOMMENDATION")
        print(f"{'='*80}")
        print(f"\n📝 Query: {query}")
        
        # Step 1: Search papers
        print(f"\n📚 Step 1: Searching research papers...")
        try:
            paper_results = self.rag.search(query, top_k=top_k_papers)
        except TypeError:
            # Fallback for simple API
            paper_results = self.rag.search(query, top_k_papers)
        
        relevant_papers = []
        all_technical_terms = {
            'models': set(),
            'frameworks': set(),
            'tasks': set(),
            'techniques': set()
        }
        
        for i, result in enumerate(paper_results):
            paper_info = {
                'rank': i + 1,
                'file': result['metadata']['file'],
                'score': result['score'],
                'snippet': result['text'][:300]
            }
            relevant_papers.append(paper_info)
            
            # Extract technical terms
            terms = self.extract_technical_terms(result['text'])
            for category, items in terms.items():
                all_technical_terms[category].update(items)
            
            print(f"  [{i+1}] {result['metadata']['file']} (score: {result['score']:.3f})")
        
        # Convert sets to lists
        for category in all_technical_terms:
            all_technical_terms[category] = list(all_technical_terms[category])
        
        print(f"\n🔧 Extracted technical terms:")
        for category, terms in all_technical_terms.items():
            if terms:
                print(f"  {category.capitalize()}: {', '.join(terms[:5])}")
        
        # Step 2: Search Papers with Code
        print(f"\n📄 Step 2: Searching Papers with Code...")
        pwc_results = self.search_papers_with_code(query, limit=5)
        print(f"  ✅ Found {len(pwc_results)} papers with code")
        
        # Step 3: Search GitHub
        print(f"\n💻 Step 3: Searching GitHub repositories...")
        github_results = self.search_github(query, limit=5)
        print(f"  ✅ Found {len(github_results)} repositories")
        
        # Step 4: Search HuggingFace
        print(f"\n🤗 Step 4: Searching HuggingFace...")
        task = all_technical_terms['tasks'][0] if all_technical_terms['tasks'] else None
        hf_results = self.search_huggingface(query, task=task, limit=5)
        print(f"  ✅ Found {len(hf_results['models'])} models, {len(hf_results['datasets'])} datasets")
        
        # Step 5: Additional resources
        print(f"\n📖 Step 5: Gathering additional resources...")
        
        additional_resources = {
            'tutorials': self._get_tutorial_links(query, all_technical_terms),
            'documentation': self._get_documentation_links(all_technical_terms),
            'courses': self._get_course_links(all_technical_terms)
        }
        
        # Compile final recommendations
        recommendations = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'relevant_papers': relevant_papers,
            'technical_terms': all_technical_terms,
            'papers_with_code': pwc_results,
            'github_repositories': github_results,
            'huggingface': hf_results,
            'additional_resources': additional_resources,
            'summary': {
                'total_papers': len(relevant_papers),
                'total_github_repos': len(github_results),
                'total_hf_models': len(hf_results['models']),
                'total_hf_datasets': len(hf_results['datasets']),
                'pwc_papers': len(pwc_results)
            }
        }
        
        return recommendations
    
    def _get_tutorial_links(self, query: str, terms: Dict) -> List[Dict]:
        """Generate tutorial resource links."""
        tutorials = []
        
        # Add general ML/AI tutorials based on detected frameworks
        if 'pytorch' in terms['frameworks']:
            tutorials.append({
                'title': 'PyTorch Tutorials',
                'url': 'https://pytorch.org/tutorials/',
                'type': 'Official Documentation'
            })
        
        if 'transformers' in terms['frameworks']:
            tutorials.append({
                'title': 'HuggingFace Transformers Course',
                'url': 'https://huggingface.co/learn/nlp-course',
                'type': 'Course'
            })
        
        if any(term in query.lower() for term in ['rag', 'retrieval']):
            tutorials.extend([
                {
                    'title': 'LangChain RAG Tutorial',
                    'url': 'https://python.langchain.com/docs/tutorials/rag/',
                    'type': 'Tutorial'
                },
                {
                    'title': 'LlamaIndex Guides',
                    'url': 'https://docs.llamaindex.ai/',
                    'type': 'Documentation'
                }
            ])
        
        return tutorials
    
    def _get_documentation_links(self, terms: Dict) -> List[Dict]:
        """Generate documentation links for detected frameworks."""
        docs = []
        
        framework_docs = {
            'pytorch': 'https://pytorch.org/docs/stable/index.html',
            'tensorflow': 'https://www.tensorflow.org/api_docs',
            'transformers': 'https://huggingface.co/docs/transformers',
            'langchain': 'https://python.langchain.com/docs/get_started/introduction',
            'llamaindex': 'https://docs.llamaindex.ai/en/stable/',
            'scikit-learn': 'https://scikit-learn.org/stable/documentation.html'
        }
        
        for framework in terms['frameworks']:
            if framework in framework_docs:
                docs.append({
                    'framework': framework,
                    'url': framework_docs[framework],
                    'type': 'Official Documentation'
                })
        
        return docs
    
    def _get_course_links(self, terms: Dict) -> List[Dict]:
        """Recommend relevant courses."""
        courses = []
        
        # General ML courses
        courses.extend([
            {
                'title': 'Fast.ai Practical Deep Learning',
                'url': 'https://course.fast.ai/',
                'type': 'Free Course'
            },
            {
                'title': 'DeepLearning.AI Courses',
                'url': 'https://www.deeplearning.ai/courses/',
                'type': 'Course Platform'
            }
        ])
        
        # Add specific courses based on detected topics
        if any(model in terms['models'] for model in ['gpt', 'bert', 'transformer']):
            courses.append({
                'title': 'Stanford CS224N: NLP with Deep Learning',
                'url': 'https://web.stanford.edu/class/cs224n/',
                'type': 'University Course'
            })
        
        return courses
    
    def save_recommendations(self, recommendations: Dict, output_file: str):
        """Save recommendations to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\n💾 Recommendations saved to {output_file}")
    
    def print_recommendations(self, recommendations: Dict):
        """Pretty print recommendations."""
        print(f"\n{'='*80}")
        print(f"📋 RECOMMENDATIONS SUMMARY")
        print(f"{'='*80}")
        
        summary = recommendations['summary']
        print(f"\n📊 Statistics:")
        print(f"  • Research Papers: {summary['total_papers']}")
        print(f"  • GitHub Repos: {summary['total_github_repos']}")
        print(f"  • HuggingFace Models: {summary['total_hf_models']}")
        print(f"  • HuggingFace Datasets: {summary['total_hf_datasets']}")
        print(f"  • Papers with Code: {summary['pwc_papers']}")
        
        # Top GitHub repos
        if recommendations['github_repositories']:
            print(f"\n⭐ Top GitHub Repositories:")
            for repo in recommendations['github_repositories'][:3]:
                print(f"  • {repo['name']} ({repo['stars']}⭐)")
                print(f"    {repo['url']}")
                print(f"    {repo['description'][:80]}...")
        
        # Top HuggingFace models
        if recommendations['huggingface']['models']:
            print(f"\n🤗 Top HuggingFace Models:")
            for model in recommendations['huggingface']['models'][:3]:
                print(f"  • {model['name']} ({model['likes']}💙)")
                print(f"    Task: {model['task']}")
                print(f"    {model['url']}")
        
        # Tutorials
        if recommendations['additional_resources']['tutorials']:
            print(f"\n📚 Recommended Tutorials:")
            for tutorial in recommendations['additional_resources']['tutorials'][:3]:
                print(f"  • {tutorial['title']}")
                print(f"    {tutorial['url']}")


if __name__ == "__main__":
    from ollama_rag import OllamaRAG
    
    print("="*80)
    print("🚀 RESOURCE RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Load RAG system
    print("\n📦 Loading RAG system...")
    rag = OllamaRAG()
    rag.load("rag_index")
    
    # Create recommender
    recommender = ResourceRecommender(rag)
    
    # Example queries
    queries = [
        "Retrieval-Augmented Generation with knowledge graphs",
        "Query optimization in databases using machine learning",
        "Hybrid search systems with dense and sparse retrieval"
    ]
    
    for query in queries:
        recommendations = recommender.recommend_resources(query, top_k_papers=3)
        recommender.print_recommendations(recommendations)
        
        # Save to file
        filename = f"recommendations_{query[:30].replace(' ', '_')}.json"
        recommender.save_recommendations(recommendations, filename)
        
        print("\n" + "="*80 + "\n")
    
    print("✅ Resource recommendation complete!")
