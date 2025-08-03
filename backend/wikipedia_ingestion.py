"""
Wikipedia Data Ingestion Service
Handles bulk ingestion of Wikipedia articles with intelligent chunking and processing
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from urllib.parse import quote
import xml.etree.ElementTree as ET
from io import StringIO

from .utils import setup_logging
from .vector_store_chroma import ChromaVectorStoreManager
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = setup_logging(__name__)

@dataclass
class WikipediaArticle:
    title: str
    content: str
    url: str
    categories: List[str]
    page_id: int
    
class WikipediaIngestionService:
    """Service for ingesting Wikipedia articles at scale"""
    
    def __init__(self, vector_store: ChromaVectorStoreManager):
        self.vector_store = vector_store
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-System/1.0 (Educational Purpose)'
        })
        
        # Text splitter for chunking long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Rate limiting
        self.request_delay = 0.1  # 10 requests per second max
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting to respect Wikipedia's servers"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_wikipedia_categories(self, limit: int = 100) -> List[str]:
        """Get list of major Wikipedia categories"""
        try:
            self._rate_limit()
            url = "https://en.wikipedia.org/api/rest_v1/page/random/title"
            
            # Get featured categories instead
            categories_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': 'Category:Main_topic_classifications',
                'cmlimit': limit
            }
            
            response = self.session.get(categories_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            categories = []
            
            if 'query' in data and 'categorymembers' in data['query']:
                for member in data['query']['categorymembers']:
                    if member['title'].startswith('Category:'):
                        categories.append(member['title'])
            
            return categories[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching Wikipedia categories: {e}")
            # Return some default major categories
            return [
                "Category:Science",
                "Category:Technology", 
                "Category:History",
                "Category:Culture",
                "Category:Geography",
                "Category:Biography",
                "Category:Arts",
                "Category:Mathematics",
                "Category:Physics",
                "Category:Computer_science"
            ]
    
    def get_articles_from_category(self, category: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get articles from a specific Wikipedia category"""
        try:
            self._rate_limit()
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmlimit': limit,
                'cmtype': 'page'  # Only get pages, not subcategories
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            if 'query' in data and 'categorymembers' in data['query']:
                for member in data['query']['categorymembers']:
                    if not member['title'].startswith('Category:'):
                        articles.append({
                            'title': member['title'],
                            'page_id': member['pageid']
                        })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching articles from category {category}: {e}")
            return []
    
    def get_random_articles(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get random Wikipedia articles for diverse content"""
        articles = []
        
        try:
            for _ in range(count):
                self._rate_limit()
                
                url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
                response = self.session.get(url)
                response.raise_for_status()
                
                data = response.json()
                articles.append({
                    'title': data.get('title', ''),
                    'page_id': data.get('pageid', 0),
                    'extract': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                })
                
        except Exception as e:
            logger.error(f"Error fetching random articles: {e}")
        
        return articles
    
    def fetch_article_content(self, title: str) -> Optional[WikipediaArticle]:
        """Fetch full content of a Wikipedia article"""
        try:
            self._rate_limit()
            
            # Get article content
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|categories|pageids',
                'exintro': False,  # Get full content
                'explaintext': True,  # Plain text, no HTML
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                page_data = next(iter(pages.values()))
                
                if 'missing' in page_data:
                    logger.warning(f"Article '{title}' not found")
                    return None
                
                content = page_data.get('extract', '')
                if len(content) < 100:  # Skip very short articles
                    return None
                
                categories = []
                if 'categories' in page_data:
                    categories = [cat['title'] for cat in page_data['categories']]
                
                article = WikipediaArticle(
                    title=title,
                    content=content,
                    url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    categories=categories,
                    page_id=page_data.get('pageid', 0)
                )
                
                return article
                
        except Exception as e:
            logger.error(f"Error fetching article '{title}': {e}")
            return None
    
    def process_article_to_documents(self, article: WikipediaArticle) -> List[Document]:
        """Convert Wikipedia article to document chunks for vector storage"""
        try:
            # Split content into chunks
            chunks = self.text_splitter.split_text(article.content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very small chunks
                    continue
                
                metadata = {
                    'source': article.url,
                    'title': article.title,
                    'source_type': 'wikipedia',
                    'page_id': article.page_id,
                    'chunk_index': i,
                    'categories': ', '.join(article.categories[:5])  # Limit categories
                }
                
                doc = Document(
                    page_content=chunk.strip(),
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing article '{article.title}': {e}")
            return []
    
    def ingest_wikipedia_batch(self, article_titles: List[str], batch_name: str = "wikipedia_batch") -> Dict[str, Any]:
        """Ingest a batch of Wikipedia articles"""
        results = {
            'total_articles': len(article_titles),
            'successful': 0,
            'failed': 0,
            'documents_created': 0,
            'failed_articles': []
        }
        
        logger.info(f"Starting ingestion of {len(article_titles)} Wikipedia articles...")
        
        # Process articles concurrently (but limited to respect rate limits)
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            future_to_title = {
                executor.submit(self.fetch_article_content, title): title 
                for title in article_titles
            }
            
            all_documents = []
            
            for future in as_completed(future_to_title):
                title = future_to_title[future]
                try:
                    article = future.result()
                    if article:
                        # Convert to documents
                        documents = self.process_article_to_documents(article)
                        if documents:
                            all_documents.extend(documents)
                            results['successful'] += 1
                            results['documents_created'] += len(documents)
                            logger.info(f"Processed article: {title} ({len(documents)} chunks)")
                        else:
                            results['failed'] += 1
                            results['failed_articles'].append(title)
                    else:
                        results['failed'] += 1
                        results['failed_articles'].append(title)
                        
                except Exception as e:
                    logger.error(f"Failed to process article '{title}': {e}")
                    results['failed'] += 1
                    results['failed_articles'].append(title)
        
        # Batch add documents to vector store
        if all_documents:
            try:
                logger.info(f"Adding {len(all_documents)} documents to vector store...")
                self.vector_store.add_documents(all_documents)
                logger.info(f"Successfully added {len(all_documents)} Wikipedia documents to vector store")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                results['failed'] = results['total_articles']
                results['successful'] = 0
                results['documents_created'] = 0
        
        return results
    
    def ingest_wikipedia_by_categories(self, categories: List[str], articles_per_category: int = 25) -> Dict[str, Any]:
        """Ingest Wikipedia articles from specific categories"""
        all_article_titles = []
        
        for category in categories:
            logger.info(f"Fetching articles from category: {category}")
            articles = self.get_articles_from_category(category, articles_per_category)
            titles = [article['title'] for article in articles]
            all_article_titles.extend(titles)
            logger.info(f"Found {len(titles)} articles in {category}")
        
        # Remove duplicates
        unique_titles = list(set(all_article_titles))
        logger.info(f"Total unique articles to process: {len(unique_titles)}")
        
        return self.ingest_wikipedia_batch(unique_titles, "wikipedia_categories")
    
    def ingest_random_wikipedia_sample(self, count: int = 100) -> Dict[str, Any]:
        """Ingest a random sample of Wikipedia articles"""
        logger.info(f"Fetching {count} random Wikipedia articles...")
        
        articles = self.get_random_articles(count)
        titles = [article['title'] for article in articles if article.get('title')]
        
        return self.ingest_wikipedia_batch(titles, "wikipedia_random")