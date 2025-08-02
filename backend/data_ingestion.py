import os
import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import hashlib

# Web scraping and content extraction
import requests
import trafilatura
from bs4 import BeautifulSoup

# Document processing
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .utils import setup_logging, validate_url, clean_text
from .models import DataSource

logger = setup_logging(__name__)

class DataIngestionService:
    """
    Service responsible for ingesting data from various online sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def _initialize_text_splitter(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize text splitter with specified parameters"""
        if chunk_size != self.text_splitter._chunk_size or chunk_overlap != self.text_splitter._chunk_overlap:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
    
    def ingest_from_url(
        self, 
        url: str, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Ingest and process content from a URL
        """
        try:
            logger.info(f"Ingesting content from URL: {url}")
            
            # Normalize and validate URL
            from .utils import normalize_url
            url = normalize_url(url)
            if not validate_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Initialize text splitter
            self._initialize_text_splitter(chunk_size, chunk_overlap)
            
            # Extract content from URL
            content = self._extract_web_content(url)
            
            if not content:
                logger.warning(f"No content extracted from URL: {url}")
                return []
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'source': url,
                    'extraction_time': time.time(),
                    'content_hash': hashlib.md5(content.encode()).hexdigest(),
                    'content_length': len(content)
                }
            )
            
            # Split document into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.page_content)
                })
            
            logger.info(f"Successfully processed {len(chunks)} chunks from {url}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error ingesting content from {url}: {str(e)}")
            return []
    
    def _extract_web_content(self, url: str) -> str:
        """
        Extract clean text content from a web page
        """
        try:
            # Enhanced headers for better scraping success
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }
            
            # Fetch the webpage with enhanced headers
            response = self.session.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Use trafilatura with enhanced settings for content extraction
            text_content = trafilatura.extract(
                response.text, 
                include_comments=False, 
                include_tables=True,
                include_formatting=False,
                favor_precision=True
            )
            
            if text_content and len(text_content.strip()) > 50:
                return clean_text(text_content)
            
            # Fallback: use BeautifulSoup if trafilatura fails
            logger.info("Trafilatura extraction failed, falling back to BeautifulSoup")
            extracted = self._extract_with_beautifulsoup(response.text, url)
            
            if extracted and len(extracted.strip()) > 50:
                return extracted
            
            # Special handling for news sites with JavaScript content
            if 'news.google.com' in url or 'google.com/news' in url:
                return self._extract_google_news_content(response.text, url)
            
            return ""
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return ""
    
    def _extract_with_beautifulsoup(self, html_content: str, url: str) -> str:
        """
        Fallback content extraction using BeautifulSoup
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry', '.article-body', '[role="main"]'
            ]
            
            extracted_text = ""
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        extracted_text += element.get_text(separator=' ', strip=True) + "\n"
                    break
            
            # If no main content found, extract from body
            if not extracted_text.strip():
                body = soup.find('body')
                if body:
                    extracted_text = body.get_text(separator=' ', strip=True)
            
            return clean_text(extracted_text) if extracted_text else ""
            
        except Exception as e:
            logger.error(f"BeautifulSoup extraction failed for {url}: {str(e)}")
            return ""
    
    def ingest_multiple_urls(
        self, 
        urls: List[str], 
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        max_workers: int = 4
    ) -> Dict[str, List[Document]]:
        """
        Ingest content from multiple URLs concurrently
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.ingest_from_url, url, chunk_size, chunk_overlap): url 
                for url in urls
            }
            
            # Collect results
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    documents = future.result()
                    results[url] = documents
                    logger.info(f"Completed ingestion for {url}: {len(documents)} chunks")
                except Exception as e:
                    logger.error(f"Failed to ingest {url}: {str(e)}")
                    results[url] = []
        
        return results
    
    def get_url_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get metadata about a URL without full content extraction
        """
        try:
            response = self.session.head(url, timeout=10)
            response.raise_for_status()
            
            metadata = {
                'url': url,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': response.headers.get('content-length'),
                'last_modified': response.headers.get('last-modified'),
                'server': response.headers.get('server', ''),
                'accessible': True
            }
            
            # Try to get title
            if 'text/html' in metadata['content_type']:
                try:
                    response = self.session.get(url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.find('title')
                    if title:
                        metadata['title'] = title.get_text().strip()
                except:
                    pass
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for {url}: {str(e)}")
            return {
                'url': url,
                'accessible': False,
                'error': str(e)
            }
    
    def validate_data_source(self, source: DataSource) -> Dict[str, Any]:
        """
        Validate a data source before processing
        """
        validation_result = {
            'valid': False,
            'issues': [],
            'metadata': {}
        }
        
        try:
            # Check URL validity
            if not validate_url(source.url):
                validation_result['issues'].append('Invalid URL format')
                return validation_result
            
            # Get URL metadata
            metadata = self.get_url_metadata(source.url)
            validation_result['metadata'] = metadata
            
            if not metadata.get('accessible', False):
                validation_result['issues'].append(f"URL not accessible: {metadata.get('error', 'Unknown error')}")
                return validation_result
            
            # Check content type
            content_type = metadata.get('content_type', '').lower()
            if not any(ct in content_type for ct in ['text/html', 'text/plain', 'application/json']):
                validation_result['issues'].append(f"Unsupported content type: {content_type}")
            
            # Check content length
            content_length = metadata.get('content_length')
            if content_length:
                try:
                    length = int(content_length)
                    if length > 10_000_000:  # 10MB limit
                        validation_result['issues'].append(f"Content too large: {length} bytes")
                except ValueError:
                    pass
            
            # If no major issues, mark as valid
            if not validation_result['issues']:
                validation_result['valid'] = True
            
            return validation_result
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
