import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from datetime import datetime

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from ..utils.utils import setup_logging, clean_text
from .models import DataSource, ProcessingStatus

logger = setup_logging(__name__)

@dataclass
class TavilySearchResult:
    """Data class for Tavily search results"""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None
    
class TavilyIntegrationService:
    """
    Service for integrating Tavily search API with RAG system
    Fetches real-time web data, cleans it, and prepares it for vector storage
    """
    
    def __init__(self):
        self.client = None
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.is_ready = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Tavily client"""
        if not TAVILY_AVAILABLE:
            logger.warning("Tavily Python package not available. Please install with: pip install tavily-python")
            return
            
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not found in environment variables")
            return
            
        try:
            self.client = TavilyClient(api_key=self.api_key)
            self.is_ready = True
            logger.info("Tavily client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            self.is_ready = False
    
    def search_and_fetch(
        self, 
        query: str, 
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[TavilySearchResult]:
        """
        Search the web using Tavily and fetch clean content
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            search_depth: Search depth ('basic' or 'advanced')
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            
        Returns:
            List of TavilySearchResult objects
        """
        if not self.is_ready:
            logger.error("Tavily client not ready")
            return []
        
        try:
            logger.info(f"Searching Tavily for: '{query}' with {max_results} results")
            
            # Prepare search parameters
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": True
            }
            
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            # Perform search
            response = self.client.search(**search_params)
            
            # Process results
            results = []
            for result in response.get("results", []):
                try:
                    # Clean and process content
                    cleaned_content = self._clean_web_content(result.get("content", ""))
                    
                    if len(cleaned_content.strip()) < 50:  # Skip very short content
                        continue
                    
                    tavily_result = TavilySearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=cleaned_content,
                        score=result.get("score", 0.0),
                        published_date=result.get("published_date")
                    )
                    
                    results.append(tavily_result)
                    
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    def _clean_web_content(self, content: str) -> str:
        """
        Clean and process web content for better embedding quality
        """
        if not content:
            return ""
        
        # Use the existing clean_text utility
        cleaned = clean_text(content)
        
        # Additional web-specific cleaning
        import re
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Remove common web artifacts
        artifacts_to_remove = [
            r'Cookie Policy',
            r'Privacy Policy',
            r'Terms of Service',
            r'Subscribe to our newsletter',
            r'Follow us on',
            r'Share this article',
            r'Advertisement',
            r'Sponsored content'
        ]
        
        for artifact in artifacts_to_remove:
            cleaned = re.sub(artifact, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def create_data_sources_from_results(
        self, 
        results: List[TavilySearchResult], 
        query: str
    ) -> List[DataSource]:
        """
        Convert Tavily search results to DataSource objects for processing
        """
        data_sources = []
        
        for i, result in enumerate(results):
            data_source = DataSource(
                id=f"tavily_{int(time.time())}_{i}",
                type="web_search",
                url=result.url,
                title=result.title or f"Web Search Result {i+1}",
                description=f"Tavily search result for query: '{query}'",
                status=ProcessingStatus.PENDING,
                content=result.content,
                metadata={
                    "search_query": query,
                    "search_score": result.score,
                    "published_date": result.published_date,
                    "source": "tavily",
                    "fetched_at": datetime.now().isoformat()
                }
            )
            data_sources.append(data_source)
        
        return data_sources
    
    def get_contextual_search_results(
        self, 
        original_query: str, 
        existing_docs_summary: str = "",
        max_results: int = 5
    ) -> List[TavilySearchResult]:
        """
        Perform contextual search based on existing documents and query
        """
        # Enhance query with context if available
        enhanced_query = original_query
        if existing_docs_summary:
            # Extract key topics to make search more targeted
            enhanced_query = f"{original_query} {existing_docs_summary[:200]}"
        
        # Prioritize recent and authoritative sources
        exclude_domains = [
            "reddit.com",
            "quora.com", 
            "answers.yahoo.com",
            "ask.com"
        ]
        
        return self.search_and_fetch(
            query=enhanced_query,
            max_results=max_results,
            search_depth="advanced",
            exclude_domains=exclude_domains
        )
    
    def is_available(self) -> bool:
        """Check if Tavily integration is available and ready"""
        return self.is_ready and TAVILY_AVAILABLE

# Service instance
_tavily_service = None

def get_tavily_integration_service() -> TavilyIntegrationService:
    """Get singleton instance of Tavily integration service"""
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilyIntegrationService()
    return _tavily_service