"""
Search API Fallback for RAG System
==================================

This module provides fallback search capabilities when vector store 
retrieval doesn't find sufficient relevant documents.
"""

import logging
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from .utils import setup_logging

logger = setup_logging(__name__)

@dataclass
class SearchResult:
    """Structure for search results"""
    title: str
    content: str
    url: str
    relevance_score: float
    source_type: str

class SearchFallbackService:
    """Fallback search service for when vector store fails"""
    
    def __init__(self):
        self.enabled = True
        self.max_results = 3
        self.timeout = 10
    
    def search_fallback(self, query: str, min_results: int = 1) -> List[SearchResult]:
        """
        Perform fallback search when vector store doesn't return enough results
        """
        if not self.enabled:
            logger.info("Search fallback disabled")
            return []
        
        try:
            logger.info(f"Performing fallback search for query: {query}")
            
            # Try multiple search strategies
            results = []
            
            # Strategy 1: Wikipedia API search
            wiki_results = self._search_wikipedia(query)
            results.extend(wiki_results)
            
            # Strategy 2: DuckDuckGo Instant Answer (if available)
            instant_results = self._search_instant_answers(query)
            results.extend(instant_results)
            
            # Remove duplicates and sort by relevance
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Fallback search found {len(sorted_results)} results")
            return sorted_results[:self.max_results]
            
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return []
    
    def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """Search Wikipedia API for relevant articles"""
        try:
            # Wikipedia API search
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            
            # Clean query for Wikipedia
            clean_query = query.replace("what is ", "").replace("define ", "").strip()
            
            response = requests.get(
                f"{search_url}{clean_query}",
                timeout=self.timeout,
                headers={'User-Agent': 'RAG-System/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'extract' in data and data['extract']:
                    return [SearchResult(
                        title=data.get('title', clean_query),
                        content=data['extract'],
                        url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        relevance_score=0.8,
                        source_type='wikipedia'
                    )]
            
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {str(e)}")
        
        return []
    
    def _search_instant_answers(self, query: str) -> List[SearchResult]:
        """Search for instant answers from DuckDuckGo"""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                
                # Check for abstract
                if data.get('Abstract'):
                    results.append(SearchResult(
                        title=data.get('Heading', query),
                        content=data['Abstract'],
                        url=data.get('AbstractURL', ''),
                        relevance_score=0.7,
                        source_type='duckduckgo'
                    ))
                
                # Check for definition
                if data.get('Definition'):
                    results.append(SearchResult(
                        title=f"Definition: {query}",
                        content=data['Definition'],
                        url=data.get('DefinitionURL', ''),
                        relevance_score=0.9,
                        source_type='definition'
                    ))
                
                return results
            
        except Exception as e:
            logger.debug(f"Instant answer search failed: {str(e)}")
        
        return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        if not results:
            return []
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Create a content hash for deduplication
            content_hash = hash(result.content[:200])
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def format_fallback_response(self, query: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """Format fallback search results into a proper response"""
        if not search_results:
            return {
                'answer': "I couldn't find relevant information in the knowledge base or external sources. Please try rephrasing your question or providing more specific details.",
                'sources': [],
                'confidence': 0.0,
                'fallback_used': True
            }
        
        # Combine search results into a comprehensive answer
        answer_parts = []
        sources = []
        
        for result in search_results:
            answer_parts.append(result.content)
            if result.url:
                sources.append(result.url)
            else:
                sources.append(f"{result.source_type}: {result.title}")
        
        # Create a coherent answer
        if len(answer_parts) == 1:
            answer = answer_parts[0]
        else:
            answer = f"Based on external sources: {' '.join(answer_parts)}"
        
        # Calculate confidence based on source quality
        avg_score = sum(r.relevance_score for r in search_results) / len(search_results)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': min(avg_score, 0.8),  # Cap confidence for external sources
            'fallback_used': True,
            'search_results_count': len(search_results)
        }

# Global instance
_search_fallback_service = None

def get_search_fallback_service() -> SearchFallbackService:
    """Get global search fallback service instance"""
    global _search_fallback_service
    if _search_fallback_service is None:
        _search_fallback_service = SearchFallbackService()
    return _search_fallback_service