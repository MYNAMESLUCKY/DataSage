"""
Speed-Optimized Tavily Integration
Provides fast web search with intelligent caching
"""

import logging
import time
from typing import Dict, List, Optional, Any
from src.backend.tavily_integration import TavilyIntegrationService
from src.backend.fast_web_cache import fast_web_cache

logger = logging.getLogger(__name__)

class FastTavilyService:
    """
    Speed-optimized wrapper for Tavily web search
    Provides sub-second responses through intelligent caching
    """
    
    def __init__(self):
        self.tavily_service = TavilyIntegrationService()
        self.web_cache = fast_web_cache
        
    def fast_search_and_fetch(self, query: str, max_results: int = 5, 
                            search_depth: str = "basic") -> Dict[str, Any]:
        """
        Fast web search with caching optimization
        Returns cached results in < 50ms when available
        """
        start_time = time.time()
        
        # Step 1: Check fast cache first
        cached_response = self.web_cache.get_fast_web_response(query)
        
        if cached_response:
            # Return cached results with fast response indicators
            processing_time = time.time() - start_time
            cached_response['total_processing_time'] = processing_time
            cached_response['optimization_used'] = True
            
            logger.info(f"Fast web cache hit for '{query[:50]}...' in {processing_time*1000:.1f}ms")
            return cached_response
        
        # Step 2: Perform fresh web search
        try:
            logger.info(f"Performing fresh web search for: {query[:50]}...")
            
            # Use Tavily for fresh search
            web_results = self.tavily_service.search_and_fetch(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )
            
            if web_results:
                # Process results for caching
                processed_results = []
                for result in web_results:
                    processed_result = {
                        "url": getattr(result, 'url', ''),
                        "title": getattr(result, 'title', ''),
                        "content": getattr(result, 'content', '')[:1000],  # Limit content for cache efficiency
                        "score": getattr(result, 'score', 0.0)
                    }
                    processed_results.append(processed_result)
                
                # Cache the results for future fast access
                self.web_cache.cache_web_response(
                    query=query,
                    results=processed_results,
                    confidence=0.8,
                    source="tavily"
                )
                
                processing_time = time.time() - start_time
                
                logger.info(f"Fresh web search completed in {processing_time:.2f}s, cached for future fast access")
                
                return {
                    "results": processed_results,
                    "confidence": 0.8,
                    "source": "tavily",
                    "processing_time_ms": processing_time * 1000,
                    "cache_type": "fresh_search",
                    "status": "success",
                    "total_processing_time": processing_time,
                    "optimization_used": False
                }
            else:
                return {
                    "results": [],
                    "confidence": 0.0,
                    "source": "tavily",
                    "status": "no_results",
                    "total_processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"Fast web search failed: {e}")
            return {
                "results": [],
                "confidence": 0.0,
                "source": "tavily",
                "status": "error",
                "error": str(e),
                "total_processing_time": time.time() - start_time
            }
    
    def is_available(self) -> bool:
        """Check if web search service is available"""
        return self.tavily_service.is_available()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for web search"""
        cache_stats = self.web_cache.get_cache_stats()
        return {
            "web_cache_stats": cache_stats,
            "tavily_available": self.is_available(),
            "performance_targets": {
                "cached_responses_ms": "< 50ms",
                "fresh_searches_s": "< 3s",
                "cache_hit_rate": f"{cache_stats['total_accesses']} total accesses"
            }
        }

# Global fast Tavily service instance
fast_tavily_service = FastTavilyService()