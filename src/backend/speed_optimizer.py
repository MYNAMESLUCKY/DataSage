"""
Speed Optimization System for Simple Queries
Provides sub-second response times through intelligent routing
"""

import time
import logging
from typing import Dict, Optional, List
from src.backend.complexity_classifier import classify_query_complexity, ComplexityLevel
from src.backend.fast_response_cache import fast_cache

logger = logging.getLogger(__name__)

class SpeedOptimizer:
    """
    Intelligent query routing system for optimal response times
    Routes simple queries to fast paths, complex queries to full processing
    """
    
    def __init__(self):
        self.simple_query_threshold = 0.3  # Complexity below this uses fast path
        self.moderate_query_threshold = 0.6  # Above this uses GPU if available
        
        # Performance targets (milliseconds)
        self.target_simple_response_time = 100  # < 100ms for simple queries
        self.target_moderate_response_time = 2000  # < 2s for moderate queries
        self.target_complex_response_time = 10000  # < 10s for complex queries
        
        logger.info("Speed optimizer initialized with performance targets")
    
    def should_use_fast_path(self, query: str) -> bool:
        """
        Determine if query should use fast response path
        """
        start_time = time.time()
        
        # Quick pattern matching for common simple queries
        simple_patterns = [
            "what is",
            "define",
            "definition of",
            "meaning of",
            "explain",
            "how does",
            "why does",
            "who is",
            "where is",
            "when is"
        ]
        
        query_lower = query.lower().strip()
        
        # Check for simple patterns
        for pattern in simple_patterns:
            if query_lower.startswith(pattern) and len(query.split()) <= 6:
                classification_time = (time.time() - start_time) * 1000
                logger.info(f"Fast path recommended via pattern matching in {classification_time:.1f}ms")
                return True
        
        # Use complexity classifier for more sophisticated analysis
        try:
            complexity_analysis = classify_query_complexity(query)
            classification_time = (time.time() - start_time) * 1000
            
            should_fast_path = complexity_analysis.score < self.simple_query_threshold
            
            logger.info(f"Complexity analysis completed in {classification_time:.1f}ms: "
                       f"score={complexity_analysis.score:.3f}, fast_path={should_fast_path}")
            
            return should_fast_path
            
        except Exception as e:
            logger.error(f"Complexity classification failed: {e}")
            # Default to fast path for safety
            return True
    
    def get_optimized_response(self, query: str, rag_processor=None) -> Dict:
        """
        Get optimized response using the fastest appropriate method
        """
        start_time = time.time()
        
        # Step 1: Check if we should use fast path
        use_fast_path = self.should_use_fast_path(query)
        
        if use_fast_path:
            # Step 2: Try fast cache first (only for very basic queries)
            fast_response = fast_cache.get_fast_response(query)
            
            # Only use cached responses for exact basic definitions
            query_lower = query.lower().strip()
            basic_definition_queries = ["what is ai", "what is machine learning", "what is deep learning"]
            
            if fast_response and query_lower in basic_definition_queries:
                total_time = (time.time() - start_time) * 1000
                fast_response['total_processing_time_ms'] = total_time
                fast_response['optimization_path'] = 'basic_definition_cache'
                
                logger.info(f"Basic definition cache response delivered in {total_time:.1f}ms")
                return fast_response
            
            # Step 3: If no cache hit, generate response and cache it
            if rag_processor:
                try:
                    # Use lightweight processing for simple queries
                    response = self._generate_lightweight_response(query, rag_processor)
                    
                    # Cache the response for future use
                    fast_cache.cache_response(
                        query=query,
                        response=response['answer'],
                        confidence=response['confidence'],
                        sources=response.get('sources', [])
                    )
                    
                    total_time = (time.time() - start_time) * 1000
                    response['total_processing_time_ms'] = total_time
                    response['optimization_path'] = 'lightweight_processing'
                    
                    logger.info(f"Lightweight response generated in {total_time:.1f}ms")
                    return response
                    
                except Exception as e:
                    logger.error(f"Lightweight processing failed: {e}")
                    # Fall back to full processing
        
        # Step 4: Use full processing for complex queries or fallback
        if rag_processor:
            try:
                response = rag_processor.process_intelligent_hybrid_query(
                    query=query,
                    max_sources=5,  # Limit sources for speed
                    web_search=False,  # Skip web search for speed
                    llm_model="sarvam-m"
                )
                
                total_time = (time.time() - start_time) * 1000
                response['total_processing_time_ms'] = total_time
                response['optimization_path'] = 'full_processing'
                
                # Cache if it was actually simple
                if use_fast_path and response.get('status') == 'success':
                    fast_cache.cache_response(
                        query=query,
                        response=response['answer'],
                        confidence=response.get('confidence', 0.8),
                        sources=response.get('sources', [])
                    )
                
                logger.info(f"Full processing completed in {total_time:.1f}ms")
                return response
                
            except Exception as e:
                logger.error(f"Full processing failed: {e}")
        
        # Final fallback
        total_time = (time.time() - start_time) * 1000
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again.",
            "confidence": 0.1,
            "sources": [],
            "status": "error",
            "total_processing_time_ms": total_time,
            "optimization_path": "error_fallback"
        }
    
    def _generate_lightweight_response(self, query: str, rag_processor) -> Dict:
        """
        Generate response using lightweight processing with KB + fast web search
        """
        try:
            # Quick knowledge base search
            kb_docs = rag_processor.vector_store.similarity_search(
                query=query,
                k=5  # Limit to 5 most relevant docs for speed
            )
            
            # Fast web search for current information
            from src.backend.fast_tavily_service import fast_tavily_service
            web_response = fast_tavily_service.fast_search_and_fetch(
                query=query,
                max_results=3,  # Limit for speed
                search_depth="basic"
            )
            
            # Combine KB and web sources
            all_docs = kb_docs.copy()
            web_sources = []
            
            if web_response.get('status') == 'success' and web_response.get('results'):
                # Add web results as documents
                for result in web_response['results'][:2]:  # Limit to 2 web results for speed
                    from langchain.schema import Document
                    web_doc = Document(
                        page_content=f"Title: {result.get('title', '')}\n\nContent: {result.get('content', '')}",
                        metadata={
                            "source": result.get('url', ''),
                            "title": result.get('title', ''),
                            "type": "web_data",
                            "score": result.get('score', 0.0)
                        }
                    )
                    all_docs.append(web_doc)
                    web_sources.append({
                        "url": result.get('url', ''),
                        "title": result.get('title', ''),
                        "score": result.get('score', 0.0)
                    })
            
            # Generate answer using combined KB + web content
            response = rag_processor.rag_engine.generate_answer(
                query=query,
                relevant_docs=all_docs,
                llm_model="sarvam-m"
            )
            
            # Add source information
            kb_sources = []
            for doc in kb_docs:
                if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    kb_sources.append(doc.metadata['source'])
            
            response['sources'] = kb_sources[:2]  # KB sources
            response['web_sources'] = web_sources  # Web sources
            response['web_cache_used'] = web_response.get('cache_type', 'none') != 'fresh_search'
            response['web_processing_time_ms'] = web_response.get('processing_time_ms', 0)
            
            return response
            
        except Exception as e:
            logger.error(f"Lightweight response generation failed: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict:
        """Get speed optimization performance metrics"""
        cache_stats = fast_cache.get_cache_stats()
        
        return {
            "performance_targets": {
                "simple_queries_ms": self.target_simple_response_time,
                "moderate_queries_ms": self.target_moderate_response_time,
                "complex_queries_ms": self.target_complex_response_time
            },
            "thresholds": {
                "simple_threshold": self.simple_query_threshold,
                "moderate_threshold": self.moderate_query_threshold
            },
            "cache_performance": cache_stats
        }

# Global speed optimizer instance
speed_optimizer = SpeedOptimizer()