"""
Fast Processing Mode for Enterprise RAG System
Optimizes query processing for speed with configurable performance settings
"""

import time
import logging
from typing import Dict, Any, Optional, List
from .hybrid_rag_processor import HybridRAGProcessor

logger = logging.getLogger(__name__)

class FastRAGProcessor:
    """
    Speed-optimized RAG processor with configurable performance modes
    """
    
    def __init__(self, hybrid_processor: HybridRAGProcessor):
        self.hybrid_processor = hybrid_processor
        self.speed_mode = "balanced"  # fast, balanced, thorough
        
    def set_speed_mode(self, mode: str):
        """Set processing speed mode: fast, balanced, thorough"""
        valid_modes = ["fast", "balanced", "thorough"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.speed_mode = mode
        logger.info(f"Speed mode set to: {mode}")
    
    def get_speed_config(self) -> Dict[str, Any]:
        """Get configuration based on speed mode"""
        configs = {
            "fast": {
                "max_query_variations": 2,
                "max_kb_searches": 2,
                "max_rerank_candidates": 15,
                "skip_web_search": False,
                "max_web_results": 3,
                "use_simple_reranking": True,
                "target_time": 8.0  # seconds
            },
            "balanced": {
                "max_query_variations": 4,
                "max_kb_searches": 3,
                "max_rerank_candidates": 20,
                "skip_web_search": False,
                "max_web_results": 5,
                "use_simple_reranking": False,
                "target_time": 15.0  # seconds
            },
            "thorough": {
                "max_query_variations": 8,
                "max_kb_searches": 5,
                "max_rerank_candidates": 30,
                "skip_web_search": False,
                "max_web_results": 10,
                "use_simple_reranking": False,
                "target_time": 30.0  # seconds
            }
        }
        return configs.get(self.speed_mode, configs["balanced"])
    
    def process_fast_query(
        self, 
        query: str, 
        llm_model: Optional[str] = None,
        max_results: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query with speed optimizations based on current mode
        """
        start_time = time.time()
        config = self.get_speed_config()
        
        try:
            logger.info(f"Processing query in {self.speed_mode} mode (target: {config['target_time']}s)")
            
            # Apply speed-specific parameters
            optimized_params = {
                'use_web_search': not config.get('skip_web_search', False),
                'max_web_results': config['max_web_results'],
                'max_results': max_results,
                'llm_model': llm_model,
                **kwargs
            }
            
            # For fast mode, check cache more aggressively
            if self.speed_mode == "fast":
                # Try to get any cached result, even if slightly stale
                cached_result = self._get_fast_cache_result(query, optimized_params)
                if cached_result:
                    cached_result['processing_time'] = time.time() - start_time
                    cached_result['speed_mode'] = self.speed_mode
                    logger.info(f"Fast cache hit - {cached_result['processing_time']:.2f}s")
                    return cached_result
            
            # Process with hybrid processor using optimized parameters
            result = self.hybrid_processor.process_intelligent_query(
                query=query,
                **optimized_params
            )
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['speed_mode'] = self.speed_mode
            result['target_time'] = config['target_time']
            
            # Add performance indicator
            if processing_time < config['target_time']:
                result['performance_status'] = 'fast'
            elif processing_time < config['target_time'] * 1.5:
                result['performance_status'] = 'acceptable'
            else:
                result['performance_status'] = 'slow'
            
            logger.info(f"Query processed in {processing_time:.2f}s ({result['performance_status']})")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Fast processing failed after {processing_time:.2f}s: {e}")
            
            return {
                'status': 'error',
                'answer': f"Fast processing failed: {str(e)}",
                'processing_time': processing_time,
                'speed_mode': self.speed_mode,
                'performance_status': 'error'
            }
    
    def _get_fast_cache_result(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result with more aggressive caching for fast mode"""
        try:
            cache_manager = self.hybrid_processor.cache_manager
            
            # Try exact cache first
            cached = cache_manager.get_cached_query_result(query, params)
            if cached and cached.get('status') != 'error':
                logger.info("Fast cache: exact match found")
                return cached
            
            # For fast mode, try similar queries cache
            if hasattr(cache_manager, 'get_similar_cached_result'):
                similar_cached = cache_manager.get_similar_cached_result(query, similarity_threshold=0.85)
                if similar_cached and similar_cached.get('status') != 'error':
                    logger.info("Fast cache: similar query match found")
                    return similar_cached
            
            return None
            
        except Exception as e:
            logger.warning(f"Fast cache lookup failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        config = self.get_speed_config()
        return {
            'speed_mode': self.speed_mode,
            'target_time': config['target_time'],
            'max_query_variations': config['max_query_variations'],
            'max_kb_searches': config['max_kb_searches'],
            'max_rerank_candidates': config['max_rerank_candidates'],
            'optimization_level': 'high' if self.speed_mode == 'fast' else 'medium' if self.speed_mode == 'balanced' else 'low'
        }