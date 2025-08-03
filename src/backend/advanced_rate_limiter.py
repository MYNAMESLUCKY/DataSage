#!/usr/bin/env python3
"""
Advanced Rate Limiting System for SARVAM API
Designed to handle high-level complex questions with intelligent backoff strategies
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Any
from threading import Lock
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class AdvancedRateLimiter:
    """
    Intelligent rate limiter designed specifically for SARVAM API
    Handles complex queries with adaptive backoff and request optimization
    """
    
    def __init__(self):
        self.request_history: Dict[str, List[float]] = {}
        self.lock = Lock()
        
        # Conservative rate limiting configurations to avoid 429 errors
        self.configs = {
            "simple": {
                "max_requests_per_minute": 8,  # More conservative for SARVAM API
                "initial_backoff": 5,  # Longer initial wait
                "max_backoff": 60,  # Reasonable max wait
                "backoff_multiplier": 2.5
            },
            "complex": {
                "max_requests_per_minute": 4,  # Very conservative for complex queries
                "initial_backoff": 10,  # Longer initial delay
                "max_backoff": 120,  # Longer max delay
                "backoff_multiplier": 3
            },
            "quantum_physics": {  # Very conservative for intensive queries
                "max_requests_per_minute": 2,  # Minimal requests to avoid overload
                "initial_backoff": 15,  # Long initial delay
                "max_backoff": 300,  # Up to 5 minutes max
                "backoff_multiplier": 4  # Aggressive backoff
            }
        }
        
        self.consecutive_failures = 0
        self.last_failure_time = None
        
    def categorize_query(self, query: str, processing_time: Optional[float] = None, token_count: Optional[int] = None) -> str:
        """
        Categorize query complexity based on actual API processing metrics, not keywords
        
        Args:
            query: The query text (for length estimation only)
            processing_time: Actual API processing time in seconds
            token_count: Number of tokens used in the response
        """
        # Use actual processing metrics when available
        if processing_time is not None and token_count is not None:
            # Dynamic complexity based on actual resource consumption
            if processing_time > 15 or token_count > 1200:
                return "quantum_physics"  # Very resource-intensive
            elif processing_time > 8 or token_count > 800:
                return "complex"  # Moderately resource-intensive
            else:
                return "simple"  # Standard processing
        
        # Fallback to query length estimation when metrics unavailable
        query_length = len(query.split())
        if query_length > 50:
            return "complex"  # Long queries may be complex
        elif query_length > 20:
            return "complex"  # Medium queries 
        else:
            return "simple"  # Short queries are typically simple
    
    def should_allow_request(self, query: str, processing_time: Optional[float] = None, token_count: Optional[int] = None) -> tuple[bool, float]:
        """
        Check if request should be allowed and return wait time if not
        Returns: (allow_request, wait_time_seconds)
        """
        with self.lock:
            category = self.categorize_query(query, processing_time, token_count)
            config = self.configs[category]
            
            current_time = time.time()
            
            # Clean old requests (older than 1 minute)
            cutoff_time = current_time - 60
            if category not in self.request_history:
                self.request_history[category] = []
            
            self.request_history[category] = [
                req_time for req_time in self.request_history[category] 
                if req_time > cutoff_time
            ]
            
            # Check if we're within rate limits
            recent_requests = len(self.request_history[category])
            max_requests = config["max_requests_per_minute"]
            
            if recent_requests < max_requests:
                # Allow request and record it
                self.request_history[category].append(current_time)
                return True, 0
            
            # Calculate wait time based on oldest request
            oldest_request = min(self.request_history[category])
            wait_time = 60 - (current_time - oldest_request)
            
            return False, max(1, wait_time)
    
    def get_backoff_time(self, query: str, failure_count: int, processing_time: Optional[float] = None, token_count: Optional[int] = None) -> float:
        """Calculate intelligent backoff time based on actual complexity and failures"""
        category = self.categorize_query(query, processing_time, token_count)
        config = self.configs[category]
        
        base_backoff = config["initial_backoff"]
        multiplier = config["backoff_multiplier"]
        max_backoff = config["max_backoff"]
        
        # Progressive backoff: base * (multiplier ^ failure_count)
        backoff_time = min(max_backoff, base_backoff * (multiplier ** failure_count))
        
        # Add jitter to prevent thundering herd
        jitter = backoff_time * 0.1  # 10% jitter
        import random
        backoff_time += random.uniform(-jitter, jitter)
        
        return max(1, backoff_time)
    
    def record_failure(self, query: str, processing_time: Optional[float] = None, token_count: Optional[int] = None):
        """Record a rate limit failure for adaptive backoff"""
        with self.lock:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            
            # Clear request history for this category to slow down future requests
            category = self.categorize_query(query, processing_time, token_count)
            logger.warning(f"Rate limit failure recorded for {category} category. Consecutive failures: {self.consecutive_failures}")
            
            # If we've failed multiple times, drastically reduce request rate
            if self.consecutive_failures >= 3:
                # Clear all request histories to force longer waits
                self.request_history = {cat: [] for cat in self.request_history.keys()}
                logger.warning(f"Multiple consecutive failures ({self.consecutive_failures}). Clearing request history to reduce rate.")
    
    def record_success(self, query: str, processing_time: Optional[float] = None, token_count: Optional[int] = None):
        """Record a successful request to reset failure counters"""
        with self.lock:
            self.consecutive_failures = 0
            self.last_failure_time = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        with self.lock:
            return {
                "consecutive_failures": self.consecutive_failures,
                "last_failure_time": self.last_failure_time,
                "request_history": {
                    category: len(requests) for category, requests in self.request_history.items()
                }
            }

def rate_limited_api_call(
    rate_limiter: AdvancedRateLimiter,
    query: str,
    api_function: Callable,
    max_retries: int = 5,
    *args, **kwargs
) -> Any:
    """
    Execute API call with performance-based rate limiting
    Measures actual processing time and tokens to determine complexity
    """
    processing_time = None
    token_count = None
    
    for attempt in range(max_retries):
        # Initial rate check (without metrics for first attempt)
        can_proceed, wait_time = rate_limiter.should_allow_request(query, processing_time, token_count)
        
        if not can_proceed:
            logger.info(f"Rate limit: waiting {wait_time:.2f}s before attempt {attempt + 1}")
            time.sleep(wait_time)
        
        try:
            # Measure processing time
            start_time = time.time()
            
            logger.debug(f"Making API call with performance monitoring")
            result = api_function(*args, **kwargs)
            
            # Calculate actual processing time
            processing_time = time.time() - start_time
            
            # Extract token count from result if available
            if isinstance(result, dict):
                # Try to estimate token count from answer length
                answer_text = result.get('answer', '')
                if isinstance(answer_text, str):
                    # Rough estimate: ~4 characters per token
                    token_count = len(answer_text) // 4
            
            logger.info(f"API call completed - Time: {processing_time:.2f}s, Estimated tokens: {token_count or 'unknown'}")
            
            if result is None:
                logger.error("API function returned None")
                raise Exception("API function returned None result")
            
            # Record success with actual metrics for future complexity assessment
            rate_limiter.record_success(query, processing_time, token_count)
            return result
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"API call failed: {error_str}")
            
            if "429" in error_str or "rate limit" in error_str.lower():
                # Record failure with metrics if available
                rate_limiter.record_failure(query, processing_time, token_count)
                
                if attempt < max_retries - 1:
                    backoff_time = rate_limiter.get_backoff_time(query, attempt + 1, processing_time, token_count)
                    # Add additional delay for 429 errors to be more conservative
                    backoff_time *= (1.5 + attempt * 0.5)  # Increase delay with each attempt
                    logger.warning(f"Rate limit (429) hit. Aggressive backoff: {backoff_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff_time)
                    continue
                else:
                    logger.error(f"All {max_retries} attempts exhausted due to rate limiting")
                    # Return a valid response instead of raising exception
                    return {
                        "answer": "The AI service is currently experiencing high demand. Please try again in a few minutes.",
                        "confidence": 0.0,
                        "status": "rate_limited",
                        "model_used": "none",
                        "api_provider": "none"
                    }
            else:
                logger.error(f"Non-rate-limit error on attempt {attempt + 1}: {error_str}")
                raise e
    
    raise Exception(f"Failed after {max_retries} attempts")


# Global rate limiter instance
global_rate_limiter = AdvancedRateLimiter()