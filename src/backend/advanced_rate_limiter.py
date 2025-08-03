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
        
        # Rate limiting configurations for different query types
        self.configs = {
            "simple": {
                "max_requests_per_minute": 20,
                "initial_backoff": 2,
                "max_backoff": 30,
                "backoff_multiplier": 2
            },
            "complex": {
                "max_requests_per_minute": 8,  # Lower for complex queries
                "initial_backoff": 5,
                "max_backoff": 120,  # Up to 2 minutes
                "backoff_multiplier": 3  # More aggressive
            },
            "quantum_physics": {  # Special category for high-level questions
                "max_requests_per_minute": 4,
                "initial_backoff": 10,
                "max_backoff": 180,  # Up to 3 minutes
                "backoff_multiplier": 4
            }
        }
        
        self.consecutive_failures = 0
        self.last_failure_time = None
        
    def categorize_query(self, query: str) -> str:
        """Categorize query complexity for appropriate rate limiting"""
        query_lower = query.lower()
        
        # High-level complex topics
        complex_indicators = [
            "quantum", "physics", "entanglement", "theorem", "principle",
            "fundamental", "theoretical", "advanced", "mathematical",
            "scientific", "research", "academic", "doctoral", "PhD",
            "comprehensive analysis", "deep dive", "implications",
            "challenge classical", "hidden variable", "locality"
        ]
        
        if any(indicator in query_lower for indicator in complex_indicators):
            if "quantum" in query_lower or "physics" in query_lower:
                return "quantum_physics"
            return "complex"
        
        return "simple"
    
    def should_allow_request(self, query: str) -> tuple[bool, float]:
        """
        Check if request should be allowed and return wait time if not
        Returns: (allow_request, wait_time_seconds)
        """
        with self.lock:
            category = self.categorize_query(query)
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
    
    def get_backoff_time(self, query: str, failure_count: int) -> float:
        """Calculate intelligent backoff time based on query complexity and failures"""
        category = self.categorize_query(query)
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
    
    def record_failure(self, query: str):
        """Record a rate limit failure for adaptive backoff"""
        with self.lock:
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
    
    def record_success(self, query: str):
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
    Execute API call with intelligent rate limiting and backoff
    """
    for attempt in range(max_retries):
        # Check if we should wait before making the request
        can_proceed, wait_time = rate_limiter.should_allow_request(query)
        
        if not can_proceed:
            logger.info(f"Rate limit: waiting {wait_time:.2f}s before attempt {attempt + 1}")
            time.sleep(wait_time)
        
        try:
            # Make the API call
            result = api_function(*args, **kwargs)
            rate_limiter.record_success(query)
            return result
            
        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str or "rate limit" in error_str.lower():
                rate_limiter.record_failure(query)
                
                if attempt < max_retries - 1:
                    backoff_time = rate_limiter.get_backoff_time(query, attempt + 1)
                    logger.info(f"Rate limit hit. Smart backoff: {backoff_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff_time)
                    continue
                else:
                    logger.error(f"All {max_retries} attempts exhausted due to rate limiting")
                    raise e
            else:
                # Non-rate-limit error, re-raise immediately
                raise e
    
    raise Exception(f"Failed after {max_retries} attempts")


# Global rate limiter instance
global_rate_limiter = AdvancedRateLimiter()