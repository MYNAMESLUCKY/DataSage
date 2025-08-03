"""
Intelligent caching system for RAG query results
Implements LRU cache with TTL and performance tracking
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheItem:
    result: Dict[str, Any]
    timestamp: float
    hits: int = 0
    last_accessed: float = 0

class QueryCacheManager:
    """
    Advanced caching system for query results with intelligent invalidation
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, CacheItem] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.total_requests = 0
        self.cache_hits = 0
        
    def _generate_cache_key(self, query: str, model: str, context_fingerprint: str = "") -> str:
        """Generate unique cache key for query + model + context"""
        # Normalize query for better cache hits
        normalized_query = query.lower().strip()
        content = f"{normalized_query}|{model}|{context_fingerprint}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, query: str, model: str, context_fingerprint: str = "") -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available and not expired"""
        self.total_requests += 1
        cache_key = self._generate_cache_key(query, model, context_fingerprint)
        
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            current_time = time.time()
            
            # Check if expired
            if current_time - cached_item.timestamp < self.ttl_seconds:
                # Update access statistics
                cached_item.hits += 1
                cached_item.last_accessed = current_time
                self.cache_hits += 1
                
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_item.result
            else:
                # Remove expired item
                logger.debug(f"Cache expired for query: {query[:50]}...")
                del self.cache[cache_key]
        
        return None
    
    def set(self, query: str, model: str, result: Dict[str, Any], context_fingerprint: str = ""):
        """Cache query result with LRU eviction if needed"""
        cache_key = self._generate_cache_key(query, model, context_fingerprint)
        current_time = time.time()
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self._evict_lru_items(1)
        
        # Cache the result
        self.cache[cache_key] = CacheItem(
            result=result,
            timestamp=current_time,
            last_accessed=current_time
        )
        
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def _evict_lru_items(self, count: int = 1):
        """Remove least recently used items"""
        if not self.cache:
            return
        
        # Sort by last_accessed time and remove oldest
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for i in range(min(count, len(sorted_items))):
            key_to_remove = sorted_items[i][0]
            del self.cache[key_to_remove]
            logger.debug(f"Evicted cache item: {key_to_remove}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern"""
        keys_to_remove = []
        for key in self.cache.keys():
            if pattern in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
    
    def clear_expired(self):
        """Remove all expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.cache.items():
            if current_time - item.timestamp >= self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        current_time = time.time()
        
        # Calculate cache size distribution
        size_stats = {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'usage_percentage': (len(self.cache) / self.max_size) * 100 if self.max_size > 0 else 0
        }
        
        # Calculate hit rate
        hit_rate = (self.cache_hits / self.total_requests) * 100 if self.total_requests > 0 else 0
        
        # Calculate age distribution
        ages = [current_time - item.timestamp for item in self.cache.values()]
        avg_age = sum(ages) / len(ages) if ages else 0
        
        # Find most popular entries
        popular_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].hits,
            reverse=True
        )[:5]
        
        return {
            'performance': {
                'hit_rate_percentage': hit_rate,
                'total_requests': self.total_requests,
                'cache_hits': self.cache_hits,
                'cache_misses': self.total_requests - self.cache_hits
            },
            'storage': size_stats,
            'content': {
                'average_age_seconds': avg_age,
                'oldest_entry_age': max(ages) if ages else 0,
                'most_popular_entries': len(popular_entries)
            },
            'ttl_seconds': self.ttl_seconds
        }
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cached queries"""
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].hits,
            reverse=True
        )
        
        popular = []
        for i, (key, item) in enumerate(sorted_items[:limit]):
            # Try to extract readable info from cache key (limited)
            popular.append({
                'rank': i + 1,
                'cache_key': key,
                'hits': item.hits,
                'age_seconds': time.time() - item.timestamp,
                'last_accessed_ago': time.time() - item.last_accessed
            })
        
        return popular

class ContextManager:
    """
    Manages document context fingerprinting for intelligent cache invalidation
    """
    
    def __init__(self):
        self.document_hashes: Dict[str, str] = {}
        self.last_context_hash = ""
        
    def update_document_context(self, documents: List[Dict[str, Any]]):
        """Update document context and generate new fingerprint"""
        # Create hash of all document IDs and their modification times
        doc_fingerprints = []
        
        for doc in documents:
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            mod_time = doc.get('modified_time', time.time())
            content_hash = doc.get('content_hash', '')
            
            doc_fingerprint = f"{doc_id}:{mod_time}:{content_hash}"
            doc_fingerprints.append(doc_fingerprint)
        
        # Sort for consistent hashing regardless of order
        doc_fingerprints.sort()
        
        # Generate context fingerprint
        context_content = "|".join(doc_fingerprints)
        new_context_hash = hashlib.md5(context_content.encode()).hexdigest()[:12]
        
        # Check if context changed
        context_changed = new_context_hash != self.last_context_hash
        self.last_context_hash = new_context_hash
        
        return new_context_hash, context_changed
    
    def get_current_fingerprint(self) -> str:
        """Get current context fingerprint"""
        return self.last_context_hash

# Global cache instance
_global_cache = None

def get_cache_manager() -> QueryCacheManager:
    """Get global cache manager instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCacheManager(max_size=1000, ttl_seconds=3600)
    return _global_cache

def clear_cache():
    """Clear global cache"""
    global _global_cache
    if _global_cache:
        _global_cache.cache.clear()
        _global_cache.total_requests = 0
        _global_cache.cache_hits = 0
        logger.info("Global cache cleared")