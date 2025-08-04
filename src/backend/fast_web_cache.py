"""
Fast Web Search Cache System
Provides millisecond response times for web searches through intelligent caching
"""

import time
import json
import hashlib
import sqlite3
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

@dataclass
class CachedWebResponse:
    query_hash: str
    results: List[Dict]
    confidence: float
    source: str
    created_at: datetime
    access_count: int
    last_accessed: datetime
    cache_duration_hours: int

class FastWebCache:
    """
    Ultra-fast cache system for web search results
    Provides sub-50ms response times for cached web searches
    """
    
    def __init__(self, cache_db_path: str = "cache_db/fast_web_cache.db"):
        self.cache_db_path = cache_db_path
        self.memory_cache: Dict[str, CachedWebResponse] = {}
        self.cache_lock = threading.RLock()
        self.max_memory_cache_size = 500
        
        # Cache duration patterns based on content type
        self.cache_patterns = {
            "news": 2,      # 2 hours for news/current events
            "research": 24,  # 24 hours for research topics
            "general": 12,   # 12 hours for general queries
            "trends": 1,     # 1 hour for trending topics
            "weather": 0.5,  # 30 minutes for weather
            "stocks": 0.25   # 15 minutes for financial data
        }
        
        self._initialize_cache_db()
        self._load_memory_cache()
        
        logger.info("Fast web cache initialized for speed-optimized web searches")
    
    def _initialize_cache_db(self):
        """Initialize SQLite database for persistent web caching"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fast_web_responses (
                        query_hash TEXT PRIMARY KEY,
                        query_text TEXT,
                        results TEXT,
                        confidence REAL,
                        source TEXT,
                        created_at TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP,
                        cache_duration_hours INTEGER DEFAULT 12
                    )
                """)
                
                # Create index for faster lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_web_last_accessed 
                    ON fast_web_responses(last_accessed)
                """)
                
                conn.commit()
                logger.info("Fast web cache database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize web cache database: {e}")
    
    def _load_memory_cache(self):
        """Load frequently accessed web responses into memory"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_hash, results, confidence, source, 
                           created_at, access_count, last_accessed, cache_duration_hours
                    FROM fast_web_responses 
                    ORDER BY access_count DESC, last_accessed DESC
                    LIMIT ?
                """, (self.max_memory_cache_size,))
                
                for row in cursor.fetchall():
                    query_hash, results_json, confidence, source, created_at, access_count, last_accessed, cache_duration = row
                    
                    cached_response = CachedWebResponse(
                        query_hash=query_hash,
                        results=json.loads(results_json),
                        confidence=confidence,
                        source=source,
                        created_at=datetime.fromisoformat(created_at),
                        access_count=access_count,
                        last_accessed=datetime.fromisoformat(last_accessed),
                        cache_duration_hours=cache_duration
                    )
                    
                    self.memory_cache[query_hash] = cached_response
                
                logger.info(f"Loaded {len(self.memory_cache)} web responses into memory cache")
                
        except Exception as e:
            logger.error(f"Failed to load web memory cache: {e}")
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate consistent hash for query normalization"""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_cache_duration(self, query: str) -> int:
        """Determine cache duration based on query content"""
        query_lower = query.lower()
        
        # Check for specific patterns
        if any(word in query_lower for word in ['news', 'breaking', 'today', 'latest', 'current']):
            return self.cache_patterns["news"]
        elif any(word in query_lower for word in ['stock', 'price', 'trading', 'market']):
            return self.cache_patterns["stocks"]
        elif any(word in query_lower for word in ['weather', 'temperature', 'forecast']):
            return self.cache_patterns["weather"]
        elif any(word in query_lower for word in ['trending', 'viral', 'popular']):
            return self.cache_patterns["trends"]
        elif any(word in query_lower for word in ['research', 'study', 'paper', 'academic']):
            return self.cache_patterns["research"]
        else:
            return self.cache_patterns["general"]
    
    def get_fast_web_response(self, query: str) -> Optional[Dict]:
        """
        Get ultra-fast web search response
        Returns cached response in < 50ms when available
        """
        start_time = time.time()
        query_hash = self._generate_query_hash(query)
        
        # Check memory cache first (< 10ms)
        with self.cache_lock:
            if query_hash in self.memory_cache:
                cached = self.memory_cache[query_hash]
                
                # Check if cache is still valid
                cache_expiry = cached.created_at + timedelta(hours=cached.cache_duration_hours)
                if datetime.now() < cache_expiry:
                    # Update access statistics
                    cached.access_count += 1
                    cached.last_accessed = datetime.now()
                    
                    # Update database asynchronously
                    threading.Thread(
                        target=self._update_web_access_stats,
                        args=(query_hash, cached.access_count),
                        daemon=True
                    ).start()
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"Web memory cache hit delivered in {processing_time:.1f}ms")
                    
                    return {
                        "results": cached.results,
                        "confidence": cached.confidence,
                        "source": cached.source,
                        "processing_time_ms": processing_time,
                        "cache_type": "memory",
                        "status": "success",
                        "cache_age_hours": (datetime.now() - cached.created_at).total_seconds() / 3600
                    }
                else:
                    # Cache expired, remove from memory
                    del self.memory_cache[query_hash]
        
        # Check database cache (< 50ms)
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT results, confidence, source, created_at, access_count, cache_duration_hours
                    FROM fast_web_responses 
                    WHERE query_hash = ?
                """, (query_hash,))
                
                row = cursor.fetchone()
                if row:
                    results_json, confidence, source, created_at, access_count, cache_duration = row
                    created_datetime = datetime.fromisoformat(created_at)
                    
                    # Check if cache is still valid
                    cache_expiry = created_datetime + timedelta(hours=cache_duration)
                    if datetime.now() < cache_expiry:
                        results = json.loads(results_json)
                        
                        # Add back to memory cache
                        cached_response = CachedWebResponse(
                            query_hash=query_hash,
                            results=results,
                            confidence=confidence,
                            source=source,
                            created_at=created_datetime,
                            access_count=access_count + 1,
                            last_accessed=datetime.now(),
                            cache_duration_hours=cache_duration
                        )
                        
                        with self.cache_lock:
                            self.memory_cache[query_hash] = cached_response
                        
                        # Update access stats
                        self._update_web_access_stats(query_hash, access_count + 1)
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        logger.info(f"Web database cache hit delivered in {processing_time:.1f}ms")
                        
                        return {
                            "results": results,
                            "confidence": confidence,
                            "source": source,
                            "processing_time_ms": processing_time,
                            "cache_type": "database",
                            "status": "success",
                            "cache_age_hours": (datetime.now() - created_datetime).total_seconds() / 3600
                        }
        
        except Exception as e:
            logger.error(f"Web database cache lookup failed: {e}")
        
        # No cache hit
        return None
    
    def cache_web_response(self, query: str, results: List[Dict], confidence: float, source: str):
        """Cache a web search response for future fast retrieval"""
        query_hash = self._generate_query_hash(query)
        now = datetime.now()
        cache_duration = self._get_cache_duration(query)
        
        cached_response = CachedWebResponse(
            query_hash=query_hash,
            results=results,
            confidence=confidence,
            source=source,
            created_at=now,
            access_count=1,
            last_accessed=now,
            cache_duration_hours=cache_duration
        )
        
        # Add to memory cache
        with self.cache_lock:
            self.memory_cache[query_hash] = cached_response
            
            # Maintain cache size limit
            if len(self.memory_cache) > self.max_memory_cache_size:
                # Remove least recently accessed item
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].last_accessed
                )
                del self.memory_cache[oldest_key]
        
        # Add to database
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO fast_web_responses 
                    (query_hash, query_text, results, confidence, source, created_at, access_count, last_accessed, cache_duration_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_hash,
                    query[:200],  # Store truncated query for debugging
                    json.dumps(results),
                    confidence,
                    source,
                    now.isoformat(),
                    1,
                    now.isoformat(),
                    cache_duration
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to cache web response: {e}")
    
    def _update_web_access_stats(self, query_hash: str, access_count: int):
        """Update access statistics in database"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    UPDATE fast_web_responses 
                    SET access_count = ?, last_accessed = ?
                    WHERE query_hash = ?
                """, (access_count, datetime.now().isoformat(), query_hash))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update web access stats: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get web cache performance statistics"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses,
                        MAX(access_count) as max_accesses,
                        AVG(cache_duration_hours) as avg_cache_duration
                    FROM fast_web_responses
                """)
                
                row = cursor.fetchone()
                total_entries, total_accesses, avg_accesses, max_accesses, avg_cache_duration = row
                
                return {
                    "memory_cache_size": len(self.memory_cache),
                    "database_entries": total_entries or 0,
                    "total_accesses": total_accesses or 0,
                    "average_accesses": round(avg_accesses or 0, 2),
                    "max_accesses": max_accesses or 0,
                    "average_cache_duration_hours": round(avg_cache_duration or 0, 2),
                    "cache_patterns": len(self.cache_patterns)
                }
                
        except Exception as e:
            logger.error(f"Failed to get web cache stats: {e}")
            return {
                "memory_cache_size": len(self.memory_cache),
                "database_entries": 0,
                "total_accesses": 0,
                "average_accesses": 0,
                "max_accesses": 0,
                "average_cache_duration_hours": 0,
                "cache_patterns": len(self.cache_patterns)
            }

# Global fast web cache instance
fast_web_cache = FastWebCache()