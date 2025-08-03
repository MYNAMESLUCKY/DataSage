"""
Enhanced Multi-Level Caching System for Enterprise RAG
Implements embedding cache, query result cache, and frequent pattern cache
"""

import os
import json
import logging
import hashlib
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class EnhancedCacheManager:
    """
    Multi-level caching system for enterprise RAG performance optimization
    
    Cache Levels:
    1. Query Result Cache - Stores complete query responses
    2. Embedding Cache - Stores computed embeddings for reuse
    3. Document Chunk Cache - Stores processed document chunks
    4. Web Search Cache - Stores web search results
    5. Reranking Cache - Stores reranking results for query patterns
    """
    
    def __init__(self, cache_dir: str = "cache_db"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache databases
        self.query_cache_db = self.cache_dir / "query_cache.db"
        self.embedding_cache_db = self.cache_dir / "embedding_cache.db"
        self.reranking_cache_db = self.cache_dir / "reranking_cache.db"
        
        # Cache settings
        self.query_cache_ttl = 3600  # 1 hour for query results
        self.embedding_cache_ttl = 86400 * 7  # 1 week for embeddings
        self.reranking_cache_ttl = 3600 * 6  # 6 hours for reranking
        
        # Initialize databases
        self._initialize_cache_dbs()
        
        # Performance tracking
        self.cache_stats = {
            "query_hits": 0,
            "query_misses": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "reranking_hits": 0,
            "reranking_misses": 0
        }
    
    def _initialize_cache_dbs(self):
        """Initialize SQLite databases for different cache levels"""
        try:
            # Query Results Cache
            with sqlite3.connect(self.query_cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        query_hash TEXT PRIMARY KEY,
                        original_query TEXT,
                        result_data BLOB,
                        metadata TEXT,
                        created_at TIMESTAMP,
                        expires_at TIMESTAMP,
                        access_count INTEGER DEFAULT 1,
                        last_accessed TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON query_cache(expires_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON query_cache(last_accessed)")
            
            # Embedding Cache
            with sqlite3.connect(self.embedding_cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        content_hash TEXT PRIMARY KEY,
                        content_preview TEXT,
                        embedding_data BLOB,
                        embedding_model TEXT,
                        created_at TIMESTAMP,
                        expires_at TIMESTAMP,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_expires ON embedding_cache(expires_at)")
            
            # Reranking Cache
            with sqlite3.connect(self.reranking_cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reranking_cache (
                        query_docs_hash TEXT PRIMARY KEY,
                        query_text TEXT,
                        doc_count INTEGER,
                        ranking_data BLOB,
                        created_at TIMESTAMP,
                        expires_at TIMESTAMP,
                        access_count INTEGER DEFAULT 1
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reranking_expires ON reranking_cache(expires_at)")
            
            logger.info("Enhanced cache databases initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache databases: {e}")
    
    def _generate_query_hash(self, query: str, parameters: Dict[str, Any] = None) -> str:
        """Generate a hash for query caching"""
        cache_key = f"{query.lower().strip()}"
        if parameters:
            # Include relevant parameters in cache key
            param_str = json.dumps(parameters, sort_keys=True)
            cache_key += f"||{param_str}"
        
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content-based caching"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.now()
            
            # Clean query cache
            with sqlite3.connect(self.query_cache_db) as conn:
                deleted = conn.execute("DELETE FROM query_cache WHERE expires_at < ?", (current_time,)).rowcount
                if deleted > 0:
                    logger.info(f"Cleaned {deleted} expired query cache entries")
            
            # Clean embedding cache
            with sqlite3.connect(self.embedding_cache_db) as conn:
                deleted = conn.execute("DELETE FROM embedding_cache WHERE expires_at < ?", (current_time,)).rowcount
                if deleted > 0:
                    logger.info(f"Cleaned {deleted} expired embedding cache entries")
            
            # Clean reranking cache
            with sqlite3.connect(self.reranking_cache_db) as conn:
                deleted = conn.execute("DELETE FROM reranking_cache WHERE expires_at < ?", (current_time,)).rowcount
                if deleted > 0:
                    logger.info(f"Cleaned {deleted} expired reranking cache entries")
                    
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def cache_query_result(self, query: str, result: Dict[str, Any], 
                          parameters: Dict[str, Any] = None, ttl: int = None) -> bool:
        """Cache a complete query result"""
        try:
            query_hash = self._generate_query_hash(query, parameters)
            
            if ttl is None:
                ttl = self.query_cache_ttl
            
            current_time = datetime.now()
            expires_at = current_time + timedelta(seconds=ttl)
            
            # Serialize result data
            result_data = pickle.dumps(result)
            metadata = json.dumps(parameters or {})
            
            with sqlite3.connect(self.query_cache_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO query_cache 
                    (query_hash, original_query, result_data, metadata, created_at, expires_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (query_hash, query, result_data, metadata, current_time, expires_at, current_time))
            
            logger.debug(f"Cached query result for: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache query result: {e}")
            return False
    
    def get_cached_query_result(self, query: str, parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached query result"""
        try:
            query_hash = self._generate_query_hash(query, parameters)
            current_time = datetime.now()
            
            with sqlite3.connect(self.query_cache_db) as conn:
                cursor = conn.execute("""
                    SELECT result_data, access_count FROM query_cache 
                    WHERE query_hash = ? AND expires_at > ?
                """, (query_hash, current_time))
                
                row = cursor.fetchone()
                if row:
                    result_data, access_count = row
                    
                    # Update access statistics
                    conn.execute("""
                        UPDATE query_cache 
                        SET access_count = ?, last_accessed = ?
                        WHERE query_hash = ?
                    """, (access_count + 1, current_time, query_hash))
                    
                    # Deserialize and return result
                    result = pickle.loads(result_data)
                    self.cache_stats["query_hits"] += 1
                    logger.debug(f"Query cache HIT for: {query[:50]}...")
                    return result
                else:
                    self.cache_stats["query_misses"] += 1
                    logger.debug(f"Query cache MISS for: {query[:50]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve cached query result: {e}")
            self.cache_stats["query_misses"] += 1
            return None
    
    def cache_embedding(self, content: str, embedding: List[float], model: str = "default") -> bool:
        """Cache computed embedding for reuse"""
        try:
            content_hash = self._generate_content_hash(content)
            current_time = datetime.now()
            expires_at = current_time + timedelta(seconds=self.embedding_cache_ttl)
            
            # Serialize embedding
            embedding_data = pickle.dumps(embedding)
            content_preview = content[:200]  # Store preview for debugging
            
            with sqlite3.connect(self.embedding_cache_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO embedding_cache 
                    (content_hash, content_preview, embedding_data, embedding_model, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (content_hash, content_preview, embedding_data, model, current_time, expires_at))
            
            logger.debug(f"Cached embedding for content: {content[:30]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False
    
    def get_cached_embedding(self, content: str, model: str = "default") -> Optional[List[float]]:
        """Retrieve cached embedding"""
        try:
            content_hash = self._generate_content_hash(content)
            current_time = datetime.now()
            
            with sqlite3.connect(self.embedding_cache_db) as conn:
                cursor = conn.execute("""
                    SELECT embedding_data, access_count FROM embedding_cache 
                    WHERE content_hash = ? AND embedding_model = ? AND expires_at > ?
                """, (content_hash, model, current_time))
                
                row = cursor.fetchone()
                if row:
                    embedding_data, access_count = row
                    
                    # Update access count
                    conn.execute("""
                        UPDATE embedding_cache 
                        SET access_count = ?
                        WHERE content_hash = ? AND embedding_model = ?
                    """, (access_count + 1, content_hash, model))
                    
                    embedding = pickle.loads(embedding_data)
                    self.cache_stats["embedding_hits"] += 1
                    logger.debug(f"Embedding cache HIT for: {content[:30]}...")
                    return embedding
                else:
                    self.cache_stats["embedding_misses"] += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve cached embedding: {e}")
            self.cache_stats["embedding_misses"] += 1
            return None
    
    def cache_reranking_result(self, query: str, doc_hashes: List[str], 
                              ranking_result: List[Tuple[Any, float]]) -> bool:
        """Cache reranking results for similar query patterns"""
        try:
            # Create hash from query + document set
            doc_set_str = "|".join(sorted(doc_hashes))
            query_docs_key = f"{query.lower().strip()}||{doc_set_str}"
            query_docs_hash = hashlib.sha256(query_docs_key.encode()).hexdigest()
            
            current_time = datetime.now()
            expires_at = current_time + timedelta(seconds=self.reranking_cache_ttl)
            
            # Serialize ranking data (store doc hashes with scores)
            ranking_data = []
            for doc, score in ranking_result:
                doc_hash = self._generate_content_hash(getattr(doc, 'page_content', str(doc))[:500])
                ranking_data.append((doc_hash, score))
            
            serialized_ranking = pickle.dumps(ranking_data)
            
            with sqlite3.connect(self.reranking_cache_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO reranking_cache 
                    (query_docs_hash, query_text, doc_count, ranking_data, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (query_docs_hash, query, len(doc_hashes), serialized_ranking, current_time, expires_at))
            
            logger.debug(f"Cached reranking result for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache reranking result: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = self.cache_stats.copy()
            
            # Calculate hit rates
            total_query = stats["query_hits"] + stats["query_misses"]
            total_embedding = stats["embedding_hits"] + stats["embedding_misses"]
            total_reranking = stats["reranking_hits"] + stats["reranking_misses"]
            
            stats["query_hit_rate"] = stats["query_hits"] / total_query if total_query > 0 else 0
            stats["embedding_hit_rate"] = stats["embedding_hits"] / total_embedding if total_embedding > 0 else 0
            stats["reranking_hit_rate"] = stats["reranking_hits"] / total_reranking if total_reranking > 0 else 0
            
            # Get database sizes
            with sqlite3.connect(self.query_cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM query_cache")
                stats["query_cache_entries"] = cursor.fetchone()[0]
            
            with sqlite3.connect(self.embedding_cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embedding_cache")
                stats["embedding_cache_entries"] = cursor.fetchone()[0]
            
            with sqlite3.connect(self.reranking_cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM reranking_cache")
                stats["reranking_cache_entries"] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return self.cache_stats.copy()
    
    def optimize_cache(self):
        """Optimize cache performance by cleaning up and reorganizing"""
        try:
            logger.info("Starting cache optimization...")
            
            # Clean up expired entries
            self._cleanup_expired_cache()
            
            # Remove least recently used entries if cache is too large
            self._cleanup_lru_entries()
            
            # Vacuum databases to reclaim space
            for db_path in [self.query_cache_db, self.embedding_cache_db, self.reranking_cache_db]:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("VACUUM")
            
            logger.info("Cache optimization completed")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    def _cleanup_lru_entries(self, max_entries: int = 10000):
        """Clean up least recently used entries if cache grows too large"""
        try:
            with sqlite3.connect(self.query_cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM query_cache")
                count = cursor.fetchone()[0]
                
                if count > max_entries:
                    # Remove oldest 20% of entries
                    remove_count = int(count * 0.2)
                    conn.execute("""
                        DELETE FROM query_cache 
                        WHERE query_hash IN (
                            SELECT query_hash FROM query_cache 
                            ORDER BY last_accessed ASC 
                            LIMIT ?
                        )
                    """, (remove_count,))
                    logger.info(f"Removed {remove_count} LRU query cache entries")
                    
        except Exception as e:
            logger.error(f"LRU cleanup failed: {e}")
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear specific cache or all caches"""
        try:
            if cache_type in ["all", "query"]:
                with sqlite3.connect(self.query_cache_db) as conn:
                    conn.execute("DELETE FROM query_cache")
                logger.info("Query cache cleared")
            
            if cache_type in ["all", "embedding"]:
                with sqlite3.connect(self.embedding_cache_db) as conn:
                    conn.execute("DELETE FROM embedding_cache")
                logger.info("Embedding cache cleared")
            
            if cache_type in ["all", "reranking"]:
                with sqlite3.connect(self.reranking_cache_db) as conn:
                    conn.execute("DELETE FROM reranking_cache")
                logger.info("Reranking cache cleared")
            
            # Reset stats
            if cache_type == "all":
                self.cache_stats = {key: 0 for key in self.cache_stats}
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

# Global cache instance
_cache_manager = None

def get_cache_manager() -> EnhancedCacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = EnhancedCacheManager()
    return _cache_manager