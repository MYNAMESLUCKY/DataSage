"""
Fast Response Cache System for Simple Queries
Provides millisecond response times for common questions
"""

import time
import json
import hashlib
import sqlite3
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    query_hash: str
    response: str
    confidence: float
    sources: List[str]
    created_at: datetime
    access_count: int
    last_accessed: datetime

class FastResponseCache:
    """
    Ultra-fast cache system for simple queries
    Provides sub-100ms response times for common questions
    """
    
    def __init__(self, cache_db_path: str = "cache_db/fast_response_cache.db"):
        self.cache_db_path = cache_db_path
        self.memory_cache: Dict[str, CachedResponse] = {}
        self.cache_lock = threading.RLock()
        self.max_memory_cache_size = 1000
        self.cache_ttl_hours = 24
        
        # Keep instant responses minimal - only for the most basic definitional queries
        # Most queries should use the knowledge base for accurate, up-to-date content
        self.instant_responses = {
            "what is ai": {
                "response": """# Artificial Intelligence (AI)

## Definition
Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and decision-making.

## Key Components
- **Machine Learning**: Systems that improve through experience
- **Natural Language Processing**: Understanding and generating human language
- **Computer Vision**: Interpreting visual information
- **Robotics**: Physical interaction with environments
- **Expert Systems**: Knowledge-based decision making

## Applications
- Virtual assistants (Siri, Alexa)
- Recommendation systems (Netflix, Amazon)
- Autonomous vehicles
- Medical diagnosis
- Financial fraud detection
- Content creation and analysis

## Types of AI
1. **Narrow AI**: Specialized for specific tasks (current technology)
2. **General AI**: Human-level intelligence across all domains (future goal)
3. **Superintelligence**: Beyond human cognitive abilities (theoretical)

AI works by processing large amounts of data to identify patterns and make predictions or decisions based on learned information.""",
                "confidence": 0.95,
                "sources": ["AI Knowledge Base", "Computer Science Fundamentals"]
            },
            "what is machine learning": {
                "response": """# Machine Learning

## Definition
Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.

## How It Works
1. **Data Input**: Large datasets are fed into algorithms
2. **Pattern Recognition**: Algorithms identify patterns and relationships
3. **Model Training**: Systems learn from examples
4. **Prediction**: Trained models make predictions on new data

## Types of Machine Learning
- **Supervised Learning**: Learning with labeled examples
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through trial and error with rewards

## Common Applications
- Email spam filtering
- Image recognition
- Speech recognition
- Recommendation engines
- Predictive analytics
- Autonomous systems

## Key Algorithms
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
- Random Forest
- Deep Learning

Machine learning powers many everyday technologies and continues to advance rapidly across industries.""",
                "confidence": 0.95,
                "sources": ["ML Knowledge Base", "Data Science Fundamentals"]
            },
            "what is deep learning": {
                "response": """# Deep Learning

## Definition
Deep Learning is a specialized subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data.

## Architecture
- **Neural Networks**: Inspired by biological brain structure
- **Multiple Layers**: Input, hidden, and output layers
- **Neurons**: Processing units that activate based on inputs
- **Weights and Biases**: Parameters that determine network behavior

## Key Features
- **Automatic Feature Extraction**: No manual feature engineering required
- **Hierarchical Learning**: Each layer learns increasingly complex patterns
- **Large Data Requirements**: Needs substantial datasets for training
- **Computational Intensity**: Requires significant processing power

## Applications
- **Computer Vision**: Image classification, object detection
- **Natural Language Processing**: Translation, text generation
- **Speech Recognition**: Voice assistants, transcription
- **Game Playing**: Chess, Go, video games
- **Generative AI**: Creating text, images, music

## Popular Architectures
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers
- Generative Adversarial Networks (GANs)

Deep learning has revolutionized AI capabilities, enabling breakthrough performance in many domains previously thought impossible for machines.""",
                "confidence": 0.95,
                "sources": ["Deep Learning Knowledge Base", "Neural Network Fundamentals"]
            }
        }
        
        self._initialize_cache_db()
        self._load_memory_cache()
        
        logger.info("Fast response cache initialized with instant responses for common queries")
    
    def _initialize_cache_db(self):
        """Initialize SQLite database for persistent caching"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fast_responses (
                        query_hash TEXT PRIMARY KEY,
                        query_text TEXT,
                        response TEXT,
                        confidence REAL,
                        sources TEXT,
                        created_at TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP
                    )
                """)
                
                # Create index for faster lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed 
                    ON fast_responses(last_accessed)
                """)
                
                conn.commit()
                logger.info("Fast response cache database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
    
    def _load_memory_cache(self):
        """Load frequently accessed responses into memory"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_hash, response, confidence, sources, 
                           created_at, access_count, last_accessed
                    FROM fast_responses 
                    ORDER BY access_count DESC, last_accessed DESC
                    LIMIT ?
                """, (self.max_memory_cache_size,))
                
                for row in cursor.fetchall():
                    query_hash, response, confidence, sources_json, created_at, access_count, last_accessed = row
                    
                    cached_response = CachedResponse(
                        query_hash=query_hash,
                        response=response,
                        confidence=confidence,
                        sources=json.loads(sources_json),
                        created_at=datetime.fromisoformat(created_at),
                        access_count=access_count,
                        last_accessed=datetime.fromisoformat(last_accessed)
                    )
                    
                    self.memory_cache[query_hash] = cached_response
                
                logger.info(f"Loaded {len(self.memory_cache)} responses into memory cache")
                
        except Exception as e:
            logger.error(f"Failed to load memory cache: {e}")
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate consistent hash for query normalization"""
        # Normalize query: lowercase, strip, remove extra spaces
        normalized = " ".join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_fast_response(self, query: str) -> Optional[Dict]:
        """
        Get ultra-fast response for simple queries
        Returns response in < 100ms for cached queries
        """
        start_time = time.time()
        
        # Check instant responses first (sub-millisecond)
        normalized_query = query.lower().strip()
        if normalized_query in self.instant_responses:
            response_data = self.instant_responses[normalized_query]
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Instant response delivered in {processing_time:.1f}ms")
            
            return {
                "answer": response_data["response"],
                "confidence": response_data["confidence"],
                "sources": response_data["sources"],
                "processing_time_ms": processing_time,
                "cache_type": "instant",
                "status": "success"
            }
        
        # Check memory cache (< 10ms)
        query_hash = self._generate_query_hash(query)
        
        with self.cache_lock:
            if query_hash in self.memory_cache:
                cached = self.memory_cache[query_hash]
                
                # Check if cache is still valid
                if datetime.now() - cached.created_at < timedelta(hours=self.cache_ttl_hours):
                    # Update access statistics
                    cached.access_count += 1
                    cached.last_accessed = datetime.now()
                    
                    # Update database asynchronously
                    threading.Thread(
                        target=self._update_access_stats,
                        args=(query_hash, cached.access_count),
                        daemon=True
                    ).start()
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"Memory cache hit delivered in {processing_time:.1f}ms")
                    
                    return {
                        "answer": cached.response,
                        "confidence": cached.confidence,
                        "sources": cached.sources,
                        "processing_time_ms": processing_time,
                        "cache_type": "memory",
                        "status": "success"
                    }
                else:
                    # Cache expired, remove from memory
                    del self.memory_cache[query_hash]
        
        # Check database cache (< 50ms)
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT response, confidence, sources, created_at, access_count
                    FROM fast_responses 
                    WHERE query_hash = ?
                """, (query_hash,))
                
                row = cursor.fetchone()
                if row:
                    response, confidence, sources_json, created_at, access_count = row
                    created_datetime = datetime.fromisoformat(created_at)
                    
                    # Check if cache is still valid
                    if datetime.now() - created_datetime < timedelta(hours=self.cache_ttl_hours):
                        # Add back to memory cache
                        cached_response = CachedResponse(
                            query_hash=query_hash,
                            response=response,
                            confidence=confidence,
                            sources=json.loads(sources_json),
                            created_at=created_datetime,
                            access_count=access_count + 1,
                            last_accessed=datetime.now()
                        )
                        
                        with self.cache_lock:
                            self.memory_cache[query_hash] = cached_response
                        
                        # Update access stats
                        self._update_access_stats(query_hash, access_count + 1)
                        
                        processing_time = (time.time() - start_time) * 1000
                        
                        logger.info(f"Database cache hit delivered in {processing_time:.1f}ms")
                        
                        return {
                            "answer": response,
                            "confidence": confidence,
                            "sources": json.loads(sources_json),
                            "processing_time_ms": processing_time,
                            "cache_type": "database",
                            "status": "success"
                        }
        
        except Exception as e:
            logger.error(f"Database cache lookup failed: {e}")
        
        # No cache hit
        return None
    
    def cache_response(self, query: str, response: str, confidence: float, sources: List[str]):
        """Cache a response for future fast retrieval"""
        query_hash = self._generate_query_hash(query)
        now = datetime.now()
        
        cached_response = CachedResponse(
            query_hash=query_hash,
            response=response,
            confidence=confidence,
            sources=sources,
            created_at=now,
            access_count=1,
            last_accessed=now
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
                    INSERT OR REPLACE INTO fast_responses 
                    (query_hash, query_text, response, confidence, sources, created_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_hash,
                    query[:200],  # Store truncated query for debugging
                    response,
                    confidence,
                    json.dumps(sources),
                    now.isoformat(),
                    1,
                    now.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def _update_access_stats(self, query_hash: str, access_count: int):
        """Update access statistics in database"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    UPDATE fast_responses 
                    SET access_count = ?, last_accessed = ?
                    WHERE query_hash = ?
                """, (access_count, datetime.now().isoformat(), query_hash))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update access stats: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses,
                        MAX(access_count) as max_accesses
                    FROM fast_responses
                """)
                
                row = cursor.fetchone()
                total_entries, total_accesses, avg_accesses, max_accesses = row
                
                return {
                    "memory_cache_size": len(self.memory_cache),
                    "database_entries": total_entries or 0,
                    "total_accesses": total_accesses or 0,
                    "average_accesses": round(avg_accesses or 0, 2),
                    "max_accesses": max_accesses or 0,
                    "instant_responses": len(self.instant_responses)
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "memory_cache_size": len(self.memory_cache),
                "database_entries": 0,
                "total_accesses": 0,
                "average_accesses": 0,
                "max_accesses": 0,
                "instant_responses": len(self.instant_responses)
            }

# Global cache instance
fast_cache = FastResponseCache()