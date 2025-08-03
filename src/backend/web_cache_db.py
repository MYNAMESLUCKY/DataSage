import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

Base = declarative_base()

class WebSearchCache(Base):
    """Database model for caching web search results"""
    __tablename__ = 'web_search_cache'
    
    id = Column(String, primary_key=True)
    query_hash = Column(String, index=True, nullable=False)
    original_query = Column(Text, nullable=False)
    search_results = Column(JSON, nullable=False)  # Store search results as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    access_count = Column(Integer, default=1)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    relevance_score = Column(Float, default=0.0)

class ProcessedWebContent(Base):
    """Database model for processed and chunked web content"""
    __tablename__ = 'processed_web_content'
    
    id = Column(String, primary_key=True)
    url = Column(String, index=True, nullable=False)
    title = Column(Text)
    content = Column(Text, nullable=False)
    content_hash = Column(String, index=True)
    source_query = Column(Text)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_validated = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    quality_score = Column(Float, default=0.0)

@dataclass
class CachedWebResult:
    """Data class for cached web results"""
    id: str
    query: str
    results: List[Dict[str, Any]]
    created_at: datetime
    access_count: int
    relevance_score: float

class WebCacheDatabase:
    """
    Database manager for caching web search results and processed content
    """
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.is_connected = False
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                logger.warning("DATABASE_URL not found, web caching disabled")
                return
            
            self.engine = create_engine(database_url, echo=False)
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            self.is_connected = True
            logger.info("Web cache database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize web cache database: {e}")
            self.is_connected = False
    
    def _get_session(self) -> Optional[Session]:
        """Get database session"""
        if not self.is_connected:
            return None
        return self.session_factory()
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query to use as cache key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def cache_search_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        relevance_score: float = 0.0
    ) -> bool:
        """Cache web search results"""
        if not self.is_connected:
            return False
        
        session = self._get_session()
        if not session:
            return False
        
        try:
            query_hash = self._generate_query_hash(query)
            cache_id = f"cache_{query_hash}_{int(datetime.utcnow().timestamp())}"
            
            # Check if similar query exists (within last 24 hours)
            existing = session.query(WebSearchCache).filter(
                WebSearchCache.query_hash == query_hash,
                WebSearchCache.created_at > datetime.utcnow() - timedelta(hours=24),
                WebSearchCache.is_active == True
            ).first()
            
            if existing:
                # Update existing cache
                existing.access_count += 1
                existing.last_accessed = datetime.utcnow()
                existing.updated_at = datetime.utcnow()
                existing.search_results = results
                logger.info(f"Updated existing cache for query: {query[:50]}...")
            else:
                # Create new cache entry
                cache_entry = WebSearchCache(
                    id=cache_id,
                    query_hash=query_hash,
                    original_query=query,
                    search_results=results,
                    relevance_score=relevance_score
                )
                session.add(cache_entry)
                logger.info(f"Cached new search results for: {query[:50]}...")
            
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to cache search results: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_cached_results(
        self, 
        query: str, 
        max_age_hours: int = 24
    ) -> Optional[CachedWebResult]:
        """Retrieve cached search results"""
        if not self.is_connected:
            return None
        
        session = self._get_session()
        if not session:
            return None
        
        try:
            query_hash = self._generate_query_hash(query)
            
            cached = session.query(WebSearchCache).filter(
                WebSearchCache.query_hash == query_hash,
                WebSearchCache.created_at > datetime.utcnow() - timedelta(hours=max_age_hours),
                WebSearchCache.is_active == True
            ).order_by(WebSearchCache.updated_at.desc()).first()
            
            if cached:
                # Update access statistics
                cached.access_count += 1
                cached.last_accessed = datetime.utcnow()
                session.commit()
                
                logger.info(f"Retrieved cached results for: {query[:50]}...")
                
                return CachedWebResult(
                    id=cached.id,
                    query=cached.original_query,
                    results=cached.search_results,
                    created_at=cached.created_at,
                    access_count=cached.access_count,
                    relevance_score=cached.relevance_score
                )
            
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve cached results: {e}")
            return None
        finally:
            session.close()
    
    def cache_processed_content(
        self, 
        url: str, 
        title: str, 
        content: str, 
        source_query: str,
        metadata: Optional[Dict[str, Any]] = None,
        quality_score: float = 0.0
    ) -> bool:
        """Cache processed web content"""
        if not self.is_connected:
            return False
        
        session = self._get_session()
        if not session:
            return False
        
        try:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_id = f"content_{content_hash}_{int(datetime.utcnow().timestamp())}"
            
            # Check if content already exists
            existing = session.query(ProcessedWebContent).filter(
                ProcessedWebContent.content_hash == content_hash,
                ProcessedWebContent.is_active == True
            ).first()
            
            if existing:
                # Update timestamp
                existing.last_validated = datetime.utcnow()
                logger.info(f"Content already cached for URL: {url}")
            else:
                # Create new content entry
                content_entry = ProcessedWebContent(
                    id=content_id,
                    url=url,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    source_query=source_query,
                    meta_data=metadata or {},
                    quality_score=quality_score
                )
                session.add(content_entry)
                logger.info(f"Cached processed content for: {url}")
            
            session.commit()
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to cache processed content: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def search_cached_content(
        self, 
        query: str, 
        limit: int = 10,
        min_quality_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search through cached content"""
        if not self.is_connected:
            return []
        
        session = self._get_session()
        if not session:
            return []
        
        try:
            # Simple text search in cached content
            search_term = f"%{query.lower()}%"
            
            results = session.query(ProcessedWebContent).filter(
                ProcessedWebContent.is_active == True,
                ProcessedWebContent.quality_score >= min_quality_score,
                ProcessedWebContent.content.ilike(search_term)
            ).order_by(ProcessedWebContent.quality_score.desc()).limit(limit).all()
            
            cached_content = []
            for result in results:
                cached_content.append({
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:1000] + "..." if len(result.content) > 1000 else result.content,
                    "source_query": result.source_query,
                    "quality_score": result.quality_score,
                    "created_at": result.created_at.isoformat(),
                    "metadata": result.meta_data
                })
            
            logger.info(f"Found {len(cached_content)} cached content items for query: {query[:50]}...")
            return cached_content
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to search cached content: {e}")
            return []
        finally:
            session.close()
    
    def cleanup_old_cache(self, days_old: int = 7) -> int:
        """Clean up old cache entries"""
        if not self.is_connected:
            return 0
        
        session = self._get_session()
        if not session:
            return 0
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Clean old search cache
            old_searches = session.query(WebSearchCache).filter(
                WebSearchCache.last_accessed < cutoff_date
            ).count()
            
            session.query(WebSearchCache).filter(
                WebSearchCache.last_accessed < cutoff_date
            ).delete()
            
            # Clean old content
            old_content = session.query(ProcessedWebContent).filter(
                ProcessedWebContent.last_validated < cutoff_date
            ).count()
            
            session.query(ProcessedWebContent).filter(
                ProcessedWebContent.last_validated < cutoff_date
            ).delete()
            
            session.commit()
            
            total_cleaned = old_searches + old_content
            logger.info(f"Cleaned up {total_cleaned} old cache entries")
            return total_cleaned
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old cache: {e}")
            session.rollback()
            return 0
        finally:
            session.close()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        if not self.is_connected:
            return {"status": "disconnected"}
        
        session = self._get_session()
        if not session:
            return {"status": "no_session"}
        
        try:
            search_count = session.query(WebSearchCache).filter(WebSearchCache.is_active == True).count()
            content_count = session.query(ProcessedWebContent).filter(ProcessedWebContent.is_active == True).count()
            
            recent_searches = session.query(WebSearchCache).filter(
                WebSearchCache.created_at > datetime.utcnow() - timedelta(hours=24)
            ).count()
            
            total_access_count = session.query(WebSearchCache).with_entities(
                WebSearchCache.access_count.sum()
            ).scalar() or 0
            
            return {
                "status": "connected",
                "total_cached_searches": search_count,
                "total_cached_content": content_count,
                "recent_searches_24h": recent_searches,
                "total_cache_hits": total_access_count,
                "database_url_configured": bool(os.getenv("DATABASE_URL"))
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            session.close()