"""
Enterprise Rate Limiting System for RAG Application
"""

import time
import sqlite3
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import streamlit as st
import hashlib
from enum import Enum

class RateLimitType(Enum):
    QUERY = "query"
    LOGIN = "login"
    REGISTRATION = "registration"
    API = "api"
    UPLOAD = "upload"

@dataclass
class RateLimit:
    max_requests: int
    window_seconds: int
    block_duration_seconds: int = 300  # 5 minutes default

class RateLimiter:
    """Advanced rate limiting system with multiple strategies"""
    
    def __init__(self, db_path: str = "rate_limits.db"):
        self.db_path = db_path
        self.limits = {
            RateLimitType.QUERY: RateLimit(max_requests=50, window_seconds=3600),  # 50 queries per hour
            RateLimitType.LOGIN: RateLimit(max_requests=5, window_seconds=300, block_duration_seconds=900),  # 5 attempts per 5 min
            RateLimitType.REGISTRATION: RateLimit(max_requests=3, window_seconds=3600),  # 3 registrations per hour
            RateLimitType.API: RateLimit(max_requests=100, window_seconds=3600),  # 100 API calls per hour
            RateLimitType.UPLOAD: RateLimit(max_requests=10, window_seconds=3600)  # 10 uploads per hour
        }
        self._init_database()
    
    def _init_database(self):
        """Initialize rate limiting database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL,
                    limit_type TEXT NOT NULL,
                    request_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blocked_identifiers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL,
                    limit_type TEXT NOT NULL,
                    blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    blocked_until TIMESTAMP NOT NULL,
                    reason TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_limit_identifier_type 
                ON rate_limit_records(identifier, limit_type, request_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_blocked_identifier_type 
                ON blocked_identifiers(identifier, limit_type, blocked_until)
            """)
    
    def _get_client_identifier(self) -> str:
        """Get unique client identifier from Streamlit session"""
        # Try to get IP from Streamlit (if available)
        try:
            # In production, you might get this from headers
            session_id = st.session_state.get('session_id', 'unknown')
            if session_id == 'unknown':
                # Generate a session-based identifier
                session_id = hashlib.md5(str(id(st.session_state)).encode()).hexdigest()
                st.session_state.session_id = session_id
            return session_id
        except:
            return "default_client"
    
    def _cleanup_old_records(self, limit_type: RateLimitType):
        """Clean up old rate limit records"""
        limit = self.limits[limit_type]
        cutoff_time = datetime.now() - timedelta(seconds=limit.window_seconds * 2)  # Keep 2x window for analysis
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM rate_limit_records 
                WHERE limit_type = ? AND request_time < ?
            """, (limit_type.value, cutoff_time))
            
            # Clean up expired blocks
            conn.execute("""
                DELETE FROM blocked_identifiers 
                WHERE limit_type = ? AND blocked_until < ?
            """, (limit_type.value, datetime.now()))
    
    def is_blocked(self, identifier: str, limit_type: RateLimitType) -> Tuple[bool, Optional[datetime]]:
        """Check if identifier is currently blocked"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT blocked_until FROM blocked_identifiers 
                WHERE identifier = ? AND limit_type = ? AND blocked_until > ?
            """, (identifier, limit_type.value, datetime.now()))
            
            result = cursor.fetchone()
            if result:
                return True, datetime.fromisoformat(result[0])
            return False, None
    
    def check_rate_limit(self, limit_type: RateLimitType, identifier: str = None) -> Dict:
        """Check if request is within rate limits"""
        if identifier is None:
            identifier = self._get_client_identifier()
        
        limit = self.limits[limit_type]
        
        # Clean up old records
        self._cleanup_old_records(limit_type)
        
        # Check if blocked
        is_blocked, blocked_until = self.is_blocked(identifier, limit_type)
        if is_blocked:
            return {
                "allowed": False,
                "reason": "blocked",
                "blocked_until": blocked_until,
                "message": f"Access blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}"
            }
        
        # Count recent requests
        window_start = datetime.now() - timedelta(seconds=limit.window_seconds)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM rate_limit_records 
                WHERE identifier = ? AND limit_type = ? AND request_time > ?
            """, (identifier, limit_type.value, window_start))
            
            request_count = cursor.fetchone()[0]
            
            if request_count >= limit.max_requests:
                # Block the identifier
                blocked_until = datetime.now() + timedelta(seconds=limit.block_duration_seconds)
                conn.execute("""
                    INSERT INTO blocked_identifiers (identifier, limit_type, blocked_until, reason)
                    VALUES (?, ?, ?, ?)
                """, (identifier, limit_type.value, blocked_until, "Rate limit exceeded"))
                
                return {
                    "allowed": False,
                    "reason": "rate_limit_exceeded",
                    "requests_made": request_count,
                    "limit": limit.max_requests,
                    "window_seconds": limit.window_seconds,
                    "blocked_until": blocked_until,
                    "message": f"Rate limit exceeded. {request_count}/{limit.max_requests} requests in {limit.window_seconds}s"
                }
            
            return {
                "allowed": True,
                "requests_made": request_count,
                "limit": limit.max_requests,
                "window_seconds": limit.window_seconds,
                "remaining": limit.max_requests - request_count
            }
    
    def record_request(self, limit_type: RateLimitType, identifier: str = None, 
                      success: bool = True, ip_address: str = None, user_agent: str = None):
        """Record a request for rate limiting"""
        if identifier is None:
            identifier = self._get_client_identifier()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rate_limit_records (identifier, limit_type, ip_address, user_agent, success)
                VALUES (?, ?, ?, ?, ?)
            """, (identifier, limit_type.value, ip_address, user_agent, success))
    
    def get_rate_limit_status(self, limit_type: RateLimitType, identifier: str = None) -> Dict:
        """Get current rate limit status for identifier"""
        if identifier is None:
            identifier = self._get_client_identifier()
        
        limit = self.limits[limit_type]
        window_start = datetime.now() - timedelta(seconds=limit.window_seconds)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM rate_limit_records 
                WHERE identifier = ? AND limit_type = ? AND request_time > ?
            """, (identifier, limit_type.value, window_start))
            
            request_count = cursor.fetchone()[0]
            
            # Check if blocked
            is_blocked, blocked_until = self.is_blocked(identifier, limit_type)
            
            return {
                "identifier": identifier,
                "limit_type": limit_type.value,
                "requests_made": request_count,
                "max_requests": limit.max_requests,
                "window_seconds": limit.window_seconds,
                "remaining": max(0, limit.max_requests - request_count),
                "is_blocked": is_blocked,
                "blocked_until": blocked_until,
                "reset_time": window_start + timedelta(seconds=limit.window_seconds)
            }
    
    def unblock_identifier(self, identifier: str, limit_type: RateLimitType) -> bool:
        """Manually unblock an identifier (admin function)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM blocked_identifiers 
                    WHERE identifier = ? AND limit_type = ?
                """, (identifier, limit_type.value))
            return True
        except:
            return False
    
    def get_blocked_identifiers(self, limit_type: RateLimitType = None) -> list:
        """Get list of currently blocked identifiers"""
        with sqlite3.connect(self.db_path) as conn:
            if limit_type:
                cursor = conn.execute("""
                    SELECT identifier, limit_type, blocked_at, blocked_until, reason 
                    FROM blocked_identifiers 
                    WHERE limit_type = ? AND blocked_until > ?
                """, (limit_type.value, datetime.now()))
            else:
                cursor = conn.execute("""
                    SELECT identifier, limit_type, blocked_at, blocked_until, reason 
                    FROM blocked_identifiers 
                    WHERE blocked_until > ?
                """, (datetime.now(),))
            
            return [
                {
                    "identifier": row[0],
                    "limit_type": row[1],
                    "blocked_at": row[2],
                    "blocked_until": row[3],
                    "reason": row[4]
                }
                for row in cursor.fetchall()
            ]

# Streamlit integration decorators
def rate_limit(limit_type: RateLimitType):
    """Decorator to apply rate limiting to Streamlit functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'rate_limiter' not in st.session_state:
                st.session_state.rate_limiter = RateLimiter()
            
            rate_limiter = st.session_state.rate_limiter
            
            # Check rate limit
            status = rate_limiter.check_rate_limit(limit_type)
            
            if not status["allowed"]:
                if status["reason"] == "blocked":
                    st.error(f"🚫 Access temporarily blocked until {status['blocked_until'].strftime('%H:%M:%S')}")
                    st.info("Please wait before trying again.")
                else:
                    st.error(f"🚫 Rate limit exceeded: {status['message']}")
                    st.info(f"Limit: {status['limit']} requests per {status['window_seconds']}s")
                st.stop()
            
            # Record the request
            try:
                result = func(*args, **kwargs)
                rate_limiter.record_request(limit_type, success=True)
                return result
            except Exception as e:
                rate_limiter.record_request(limit_type, success=False)
                raise e
        
        return wrapper
    return decorator

def show_rate_limit_info(limit_type: RateLimitType):
    """Show rate limit information in Streamlit sidebar"""
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    
    rate_limiter = st.session_state.rate_limiter
    status = rate_limiter.get_rate_limit_status(limit_type)
    
    with st.sidebar:
        st.subheader("Rate Limit Status")
        
        # Progress bar
        progress = status["requests_made"] / status["max_requests"]
        st.progress(progress)
        
        # Status info
        st.text(f"Requests: {status['requests_made']}/{status['max_requests']}")
        st.text(f"Remaining: {status['remaining']}")
        st.text(f"Window: {status['window_seconds']}s")
        
        if status["is_blocked"]:
            st.error(f"🚫 Blocked until {status['blocked_until'].strftime('%H:%M:%S')}")
        elif status["requests_made"] > status["max_requests"] * 0.8:
            st.warning("⚠️ Approaching rate limit")
        else:
            st.success("✅ Within limits")