#!/usr/bin/env python3
"""
API Key Generation and Management System
Provides secure key generation, storage, and management for enterprise users
"""

import secrets
import hashlib
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import json
import logging
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

class KeyScope(Enum):
    """API Key access scopes"""
    READ_ONLY = "read_only"
    QUERY_ONLY = "query_only"
    INGEST_ONLY = "ingest_only"
    FULL_ACCESS = "full_access"
    ADMIN = "admin"

class KeyStatus(Enum):
    """API Key status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"

@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    key_prefix: str
    key_hash: str
    name: str
    description: str
    user_id: str
    scope: KeyScope
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    rate_limit: int
    rate_window: int
    metadata: Dict[str, Any]

class APIKeyManager:
    """Manages API key generation, validation, and lifecycle"""
    
    def __init__(self, db_path: str = "api_keys.db"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize the API keys database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id TEXT PRIMARY KEY,
                        key_prefix TEXT NOT NULL,
                        key_hash TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        description TEXT,
                        user_id TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP,
                        last_used TIMESTAMP,
                        usage_count INTEGER DEFAULT 0,
                        rate_limit INTEGER DEFAULT 100,
                        rate_window INTEGER DEFAULT 3600,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS key_usage_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key_id TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        FOREIGN KEY (key_id) REFERENCES api_keys (key_id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_key_hash ON api_keys(key_hash);
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id ON api_keys(user_id);
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_key_usage_timestamp ON key_usage_logs(timestamp);
                """)
                
                conn.commit()
                logger.info("API key database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize API key database: {e}")
            raise
    
    def generate_api_key(
        self,
        name: str,
        user_id: str,
        scope: KeyScope = KeyScope.QUERY_ONLY,
        description: str = "",
        expires_in_days: Optional[int] = None,
        rate_limit: int = 100,
        rate_window: int = 3600,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, APIKey]:
        """Generate a new API key"""
        try:
            # Generate secure API key
            key_id = str(uuid.uuid4())
            
            # Generate the actual key: prefix + random secure string
            prefix = "rag_"
            secure_part = secrets.token_urlsafe(32)
            full_key = f"{prefix}{secure_part}"
            
            # Hash the key for storage
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()
            
            # Calculate expiry
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Create API key object
            api_key = APIKey(
                key_id=key_id,
                key_prefix=prefix,
                key_hash=key_hash,
                name=name,
                description=description,
                user_id=user_id,
                scope=scope,
                status=KeyStatus.ACTIVE,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                last_used=None,
                usage_count=0,
                rate_limit=rate_limit,
                rate_window=rate_window,
                metadata=metadata or {}
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_keys (
                        key_id, key_prefix, key_hash, name, description, user_id,
                        scope, status, created_at, expires_at, last_used,
                        usage_count, rate_limit, rate_window, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    api_key.key_id, api_key.key_prefix, api_key.key_hash,
                    api_key.name, api_key.description, api_key.user_id,
                    api_key.scope.value, api_key.status.value,
                    api_key.created_at, api_key.expires_at, api_key.last_used,
                    api_key.usage_count, api_key.rate_limit, api_key.rate_window,
                    json.dumps(api_key.metadata)
                ))
                conn.commit()
            
            logger.info(f"Generated API key for user {user_id}: {name}")
            return full_key, api_key
            
        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            raise
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key and return key info if valid"""
        try:
            # Hash the provided key
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM api_keys WHERE key_hash = ?
                """, (key_hash,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Convert row to APIKey object
                api_key = self._row_to_api_key(row)
                
                # Check if key is valid
                if api_key.status != KeyStatus.ACTIVE:
                    return None
                
                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    # Mark as expired
                    self.update_key_status(api_key.key_id, KeyStatus.EXPIRED)
                    return None
                
                # Update last used timestamp
                self._update_last_used(api_key.key_id)
                
                return api_key
                
        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            return None
    
    def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC
                """, (user_id,))
                
                keys = []
                for row in cursor.fetchall():
                    keys.append(self._row_to_api_key(row))
                
                return keys
                
        except Exception as e:
            logger.error(f"Failed to list user keys: {e}")
            return []
    
    def get_key_info(self, key_id: str) -> Optional[APIKey]:
        """Get information about a specific API key"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM api_keys WHERE key_id = ?
                """, (key_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_api_key(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get key info: {e}")
            return None
    
    def update_key_status(self, key_id: str, status: KeyStatus) -> bool:
        """Update the status of an API key"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE api_keys SET status = ? WHERE key_id = ?
                """, (status.value, key_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update key status: {e}")
            return False
    
    def revoke_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key (user can only revoke their own keys)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE api_keys SET status = ? WHERE key_id = ? AND user_id = ?
                """, (KeyStatus.REVOKED.value, key_id, user_id))
                
                conn.commit()
                success = cursor.rowcount > 0
                
                if success:
                    logger.info(f"Revoked API key {key_id} for user {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to revoke key: {e}")
            return False
    
    def update_key_metadata(self, key_id: str, user_id: str, name: str = None, description: str = None) -> bool:
        """Update key metadata"""
        try:
            updates = []
            values = []
            
            if name is not None:
                updates.append("name = ?")
                values.append(name)
            
            if description is not None:
                updates.append("description = ?")
                values.append(description)
            
            if not updates:
                return True
            
            values.extend([key_id, user_id])
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    UPDATE api_keys SET {', '.join(updates)} WHERE key_id = ? AND user_id = ?
                """, values)
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update key metadata: {e}")
            return False
    
    def log_key_usage(
        self,
        key_id: str,
        endpoint: str,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True,
        error_message: str = None
    ):
        """Log API key usage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO key_usage_logs (
                        key_id, endpoint, timestamp, ip_address, user_agent, success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    key_id, endpoint, datetime.utcnow(), ip_address, user_agent, success, error_message
                ))
                
                # Increment usage count
                conn.execute("""
                    UPDATE api_keys SET usage_count = usage_count + 1 WHERE key_id = ?
                """, (key_id,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log key usage: {e}")
    
    def get_key_usage_stats(self, key_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for an API key"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Total usage
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM key_usage_logs 
                    WHERE key_id = ? AND timestamp > ?
                """, (key_id, cutoff_date))
                total_requests = cursor.fetchone()[0]
                
                # Success rate
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM key_usage_logs 
                    WHERE key_id = ? AND timestamp > ? AND success = 1
                """, (key_id, cutoff_date))
                successful_requests = cursor.fetchone()[0]
                
                # Most used endpoints
                cursor = conn.execute("""
                    SELECT endpoint, COUNT(*) as count FROM key_usage_logs 
                    WHERE key_id = ? AND timestamp > ?
                    GROUP BY endpoint ORDER BY count DESC LIMIT 5
                """, (key_id, cutoff_date))
                top_endpoints = [{"endpoint": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Daily usage
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count 
                    FROM key_usage_logs 
                    WHERE key_id = ? AND timestamp > ?
                    GROUP BY DATE(timestamp) ORDER BY date DESC
                """, (key_id, cutoff_date))
                daily_usage = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
                
                return {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": round(success_rate, 2),
                    "top_endpoints": top_endpoints,
                    "daily_usage": daily_usage
                }
                
        except Exception as e:
            logger.error(f"Failed to get key usage stats: {e}")
            return {}
    
    def cleanup_expired_keys(self):
        """Clean up expired keys (mark as expired)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE api_keys SET status = ? 
                    WHERE expires_at < ? AND status = ?
                """, (KeyStatus.EXPIRED.value, datetime.utcnow(), KeyStatus.ACTIVE.value))
                
                conn.commit()
                expired_count = cursor.rowcount
                
                if expired_count > 0:
                    logger.info(f"Marked {expired_count} keys as expired")
                
                return expired_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0
    
    def _row_to_api_key(self, row) -> APIKey:
        """Convert database row to APIKey object"""
        return APIKey(
            key_id=row[0],
            key_prefix=row[1],
            key_hash=row[2],
            name=row[3],
            description=row[4] or "",
            user_id=row[5],
            scope=KeyScope(row[6]),
            status=KeyStatus(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
            last_used=datetime.fromisoformat(row[10]) if row[10] else None,
            usage_count=row[11],
            rate_limit=row[12],
            rate_window=row[13],
            metadata=json.loads(row[14]) if row[14] else {}
        )
    
    def _update_last_used(self, key_id: str):
        """Update the last used timestamp for a key"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE api_keys SET last_used = ? WHERE key_id = ?
                """, (datetime.utcnow(), key_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update last used timestamp: {e}")

# Global key manager instance
key_manager = APIKeyManager()