"""
Simple Authentication Service for Enterprise RAG System
Provides basic JWT authentication and user management
"""

import os
import jwt
import bcrypt
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    id: str
    email: str
    name: str
    subscription_tier: str = "free"
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class AuthService:
    def __init__(self, db_path: str = "auth.db"):
        self.db_path = db_path
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.init_database()
    
    def init_database(self):
        """Initialize the authentication database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                subscription_tier TEXT DEFAULT 'free',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create default demo user
        demo_user_hash = bcrypt.hashpw("demo".encode('utf-8'), bcrypt.gensalt())
        cursor.execute('''
            INSERT OR REPLACE INTO users (id, email, name, password_hash, subscription_tier)
            VALUES (?, ?, ?, ?, ?)
        ''', ("demo-user-123", "demo@example.com", "Demo User", demo_user_hash.decode('utf-8'), "free"))
        
        conn.commit()
        conn.close()
        logger.info("Authentication database initialized")
    
    def create_user(self, email: str, name: str, password: str, subscription_tier: str = "free") -> Optional[User]:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return None
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            cursor.execute('''
                INSERT INTO users (id, email, name, password_hash, subscription_tier)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, email, name, password_hash.decode('utf-8'), subscription_tier))
            
            conn.commit()
            conn.close()
            
            return User(
                id=user_id,
                email=email,
                name=name,
                subscription_tier=subscription_tier,
                created_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, password_hash, subscription_tier, created_at
                FROM users WHERE email = ?
            ''', (email,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            user_id, email, name, password_hash, subscription_tier, created_at = result
            
            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user_id,))
                conn.commit()
                conn.close()
                
                return User(
                    id=user_id,
                    email=email,
                    name=name,
                    subscription_tier=subscription_tier,
                    created_at=datetime.fromisoformat(created_at) if created_at else None,
                    last_login=datetime.now()
                )
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, subscription_tier, created_at, last_login
                FROM users WHERE id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                user_id, email, name, subscription_tier, created_at, last_login = result
                return User(
                    id=user_id,
                    email=email,
                    name=name,
                    subscription_tier=subscription_tier,
                    created_at=datetime.fromisoformat(created_at) if created_at else None,
                    last_login=datetime.fromisoformat(last_login) if last_login else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def create_jwt_token(self, user: User) -> str:
        """Create JWT token for user"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'subscription_tier': user.subscription_tier,
            'exp': datetime.utcnow() + timedelta(days=7)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

# Global auth service instance
auth_service = AuthService()