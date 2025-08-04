"""
Enterprise Authentication System for RAG Application
"""

# Removed problematic imports
import os
import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional, Tuple
import jwt
from datetime import datetime, timedelta
import sqlite3
import streamlit as st
from dataclasses import dataclass
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

@dataclass
class User:
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class AuthenticationSystem:
    """Secure authentication system with JWT tokens and rate limiting"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'users.db')
        self.db_path = db_path
        self.jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.token_expiry_hours = 24
        self.rate_limit_window = 300  # 5 minutes
        self.max_login_attempts = 5
        self._init_database()
    
    def _init_database(self):
        """Initialize authentication database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    ip_address TEXT,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Securely hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return password_hash.hex(), salt
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        password_hash, _ = self._hash_password(password, salt)
        return hmac.compare_digest(password_hash, stored_hash)
    
    def _generate_jwt_token(self, username: str, role: str) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'username': username,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def _check_rate_limit(self, username: str, ip_address: Optional[str] = None) -> bool:
        """Check if user has exceeded login attempt rate limit"""
        cutoff_time = datetime.now() - timedelta(seconds=self.rate_limit_window)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM login_attempts 
                WHERE username = ? AND attempt_time > ? AND success = 0
            """, (username, cutoff_time))
            
            failed_attempts = cursor.fetchone()[0]
            return failed_attempts < self.max_login_attempts
    
    def _has_any_users(self) -> bool:
        """Check if any users exist in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            return count > 0
    
    def _get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email address"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT username, email, role, created_at, last_login, is_active 
                FROM users WHERE email = ?
            """, (email,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'username': row[0],
                    'email': row[1], 
                    'role': row[2],
                    'created_at': row[3],
                    'last_login': row[4],
                    'is_active': row[5]
                }
            return None
    
    def register_firebase_user(self, username: str, email: str, role: str, firebase_uid: str) -> Dict:
        """Register Firebase user in local database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, salt, role, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (
                    username,
                    email,
                    f"firebase:{firebase_uid}",  # Use Firebase UID as password hash
                    "firebase_auth",  # Special salt for Firebase users
                    role,
                    datetime.now()
                ))
                
                return {"success": True, "message": "Firebase user registered successfully"}
                
        except sqlite3.IntegrityError:
            return {"success": False, "message": "User already exists"}
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}

    def register_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> Dict:
        """Register new user"""
        try:
            # Validate input
            if len(username) < 3 or len(password) < 8:
                return {"success": False, "message": "Username must be 3+ chars, password 8+ chars"}
            
            if "@" not in email:
                return {"success": False, "message": "Invalid email format"}
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (username, email, password_hash, salt, role)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, email, password_hash, salt, role.value))
            
            return {"success": True, "message": "User registered successfully"}
            
        except sqlite3.IntegrityError:
            return {"success": False, "message": "Username or email already exists"}
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}
    
    def authenticate_user(self, username: str, password: str, ip_address: Optional[str] = None) -> Dict:
        """Authenticate user and return JWT token"""
        try:
            # Check rate limiting
            if not self._check_rate_limit(username, ip_address):
                return {"success": False, "message": "Too many failed attempts. Try again later."}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT password_hash, salt, role, is_active 
                    FROM users WHERE username = ?
                """, (username,))
                
                user_data = cursor.fetchone()
                
                # Log attempt
                conn.execute("""
                    INSERT INTO login_attempts (username, ip_address, success)
                    VALUES (?, ?, ?)
                """, (username, ip_address, user_data is not None))
                
                if not user_data:
                    return {"success": False, "message": "Invalid credentials"}
                
                password_hash, salt, role, is_active = user_data
                
                if not is_active:
                    return {"success": False, "message": "Account is deactivated"}
                
                if not self._verify_password(password, password_hash, salt):
                    return {"success": False, "message": "Invalid credentials"}
                
                # Update last login
                conn.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE username = ?
                """, (username,))
                
                # Generate token
                token = self._generate_jwt_token(username, role)
                
                # Store session
                token_hash = hashlib.sha256(token.encode()).hexdigest()
                expires_at = datetime.now() + timedelta(hours=self.token_expiry_hours)
                
                conn.execute("""
                    INSERT INTO sessions (username, token_hash, expires_at)
                    VALUES (?, ?, ?)
                """, (username, token_hash, expires_at))
                
                return {
                    "success": True,
                    "token": token,
                    "username": username,
                    "role": role,
                    "message": "Login successful"
                }
                
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}
    
    def verify_session(self, token: str) -> Optional[Dict]:
        """Verify user session and return user info"""
        payload = self._verify_jwt_token(token)
        if not payload:
            return None
        
        # Check if session exists and is active
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT username, expires_at, is_active 
                FROM sessions WHERE token_hash = ?
            """, (token_hash,))
            
            session_data = cursor.fetchone()
            
            if not session_data or not session_data[2]:
                return None
            
            username, expires_at, is_active = session_data
            expires_at = datetime.fromisoformat(expires_at)
            
            if datetime.now() > expires_at:
                # Deactivate expired session
                conn.execute("""
                    UPDATE sessions SET is_active = 0 WHERE token_hash = ?
                """, (token_hash,))
                return None
        
        return payload
    
    def logout_user(self, token: str) -> bool:
        """Logout user by deactivating session"""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE sessions SET is_active = 0 WHERE token_hash = ?
                """, (token_hash,))
            
            return True
        except:
            return False
    
    def get_user_info(self, username: str) -> Optional[User]:
        """Get user information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT username, email, role, created_at, last_login, is_active
                FROM users WHERE username = ?
            """, (username,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return None
            
            username, email, role, created_at, last_login, is_active = user_data
            
            return User(
                username=username,
                email=email,
                role=UserRole(role),
                created_at=datetime.fromisoformat(created_at),
                last_login=datetime.fromisoformat(last_login) if last_login else None,
                is_active=bool(is_active)
            )

# Streamlit session state integration
def init_auth_session():
    """Initialize authentication in Streamlit session"""
    # Force clear any existing session data to prevent cross-user contamination
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = AuthenticationSystem()
    
    # Always initialize these to prevent user session mixing
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'user_token' not in st.session_state:
        st.session_state.user_token = None
    
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
        
    # Generate unique session ID to prevent cross-contamination
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        
    # Clear any cached user data if session is not authenticated
    if not st.session_state.get('authenticated', False):
        st.session_state.user_info = None
        st.session_state.user_token = None

def check_authentication() -> bool:
    """Check if user is authenticated"""
    if not st.session_state.get('authenticated', False):
        return False
    
    if not st.session_state.get('user_token'):
        return False
    
    # Verify token
    auth_system = st.session_state.get('auth_system')
    if not auth_system:
        return False
    
    user_data = auth_system.verify_session(st.session_state.user_token)
    if not user_data:
        # Clear invalid session completely
        clear_user_session()
        return False
    
    # Ensure user_info matches the token to prevent session mixing
    if st.session_state.get('user_info'):
        token_username = user_data.get('username')
        session_username = st.session_state.user_info.get('username')
        
        if token_username != session_username:
            # Mismatch detected - clear session for security
            clear_user_session()
            return False
    
    return True

def clear_user_session():
    """Completely clear user session data"""
    st.session_state.authenticated = False
    st.session_state.user_token = None
    st.session_state.user_info = None
    
    # Clear any other user-specific data
    for key in list(st.session_state.keys()):
        if key.startswith('user_') or key in ['query_history', 'upload_history']:
            del st.session_state[key]

def require_auth(allowed_roles=None):
    """Decorator to require authentication for Streamlit functions"""
    if allowed_roles is None:
        allowed_roles = [UserRole.USER, UserRole.ADMIN]
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_authentication():
                st.error("Authentication required")
                st.stop()
            
            user_role = UserRole(st.session_state.user_info.get('role', 'user'))
            if user_role not in allowed_roles:
                st.error("Insufficient permissions")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator