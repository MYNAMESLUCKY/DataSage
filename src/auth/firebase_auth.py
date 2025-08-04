"""
Firebase Authentication Integration for Enterprise RAG System
"""

import os
import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import jwt

logger = logging.getLogger(__name__)

class FirebaseAuthManager:
    """Firebase authentication manager for Google OAuth integration"""
    
    def __init__(self):
        self.app = None
        self.jwt_secret = os.getenv("JWT_SECRET", "fallback-secret-key")
        self.token_expiry_hours = 24
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Get Firebase credentials from environment
                project_id = os.getenv("FIREBASE_PROJECT_ID")
                private_key = os.getenv("FIREBASE_PRIVATE_KEY")
                client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
                
                if not all([project_id, private_key, client_email]):
                    logger.warning("Firebase credentials not found in environment")
                    return False
                
                # Replace escaped newlines in private key
                if private_key:
                    private_key = private_key.replace('\\n', '\n')
                
                # Create credentials dict
                cred_dict = {
                    "type": "service_account",
                    "project_id": project_id,
                    "private_key_id": "firebase-key-id",
                    "private_key": private_key,
                    "client_email": client_email,
                    "client_id": "firebase-client-id",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
                }
                
                # Initialize Firebase
                cred = credentials.Certificate(cred_dict)
                self.app = firebase_admin.initialize_app(cred)
                logger.info("Firebase initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False
    
    def verify_firebase_token(self, id_token: str) -> Optional[Dict]:
        """Verify Firebase ID token and return user info"""
        try:
            if not self.app:
                logger.error("Firebase not initialized")
                return None
            
            # Verify the ID token
            decoded_token = firebase_auth.verify_id_token(id_token)
            
            # Extract user information
            user_info = {
                'uid': decoded_token.get('uid'),
                'email': decoded_token.get('email'),
                'name': decoded_token.get('name'),
                'picture': decoded_token.get('picture'),
                'email_verified': decoded_token.get('email_verified', False),
                'provider': 'google',
                'firebase_token': id_token
            }
            
            logger.info(f"Successfully verified Firebase token for user: {user_info['email']}")
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to verify Firebase token: {e}")
            return None
    
    def create_jwt_from_firebase(self, firebase_user: Dict, role: str = 'user') -> Optional[str]:
        """Create JWT token from Firebase user info"""
        try:
            payload = {
                'username': firebase_user['email'],
                'email': firebase_user['email'],
                'name': firebase_user.get('name', ''),
                'role': role,
                'provider': 'google',
                'uid': firebase_user['uid'],
                'email_verified': firebase_user.get('email_verified', False),
                'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            logger.info(f"Created JWT token for Firebase user: {firebase_user['email']}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create JWT from Firebase user: {e}")
            return None
    
    def get_google_oauth_config(self) -> Optional[Dict]:
        """Get Google OAuth configuration for Streamlit OAuth"""
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            logger.error("Google OAuth credentials not found")
            return None
        
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": "http://localhost:5000",  # Will be updated for production
            "scope": [
                "openid",
                "email", 
                "profile"
            ],
            "authorization_url": "https://accounts.google.com/o/oauth2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://www.googleapis.com/oauth2/v1/userinfo"
        }
    
    def process_oauth_callback(self, token_info: Dict) -> Optional[Dict]:
        """Process OAuth callback and create user session"""
        try:
            # Extract user info from OAuth token
            user_info = {
                'email': token_info.get('email'),
                'name': token_info.get('name'),
                'picture': token_info.get('picture'),
                'email_verified': token_info.get('email_verified', True),
                'provider': 'google',
                'uid': token_info.get('sub') or token_info.get('id')
            }
            
            if not user_info['email']:
                logger.error("No email found in OAuth token")
                return None
            
            # Determine user role (default to 'user', first user becomes admin)
            role = self._determine_user_role(user_info['email'])
            
            # Create JWT token
            jwt_token = self.create_jwt_from_firebase(user_info, role)
            
            if jwt_token:
                return {
                    'success': True,
                    'token': jwt_token,
                    'user': user_info,
                    'role': role
                }
            else:
                return {'success': False, 'message': 'Failed to create session token'}
                
        except Exception as e:
            logger.error(f"Failed to process OAuth callback: {e}")
            return {'success': False, 'message': str(e)}
    
    def _determine_user_role(self, email: str) -> str:
        """Determine user role based on email and existing users"""
        try:
            # Import here to avoid circular imports
            from src.auth.auth_system import AuthenticationSystem
            
            auth_system = AuthenticationSystem()
            
            # Check if this is the first user (should be admin)
            if not auth_system._has_any_users():
                logger.info(f"First user {email} assigned admin role")
                return 'admin'
            
            # Check if user already exists in local database
            existing_user = auth_system._get_user_by_email(email)
            if existing_user:
                return existing_user.get('role', 'user')
            
            # Default role for new users
            return 'user'
            
        except Exception as e:
            logger.error(f"Error determining user role: {e}")
            return 'user'
    
    def sync_with_local_auth(self, firebase_user: Dict, role: str):
        """Sync Firebase user with local authentication system"""
        try:
            from src.auth.auth_system import AuthenticationSystem
            
            auth_system = AuthenticationSystem()
            
            # Check if user exists locally
            existing_user = auth_system._get_user_by_email(firebase_user['email'])
            
            if not existing_user:
                # Create local user record for Firebase user
                result = auth_system.register_firebase_user(
                    username=firebase_user['email'],
                    email=firebase_user['email'],
                    role=role,
                    firebase_uid=firebase_user['uid']
                )
                
                if result['success']:
                    logger.info(f"Created local record for Firebase user: {firebase_user['email']}")
                else:
                    logger.error(f"Failed to create local record: {result['message']}")
            
        except Exception as e:
            logger.error(f"Failed to sync with local auth: {e}")
    
    def is_firebase_available(self) -> bool:
        """Check if Firebase is properly configured and available"""
        return self.app is not None and all([
            os.getenv("FIREBASE_PROJECT_ID"),
            os.getenv("FIREBASE_PRIVATE_KEY"), 
            os.getenv("FIREBASE_CLIENT_EMAIL"),
            os.getenv("GOOGLE_CLIENT_ID"),
            os.getenv("GOOGLE_CLIENT_SECRET")
        ])