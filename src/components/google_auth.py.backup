"""
Simple Google Authentication Component for Firebase Integration
"""

import streamlit as st
import os
import requests
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GoogleAuthComponent:
    """Simple Google OAuth component for Firebase integration"""
    
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = "http://localhost:5000/auth/callback"
        
    def is_configured(self) -> bool:
        """Check if Google OAuth is properly configured"""
        return bool(self.client_id and self.client_secret)
    
    def get_authorization_url(self) -> str:
        """Get Google OAuth authorization URL"""
        if not self.is_configured():
            return None
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"https://accounts.google.com/o/oauth2/auth?{param_string}"
    
    def exchange_code_for_token(self, code: str) -> Optional[Dict]:
        """Exchange authorization code for access token"""
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': self.redirect_uri
            }
            
            response = requests.post('https://oauth2.googleapis.com/token', data=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Token exchange failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None
    
    def get_user_info(self, access_token: str) -> Optional[Dict]:
        """Get user information from Google"""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get('https://www.googleapis.com/oauth2/v2/userinfo', headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get user info: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None

def show_google_login_button():
    """Show Google login button and handle OAuth flow"""
    google_auth = GoogleAuthComponent()
    
    if not google_auth.is_configured():
        st.warning("🔧 Google Authentication not configured")
        return False
    
    st.markdown("### 🌐 Google Authentication")
    
    # Check for OAuth callback
    query_params = st.query_params
    
    if "code" in query_params:
        # Handle OAuth callback
        code = query_params["code"]
        
        with st.spinner("Processing Google authentication..."):
            # Exchange code for token
            token_data = google_auth.exchange_code_for_token(code)
            
            if token_data and 'access_token' in token_data:
                # Get user info
                user_info = google_auth.get_user_info(token_data['access_token'])
                
                if user_info:
                    # Process successful authentication
                    success = process_google_user(user_info)
                    
                    if success:
                        # Clear query params and redirect
                        st.query_params.clear()
                        st.rerun()
                    else:
                        st.error("Failed to process Google authentication")
                else:
                    st.error("Failed to get user information from Google")
            else:
                st.error("Failed to exchange authorization code for token")
    else:
        # Show login button
        if st.button("🚀 Sign in with Google", key="google_login", use_container_width=True):
            auth_url = google_auth.get_authorization_url()
            if auth_url:
                st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', 
                           unsafe_allow_html=True)
            else:
                st.error("Failed to generate Google authentication URL")
    
    return False

def process_google_user(user_info: Dict) -> bool:
    """Process Google user authentication"""
    try:
        # Initialize Firebase auth if available
        if 'firebase_auth' not in st.session_state:
            try:
                from src.auth.firebase_auth import FirebaseAuthManager
                st.session_state.firebase_auth = FirebaseAuthManager()
            except ImportError:
                st.session_state.firebase_auth = None
        
        firebase_auth = st.session_state.firebase_auth
        auth_system = st.session_state.auth_system
        
        # Determine user role
        email = user_info.get('email')
        if not email:
            st.error("No email provided by Google")
            return False
        
        # Check if first user (becomes admin)
        role = 'admin' if not auth_system._has_any_users() else 'user'
        
        # Check if user exists locally
        existing_user = auth_system._get_user_by_email(email)
        if existing_user:
            role = existing_user.get('role', 'user')
        
        # Create/update user in local database
        if not existing_user:
            result = auth_system.register_firebase_user(
                username=email,
                email=email,
                role=role,
                firebase_uid=user_info.get('id', user_info.get('sub', ''))
            )
            
            if not result['success']:
                st.error(f"Failed to create local user: {result['message']}")
                return False
        
        # Create JWT token
        if firebase_auth:
            jwt_token = firebase_auth.create_jwt_from_firebase({
                'uid': user_info.get('id', user_info.get('sub', '')),
                'email': email,
                'name': user_info.get('name', ''),
                'picture': user_info.get('picture', ''),
                'email_verified': user_info.get('verified_email', True)
            }, role)
        else:
            # Fallback JWT creation
            from src.auth.auth_system import AuthenticationSystem
            auth = AuthenticationSystem()
            jwt_token = auth._generate_jwt_token(email, role)
        
        if jwt_token:
            # Set session state
            st.session_state.authenticated = True
            st.session_state.user_token = jwt_token
            st.session_state.user_info = {
                'username': email,
                'email': email,
                'name': user_info.get('name', ''),
                'role': role,
                'provider': 'google',
                'picture': user_info.get('picture')
            }
            
            st.success(f"✅ Successfully authenticated as {user_info.get('name', email)} ({role})")
            return True
        else:
            st.error("Failed to create authentication token")
            return False
            
    except Exception as e:
        st.error(f"Authentication processing failed: {str(e)}")
        logger.error(f"Google auth processing error: {e}")
        return False