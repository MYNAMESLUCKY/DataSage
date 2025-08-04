"""
Enhanced Session Management for User Isolation
Prevents cross-user session contamination
"""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Enhanced session management with user isolation"""
    
    @staticmethod
    def create_user_session(user_info: Dict, token: str) -> str:
        """Create a new isolated user session"""
        try:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Clear any existing session data completely
            SessionManager.clear_all_session_data()
            
            # Set new session data with isolation
            st.session_state.session_id = session_id
            st.session_state.authenticated = True
            st.session_state.user_token = token
            st.session_state.user_info = user_info.copy()
            st.session_state.login_time = datetime.now().isoformat()
            
            # Initialize user-specific storage
            st.session_state.user_query_history = []
            st.session_state.user_upload_history = []
            st.session_state.user_preferences = {}
            
            logger.info(f"Created isolated session for user: {user_info.get('username')} (ID: {session_id[:8]})")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create user session: {e}")
            raise
    
    @staticmethod
    def clear_all_session_data():
        """Completely clear all session data to prevent contamination"""
        try:
            # List of keys to preserve (system-level, not user-specific)
            preserve_keys = {'auth_system', 'rate_limiter'}
            
            # Clear all user-specific data
            keys_to_remove = []
            for key in st.session_state.keys():
                if key not in preserve_keys:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del st.session_state[key]
                
            logger.info("Cleared all user session data")
            
        except Exception as e:
            logger.error(f"Failed to clear session data: {e}")
    
    @staticmethod
    def validate_session() -> bool:
        """Validate current session integrity"""
        try:
            if not st.session_state.get('authenticated', False):
                return False
                
            if not st.session_state.get('session_id'):
                return False
                
            if not st.session_state.get('user_info'):
                return False
                
            if not st.session_state.get('user_token'):
                return False
            
            # Check session age (optional timeout)
            login_time_str = st.session_state.get('login_time')
            if login_time_str:
                login_time = datetime.fromisoformat(login_time_str)
                age_hours = (datetime.now() - login_time).total_seconds() / 3600
                
                # Auto-logout after 24 hours
                if age_hours > 24:
                    logger.warning(f"Session expired after {age_hours:.1f} hours")
                    SessionManager.clear_all_session_data()
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return False
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user's unique identifier"""
        user_info = st.session_state.get('user_info')
        if user_info:
            return user_info.get('username')
        return None
    
    @staticmethod
    def get_session_info() -> Dict:
        """Get current session information"""
        return {
            'session_id': st.session_state.get('session_id'),
            'user_id': SessionManager.get_user_id(),
            'authenticated': st.session_state.get('authenticated', False),
            'login_time': st.session_state.get('login_time'),
            'provider': st.session_state.get('user_info', {}).get('provider'),
            'role': st.session_state.get('user_info', {}).get('role')
        }
    
    @staticmethod
    def logout_user():
        """Safely logout user and clear all session data"""
        try:
            user_id = SessionManager.get_user_id()
            session_id = st.session_state.get('session_id', 'unknown')[:8]
            
            # Clear all session data
            SessionManager.clear_all_session_data()
            
            # Reset to unauthenticated state
            st.session_state.authenticated = False
            st.session_state.user_token = None
            st.session_state.user_info = None
            
            logger.info(f"User logged out: {user_id} (Session: {session_id})")
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")

# Global session manager instance
session_manager = SessionManager()