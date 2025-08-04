"""
Firebase Authentication UI Components
"""

import streamlit as st
import logging
from streamlit_oauth import OAuth2Component
from src.auth.firebase_auth import FirebaseAuthManager
from src.auth.auth_system import UserRole
import time

logger = logging.getLogger(__name__)

def show_google_login():
    """Display Google OAuth login component"""
    
    # Initialize Firebase Auth Manager
    if 'firebase_auth' not in st.session_state:
        st.session_state.firebase_auth = FirebaseAuthManager()
    
    firebase_auth = st.session_state.firebase_auth
    
    # Check if Firebase is available
    if not firebase_auth.is_firebase_available():
        st.warning("🔧 Google Authentication is not configured. Please contact administrator.")
        return False
    
    st.markdown("### 🔐 Enterprise Authentication")
    
    # Create tabs for different login methods
    tab1, tab2 = st.tabs(["🔑 Standard Login", "🌐 Google Login"])
    
    with tab1:
        st.info("Use your registered username and password")
        # This will show the standard login form
        from src.components.auth_ui import show_login_form
        return show_login_form()
    
    with tab2:
        st.info("Sign in with your Google account for secure access")
        
        # Get OAuth configuration
        oauth_config = firebase_auth.get_google_oauth_config()
        
        if not oauth_config:
            st.error("Google OAuth configuration not available")
            return False
        
        # Create OAuth2 component
        oauth2 = OAuth2Component(
            client_id=oauth_config["client_id"],
            client_secret=oauth_config["client_secret"],
            redirect_uri=oauth_config["redirect_uri"],
            scope=" ".join(oauth_config["scope"]),
            authorization_url=oauth_config["authorization_url"],
            token_url=oauth_config["token_url"],
            userinfo_url=oauth_config["userinfo_url"]
        )
        
        # Handle OAuth flow
        if st.button("🚀 Sign in with Google", key="google_auth_btn", use_container_width=True):
            try:
                # Start OAuth flow
                authorization_url = oauth2.get_authorization_url()
                st.markdown(f'<meta http-equiv="refresh" content="0; url={authorization_url}">', 
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to initiate Google login: {str(e)}")
                logger.error(f"Google OAuth error: {e}")
        
        # Handle OAuth callback
        query_params = st.query_params
        
        if "code" in query_params:
            try:
                # Exchange code for token
                token_info = oauth2.get_token(query_params["code"])
                
                if token_info:
                    # Process OAuth callback
                    result = firebase_auth.process_oauth_callback(token_info)
                    
                    if result and result['success']:
                        # Set session state
                        st.session_state.authenticated = True
                        st.session_state.user_token = result['token']
                        st.session_state.user_info = {
                            'username': result['user']['email'],
                            'email': result['user']['email'],
                            'name': result['user'].get('name', ''),
                            'role': result['role'],
                            'provider': 'google',
                            'picture': result['user'].get('picture')
                        }
                        
                        # Sync with local auth system
                        firebase_auth.sync_with_local_auth(result['user'], result['role'])
                        
                        st.success(f"✅ Successfully logged in as {result['user']['name']} ({result['role']})")
                        time.sleep(1)
                        
                        # Clear query parameters and rerun
                        st.query_params.clear()
                        st.rerun()
                        
                        return True
                    else:
                        st.error(f"Authentication failed: {result.get('message', 'Unknown error')}")
                else:
                    st.error("Failed to obtain access token")
                    
            except Exception as e:
                st.error(f"OAuth callback error: {str(e)}")
                logger.error(f"OAuth callback error: {e}")
        
        # Show additional info
        st.markdown("---")
        st.markdown("""
        **Google Authentication Benefits:**
        - 🔐 Secure OAuth 2.0 authentication
        - 🚀 Single sign-on with Google account
        - ✅ Automatic email verification
        - 🛡️ Enterprise-grade security
        """)
    
    return False

def show_user_profile_with_google():
    """Show user profile with Google account information"""
    
    if not st.session_state.get('authenticated', False):
        return
    
    user_info = st.session_state.get('user_info', {})
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 User Profile")
    
    # Show profile picture if available (Google users)
    if user_info.get('picture'):
        st.sidebar.image(user_info['picture'], width=60)
    
    # Show user information
    st.sidebar.write(f"**Name:** {user_info.get('name', 'N/A')}")
    st.sidebar.write(f"**Email:** {user_info.get('email', 'N/A')}")
    st.sidebar.write(f"**Role:** {user_info.get('role', 'user').title()}")
    
    # Show authentication provider
    provider = user_info.get('provider', 'local')
    provider_icon = "🌐" if provider == 'google' else "🔑"
    st.sidebar.write(f"**Auth:** {provider_icon} {provider.title()}")
    
    # Security status
    if provider == 'google':
        st.sidebar.success("🔐 Google Verified")
    else:
        st.sidebar.info("🔑 Local Account")

def show_admin_google_management():
    """Show admin panel for Google authentication management"""
    
    if not st.session_state.get('authenticated', False):
        return
    
    user_info = st.session_state.get('user_info', {})
    if user_info.get('role') != 'admin':
        return
    
    st.subheader("🌐 Google Authentication Management")
    
    # Firebase status
    firebase_auth = st.session_state.get('firebase_auth')
    if firebase_auth and firebase_auth.is_firebase_available():
        st.success("✅ Google Authentication is configured and active")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Authentication Provider", "Google Firebase", "Active")
        
        with col2:  
            st.metric("OAuth Status", "Configured", "Ready")
        
    else:
        st.warning("⚠️ Google Authentication not configured")
        st.info("""
        **Required Configuration:**
        - FIREBASE_PROJECT_ID
        - FIREBASE_PRIVATE_KEY  
        - FIREBASE_CLIENT_EMAIL
        - GOOGLE_CLIENT_ID
        - GOOGLE_CLIENT_SECRET
        
        Contact system administrator to configure these secrets.
        """)
    
    # Show Google user statistics
    st.markdown("---")
    st.write("**Authentication Statistics:**")
    
    # In a real implementation, you'd query actual user data
    google_users = 0  # Count of Google authenticated users
    local_users = 0   # Count of local authenticated users
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Google Users", google_users)
    
    with col2:
        st.metric("Local Users", local_users)
    
    with col3:
        st.metric("Total Users", google_users + local_users)