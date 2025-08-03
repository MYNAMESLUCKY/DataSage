"""
Authentication UI Components for Enterprise RAG System
"""

import streamlit as st
from typing import Optional
import time
from ..auth.auth_system import AuthenticationSystem, UserRole, init_auth_session, check_authentication
from ..security.rate_limiter import RateLimiter, RateLimitType, rate_limit

def show_login_page():
    """Display login page"""
    st.title("🔐 RAG System Login")
    st.markdown("---")
    
    # Initialize auth system
    init_auth_session()
    auth_system = st.session_state.auth_system
    
    # Login form
    with st.form("login_form"):
        st.subheader("Sign In")
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_clicked = st.form_submit_button("🚀 Login", use_container_width=True)
        with col2:
            register_clicked = st.form_submit_button("📝 Register", use_container_width=True)
        
        if login_clicked and username and password:
            with st.spinner("Authenticating..."):
                result = auth_system.authenticate_user(username, password)
                
                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.user_token = result["token"]
                    st.session_state.user_info = {
                        "username": result["username"],
                        "role": result["role"]
                    }
                    st.success("✅ Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
        
        if register_clicked:
            st.session_state.show_register = True
            st.rerun()
    
    # Show registration form if requested
    if st.session_state.get('show_register', False):
        show_registration_form()
    
    # Additional info
    st.markdown("---")
    st.info("**New User Registration:**\nClick 'Register' to create your account with secure credentials.")
    
    # Rate limit info
    if 'rate_limiter' not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    
    rate_limiter = st.session_state.rate_limiter
    status = rate_limiter.get_rate_limit_status(RateLimitType.LOGIN)
    
    if status["requests_made"] > 0:
        st.caption(f"Login attempts: {status['requests_made']}/{status['max_requests']} (resets in {status['window_seconds']}s)")

def show_registration_form():
    """Display registration form"""
    st.markdown("---")
    st.subheader("📝 Create New Account")
    
    auth_system = st.session_state.auth_system
    
    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Choose a username (min 3 characters)")
        email = st.text_input("Email", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Create password (min 8 characters)")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        col1, col2 = st.columns(2)
        with col1:
            register_clicked = st.form_submit_button("✅ Create Account", use_container_width=True)
        with col2:
            cancel_clicked = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if register_clicked:
            if password != confirm_password:
                st.error("❌ Passwords don't match")
            elif len(username) < 3:
                st.error("❌ Username must be at least 3 characters")
            elif len(password) < 8:
                st.error("❌ Password must be at least 8 characters")
            elif "@" not in email:
                st.error("❌ Please enter a valid email address")
            else:
                with st.spinner("Creating account..."):
                    result = auth_system.register_user(username, email, password)
                    
                    if result["success"]:
                        st.success("✅ Account created successfully! Please login.")
                        st.session_state.show_register = False
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        
        if cancel_clicked:
            st.session_state.show_register = False
            st.rerun()

def show_user_dashboard():
    """Display user dashboard in sidebar"""
    if not check_authentication():
        return
    
    user_info = st.session_state.user_info
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("👤 User Dashboard")
        
        # User info
        st.write(f"**User:** {user_info['username']}")
        st.write(f"**Role:** {user_info['role'].title()}")
        
        # Role badge
        if user_info['role'] == 'admin':
            st.success("🔰 Administrator")
        elif user_info['role'] == 'user':
            st.info("👤 Standard User")
        else:
            st.warning("👁️ Viewer")
        
        # Logout button
        if st.button("🚪 Logout", key="sidebar_logout", use_container_width=True):
            auth_system = st.session_state.auth_system
            if st.session_state.user_token:
                auth_system.logout_user(st.session_state.user_token)
            
            # Clear session
            st.session_state.authenticated = False
            st.session_state.user_token = None
            st.session_state.user_info = None
            
            st.success("✅ Logged out successfully!")
            time.sleep(1)
            st.rerun()

def show_admin_panel():
    """Display admin panel for user management"""
    if not check_authentication():
        return
    
    user_info = st.session_state.user_info
    if user_info['role'] != 'admin':
        st.error("🚫 Admin access required")
        return
    
    st.subheader("🔰 Admin Panel")
    
    tabs = st.tabs(["User Management", "Rate Limits", "System Stats"])
    
    with tabs[0]:
        st.write("**User Management**")
        
        # In a real implementation, you'd fetch and display user list
        st.info("User management features would be implemented here")
        
        # Demo user creation for admin
        # Demo accounts removed for production security
    
    with tabs[1]:
        st.write("**Rate Limit Management**")
        
        if 'rate_limiter' not in st.session_state:
            st.session_state.rate_limiter = RateLimiter()
        
        rate_limiter = st.session_state.rate_limiter
        
        # Show blocked identifiers
        blocked = rate_limiter.get_blocked_identifiers()
        
        if blocked:
            st.write("**Currently Blocked:**")
            for block in blocked:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(f"{block['identifier'][:8]}...")
                with col2:
                    st.text(f"{block['limit_type']} until {block['blocked_until']}")
                with col3:
                    if st.button("Unblock", key=f"unblock_{block['identifier']}_{block['limit_type']}"):
                        if rate_limiter.unblock_identifier(block['identifier'], RateLimitType(block['limit_type'])):
                            st.success("Unblocked!")
                            st.rerun()
        else:
            st.info("No blocked identifiers")
    
    with tabs[2]:
        st.write("**System Statistics**")
        
        # System stats would be implemented here
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Sessions", "12")  # Placeholder
        
        with col2:
            st.metric("Total Queries Today", "1,247")  # Placeholder
        
        with col3:
            st.metric("Rate Limit Blocks", len(blocked) if 'blocked' in locals() else 0)

def require_authentication():
    """Middleware function to require authentication"""
    init_auth_session()
    
    if not check_authentication():
        show_login_page()
        st.stop()
    
    # Show user dashboard
    show_user_dashboard()

def show_security_info():
    """Show security information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🛡️ Security")
        
        # Rate limit status for queries
        if 'rate_limiter' not in st.session_state:
            st.session_state.rate_limiter = RateLimiter()
        
        rate_limiter = st.session_state.rate_limiter
        query_status = rate_limiter.get_rate_limit_status(RateLimitType.QUERY)
        
        # Query rate limit progress
        st.write("**Query Limits**")
        progress = query_status["requests_made"] / query_status["max_requests"]
        st.progress(progress)
        st.caption(f"{query_status['remaining']} queries remaining this hour")
        
        if query_status["is_blocked"]:
            st.error("🚫 Query limit exceeded")
        elif progress > 0.8:
            st.warning("⚠️ Approaching query limit")
        else:
            st.success("✅ Within query limits")

# Utility function for protected routes
def protected_page(title: str, allowed_roles=None):
    """Decorator for protected pages"""
    if allowed_roles is None:
        allowed_roles = [UserRole.USER, UserRole.ADMIN]
    
    require_authentication()
    
    user_role = UserRole(st.session_state.user_info['role'])
    if user_role not in allowed_roles:
        st.error("🚫 Insufficient permissions for this page")
        st.stop()
    
    st.title(title)
    return True