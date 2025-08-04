"""
Authentication UI Components for Enterprise RAG System
"""

import streamlit as st
from typing import Optional
import time
from ..auth.auth_system import AuthenticationSystem, UserRole, init_auth_session, check_authentication
from ..security.rate_limiter import RateLimiter, RateLimitType, rate_limit
import os

def show_login_page():
    """Display login page with Google Firebase integration"""
    # Mobile-optimized header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Enterprise RAG System</h1>
        <p>Secure Authentication Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        st.success("✅ Already authenticated!")
        return True
    
    try:
        # Initialize auth system
        init_auth_session()
        auth_system = st.session_state.auth_system
        
        # Check if Firebase is available
        firebase_available = check_firebase_availability()
        
        if firebase_available:
            # Show Firebase Google login only
            show_firebase_google_login()
            
            # Also show standard login as fallback
            st.markdown("---")
            st.info("Alternative: Use standard username/password login below")
            show_standard_login_form()
        else:
            # Show only standard login
            st.info("💡 Standard authentication available. Configure Firebase for Google login.")
            show_standard_login_form()
            
    except Exception as e:
        st.error(f"Authentication system error: {str(e)}")
        st.info("Please refresh the page or contact support.")
        
    return False

def check_firebase_availability():
    """Check if Firebase credentials are available"""
    required_vars = [
        "FIREBASE_PROJECT_ID",
        "FIREBASE_PRIVATE_KEY", 
        "FIREBASE_CLIENT_EMAIL",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET"
    ]
    return all(os.getenv(var) for var in required_vars)

def show_firebase_google_login():
    """Display Firebase Google login option"""
    st.markdown("### 🌐 Firebase Google Authentication")
    st.info("Sign in with your Google account for secure access")
    
    # Firebase Web SDK integration
    firebase_config = get_firebase_web_config()
    
    if firebase_config:
        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display debug information
            web_api_key = os.getenv("FIREBASE_WEB_API_KEY")
            if not web_api_key or web_api_key == "AIzaSyDummyWebAPIKey":
                st.warning("⚠️ Firebase Web API Key not configured. The button won't work until you add FIREBASE_WEB_API_KEY to your environment variables.")
                st.info("Get this from Firebase Console → Project Settings → Web API Key")
            
            # Use Streamlit button instead of HTML button for better compatibility
            if st.button("🚀 Sign in with Google", 
                        key="google_login_btn",
                        help="Click to sign in with your Google account",
                        use_container_width=True):
                
                if not web_api_key or web_api_key == "AIzaSyDummyWebAPIKey":
                    st.error("❌ Cannot authenticate: Firebase Web API Key is required")
                    st.info("Please add your FIREBASE_WEB_API_KEY to environment variables")
                    return
                
                # Store Firebase config in session state for JavaScript access
                st.session_state.firebase_config = firebase_config
                st.session_state.show_firebase_auth = True
                st.rerun()
            
            # Show Firebase authentication popup if triggered
            if st.session_state.get('show_firebase_auth', False):
                st.session_state.show_firebase_auth = False
                
                # Use a different approach - redirect to a simple authentication page
                st.info("🔄 Redirecting to Google authentication...")
                
                # Create a simple authentication URL with Firebase config stored securely
                project_id = os.getenv("FIREBASE_PROJECT_ID")
                auth_domain = f"{project_id}.firebaseapp.com"
                
                # Use Google's OAuth URL directly
                google_auth_url = f"https://accounts.google.com/o/oauth2/v2/auth"
                client_id = os.getenv("GOOGLE_CLIENT_ID")
                redirect_uri = st.get_option("server.baseUrlPath") or "http://localhost:5000"
                
                params = {
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "response_type": "code",
                    "scope": "email profile openid",
                    "state": "firebase_auth"
                }
                
                # Create the authentication URL
                auth_url = google_auth_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <p>Redirecting to secure Google authentication...</p>
                    <script>
                        setTimeout(function() {{
                            window.open('{auth_url}', '_blank', 'width=500,height=600');
                        }}, 1000);
                    </script>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("After authentication, please return to this page and refresh.")
        
        # Custom CSS for the button styling
        st.markdown("""
        <style>
        .stButton > button {
            background: linear-gradient(135deg, #4285f4 0%, #3367d6 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 500;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #3367d6 0%, #2d5aa0 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Handle authentication result
        handle_firebase_auth_result()
    else:
        st.error("Firebase Google Authentication not configured")
        st.info("Please configure Firebase credentials to enable Google login")

def get_firebase_web_config():
    """Get Firebase web configuration - SECURE VERSION"""
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    
    if not project_id:
        return None
    
    # Return a secure config indicator without exposing keys
    return f"firebase-config-{project_id}"

def handle_firebase_auth_result():
    """Handle Firebase authentication result"""
    # Listen for authentication messages from JavaScript
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'firebase-auth-success') {
            // Show success message
            console.log('Firebase auth success:', event.data);
            
            // Store token for processing
            sessionStorage.setItem('firebase_token', event.data.idToken);
            sessionStorage.setItem('firebase_user', JSON.stringify(event.data.user));
            
            // Trigger page reload to process authentication
            window.location.reload();
        } else if (event.data.type === 'firebase-auth-error') {
            console.error('Firebase auth error:', event.data.error);
            alert('Authentication failed: ' + event.data.error);
        }
    });
    
    // Check for stored authentication result on page load
    const storedToken = sessionStorage.getItem('firebase_token');
    const storedUser = sessionStorage.getItem('firebase_user');
    
    if (storedToken && storedUser) {
        // Clear from session storage
        sessionStorage.removeItem('firebase_token');
        sessionStorage.removeItem('firebase_user');
        
        // Process authentication by setting URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        if (!urlParams.has('firebase_token')) {
            urlParams.set('firebase_token', storedToken);
            window.location.search = urlParams.toString();
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Process authentication if Firebase token is in URL parameters
    query_params = st.query_params
    if "firebase_token" in query_params:
        id_token = query_params["firebase_token"]
        
        with st.spinner("Processing Google authentication..."):
            process_firebase_token(id_token)
            
            # Clear query parameters
            st.query_params.clear()
            st.rerun()

def process_firebase_token(id_token: str):
    """Process Firebase ID token for authentication"""
    import time
    
    try:
        from src.auth.firebase_auth import FirebaseAuthManager
        from src.auth.auth_system import UserRole
        
        # Initialize Firebase Auth Manager
        firebase_auth = FirebaseAuthManager()
        
        # Verify Firebase token
        verified_user = firebase_auth.verify_firebase_token(id_token)
        
        if verified_user:
            # Get auth system from session state
            auth_system = st.session_state.auth_system
            
            # Determine user role
            if not auth_system._has_any_users():
                role = UserRole.ADMIN  # First user becomes admin
            else:
                existing_user = auth_system._get_user_by_email(verified_user['email'])
                if existing_user:
                    role = UserRole(existing_user.get('role', 'user'))
                else:
                    role = UserRole.USER  # New users get user role
            
            # Create local user account if doesn't exist
            username = verified_user['email'].split('@')[0]  # Use email prefix as username
            
            # Try to register or get existing user
            user_result = auth_system.register_user(
                username=username,
                email=verified_user['email'],
                password="firebase_user",  # Placeholder password for Firebase users
                role=role
            )
            
            if user_result['success'] or "already exists" in user_result.get('message', ''):
                # Set session state
                st.session_state.authenticated = True
                st.session_state.user_token = id_token
                st.session_state.user_info = {
                    'username': username,
                    'email': verified_user['email'],
                    'role': role.value,
                    'provider': 'firebase'
                }
                
                st.success(f"✅ Successfully authenticated as {verified_user.get('name', verified_user['email'])} ({role.value})")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ Failed to create user account: {user_result.get('message', 'Unknown error')}")
        else:
            st.error("❌ Invalid Firebase token")
            
    except Exception as e:
        st.error(f"❌ Authentication failed: {str(e)}")
        import traceback
        traceback.print_exc()



def show_standard_login_form():
    """Show standard username/password login form"""
    st.info("🔑 Use your registered username and password")
    
    try:
        # Get auth system from session state
        auth_system = st.session_state.auth_system
        
        # Mobile-optimized login form
        with st.form("login_form"):
            st.subheader("Sign In")
            
            username = st.text_input(
                "Username", 
                placeholder="Enter your username",
                help="Your unique username"
            )
            password = st.text_input(
                "Password", 
                type="password", 
                placeholder="Enter your password",
                help="Your secure password"
            )
            
            # Mobile-friendly buttons
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
                        "role": result["role"],
                        "provider": "local"
                    }
                    st.success("✅ Login successful!")
                    time.sleep(1)
                    st.rerun()
                    return True
                else:
                    st.error(f"❌ {result['message']}")
        
        if register_clicked:
            st.session_state.show_register = True
            st.rerun()
    
    except Exception as e:
        st.error(f"Login form error: {str(e)}")
        return False
    
    # Show registration form if requested
    if st.session_state.get('show_register', False):
        show_registration_form()
        return False
    
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
    
    return False

def show_registration_form():
    """Display registration form"""
    st.markdown("---")
    st.subheader("📝 Create New Account")
    
    # Get auth system from session state
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
                    result = auth_system.register_user(username, email, password, UserRole.USER)
                    
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