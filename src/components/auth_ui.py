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
        # Embed Firebase Web SDK with authentication handling
        st.markdown(f"""
        <script type="module">
        import {{ initializeApp }} from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-app.js';
        import {{ getAuth, signInWithPopup, GoogleAuthProvider, getIdToken }} from 'https://www.gstatic.com/firebasejs/9.0.0/firebase-auth.js';
        
        // Firebase configuration
        const firebaseConfig = {firebase_config};
        
        // Initialize Firebase
        const app = initializeApp(firebaseConfig);
        const auth = getAuth(app);
        
        // Google Sign-In
        window.signInWithGoogle = async function() {{
            const provider = new GoogleAuthProvider();
            try {{
                const result = await signInWithPopup(auth, provider);
                const idToken = await getIdToken(result.user);
                
                // Send authentication data to Streamlit
                window.parent.postMessage({{
                    type: 'firebase-auth-success',
                    idToken: idToken,
                    user: {{
                        uid: result.user.uid,
                        email: result.user.email,
                        displayName: result.user.displayName,
                        photoURL: result.user.photoURL
                    }}
                }}, '*');
                
                // Reload page to update authentication state
                setTimeout(() => {{ window.location.reload(); }}, 1000);
                
            }} catch (error) {{
                console.error('Firebase authentication error:', error);
                window.parent.postMessage({{
                    type: 'firebase-auth-error',
                    error: error.message
                }}, '*');
            }}
        }}
        </script>
        
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
            <button onclick="signInWithGoogle()" 
                    style="background-color: #4285f4; color: white; border: none; 
                           padding: 15px 30px; border-radius: 8px; font-size: 16px; 
                           cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                           display: flex; align-items: center; margin: 0 auto;">
                <svg width="20" height="20" viewBox="0 0 24 24" style="margin-right: 8px;">
                    <path fill="white" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="white" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="white" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="white" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Sign in with Google
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle authentication result
        handle_firebase_auth_result()
    else:
        st.error("Firebase Google Authentication not configured")
        st.info("Please configure Firebase credentials to enable Google login")

def get_firebase_web_config():
    """Get Firebase web configuration"""
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    
    if not project_id:
        return None
    
    # Extract API key from client ID (common pattern)
    api_key = client_id.split('-')[0] if client_id else "auto-generated-key"
    
    return f"""{{
        apiKey: "{api_key}",
        authDomain: "{project_id}.firebaseapp.com",
        projectId: "{project_id}",
        storageBucket: "{project_id}.appspot.com",
        messagingSenderId: "sender-id",
        appId: "app-id"
    }}"""

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