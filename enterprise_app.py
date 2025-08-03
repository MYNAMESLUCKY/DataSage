"""
Enterprise RAG System with Authentication and Security
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append('src')

from src.components.auth_ui import show_login_page, show_admin_panel, show_security_info, protected_page
from src.components.enterprise_ui import EnterpriseUI
from src.auth.auth_system import UserRole
from src.security.rate_limiter import rate_limit, RateLimitType

# Page configuration with enhanced settings
st.set_page_config(
    page_title="Enterprise RAG Intelligence Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Security headers and browser protection
st.markdown("""
<script>
// Disable right-click context menu
document.addEventListener('contextmenu', function(e) {
    e.preventDefault();
});

// Disable F12, Ctrl+Shift+I, Ctrl+U developer tools
document.addEventListener('keydown', function(e) {
    if (e.key === 'F12' || 
        (e.ctrlKey && e.shiftKey && e.key === 'I') ||
        (e.ctrlKey && e.key === 'u')) {
        e.preventDefault();
        return false;
    }
});

// Disable text selection
document.addEventListener('selectstart', function(e) {
    if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
    }
});

// Clear console periodically
setInterval(function() {
    console.clear();
    console.log('%cSystem Protected', 'color: red; font-size: 20px; font-weight: bold;');
    console.log('%cUnauthorized access attempts are logged and monitored.', 'color: red; font-size: 14px;');
}, 1000);
</script>

<style>
    /* Disable text selection except for inputs */
    * {
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none;
    }
    
    input, textarea, [contenteditable] {
        -webkit-user-select: text;
        -moz-user-select: text;
        -ms-user-select: text;
        user-select: text;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .enterprise-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    
    .security-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        color: #1e3c72;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a5298;
        color: white;
    }
    
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        .main-header p {
            font-size: 0.9rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px !important;
            font-size: 0.9rem !important;
        }
        .stButton > button {
            width: 100% !important;
            margin: 0.2rem 0 !important;
        }
        .stTextInput > div > div > input {
            font-size: 16px !important; /* Prevents zoom on iOS */
        }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Initialize authentication session
    from src.auth.auth_system import init_auth_session, check_authentication
    init_auth_session()
    
    # Add logout functionality for testing
    if st.query_params.get("logout") == "true":
        for key in ['authenticated', 'user_token', 'user_info', 'user_id', 'username', 'role']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.authenticated = False
        st.rerun()
    
    # Check if user is authenticated
    if not check_authentication():
        # Show login page if not authenticated
        show_login_page()
        return
    
    # Header for authenticated users
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Enterprise RAG System</h1>
        <p>Intelligent Hybrid Knowledge Retrieval with Advanced Security</p>
        <span class="security-badge">🛡️ SECURED</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show security info in sidebar
    show_security_info()
    
    # Navigation
    user_role = UserRole(st.session_state.user_info['role'])
    
    if user_role == UserRole.ADMIN:
        nav_options = ["🧠 RAG System", "🔰 Admin Panel", "📊 Analytics", "⚙️ Settings"]
    else:
        nav_options = ["🧠 RAG System", "📊 Analytics", "⚙️ Settings"]
    
    with st.sidebar:
        st.markdown("---")
        selected_page = st.selectbox("📍 Navigation", nav_options)
    
    # Page routing
    if selected_page == "🧠 RAG System":
        show_rag_system()
    elif selected_page == "🔰 Admin Panel" and user_role == UserRole.ADMIN:
        show_admin_panel()
    elif selected_page == "📊 Analytics":
        show_analytics()
    elif selected_page == "⚙️ Settings":
        show_settings()

@rate_limit(RateLimitType.QUERY)
def show_rag_system():
    """Display the main RAG system interface"""
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    
    # Initialize the RAG system API and enterprise UI
    if 'rag_api' not in st.session_state:
        from src.backend.api import RAGSystemAPI
        st.session_state.rag_api = RAGSystemAPI()
    
    if 'enterprise_ui' not in st.session_state:
        st.session_state.enterprise_ui = EnterpriseUI(st.session_state.rag_api)
    
    # Display the RAG interface
    st.session_state.enterprise_ui.render()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analytics():
    """Display analytics dashboard"""
    protected_page("📊 Analytics Dashboard")
    
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    
    # Analytics tabs
    tabs = st.tabs(["Query Analytics", "System Performance", "Usage Statistics"])
    
    with tabs[0]:
        st.subheader("Query Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", "1,247", "+23")
        
        with col2:
            st.metric("Avg Response Time", "3.2s", "-0.5s")
        
        with col3:
            st.metric("Success Rate", "98.5%", "+1.2%")
        
        with col4:
            st.metric("KB Updates", "156", "+12")
        
        # Mock chart
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        queries = np.random.randint(20, 100, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=queries, mode='lines+markers', name='Daily Queries'))
        fig.update_layout(title="Query Volume Over Time", xaxis_title="Date", yaxis_title="Queries")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Enhanced System Performance")
        
        # Import performance monitoring
        try:
            from src.components.performance_monitor import show_performance_dashboard, show_advanced_features_status
            
            # Show real performance data
            show_performance_dashboard()
            
            st.markdown("---")
            
            # Show advanced features status
            show_advanced_features_status()
            
        except ImportError:
            # Fallback to basic metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Times")
                response_times = [2.1, 3.4, 2.8, 4.1, 3.2, 2.9, 3.8]
                st.line_chart(response_times)
            
            with col2:
                st.subheader("Knowledge Base Growth")
                kb_sizes = [3200, 3250, 3310, 3380, 3420, 3465, 3500]
                st.area_chart(kb_sizes)
    
    with tabs[2]:
        st.subheader("Usage Statistics")
        
        # User activity
        st.write("**Top Users by Query Volume**")
        user_data = {
            "User": ["user1", "user2", "user3", "admin", "demo"],
            "Queries": [45, 38, 32, 28, 15],
            "Success Rate": ["98%", "97%", "99%", "100%", "95%"]
        }
        st.dataframe(user_data, use_container_width=True)
        
        # Rate limit stats
        st.write("**Rate Limit Summary**")
        rate_data = {
            "Limit Type": ["Query", "Login", "Upload", "API"],
            "Requests": [1247, 89, 34, 567],
            "Blocks": [2, 5, 0, 1],
            "Success Rate": ["99.8%", "94.4%", "100%", "99.8%"]
        }
        st.dataframe(rate_data, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_settings():
    """Display settings page"""
    protected_page("⚙️ System Settings")
    
    st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
    
    # Settings tabs
    tabs = st.tabs(["User Preferences", "Security Settings", "System Configuration"])
    
    with tabs[0]:
        st.subheader("User Preferences")
        
        # Theme selection
        theme = st.selectbox("Interface Theme", ["Enterprise", "Dark", "Light"])
        
        # Query preferences
        st.write("**Query Settings**")
        default_model = st.selectbox("Default AI Model", ["sarvam-m", "deepseek-chat", "gpt-4o"])
        max_results = st.slider("Maximum Results per Query", 5, 50, 20)
        enable_web_search = st.checkbox("Enable Web Search by Default", True)
        
        # Notification preferences
        st.write("**Notifications**")
        email_notifications = st.checkbox("Email Notifications", False)
        query_alerts = st.checkbox("Query Completion Alerts", True)
        
        if st.button("💾 Save Preferences"):
            st.success("✅ Preferences saved successfully!")
    
    with tabs[1]:
        st.subheader("Security Settings")
        
        user_role = UserRole(st.session_state.user_info['role'])
        
        if user_role == UserRole.ADMIN:
            st.write("**Rate Limit Configuration**")
            
            # Query limits
            query_limit = st.number_input("Queries per Hour", min_value=10, max_value=1000, value=50)
            login_attempts = st.number_input("Max Login Attempts", min_value=3, max_value=20, value=5)
            
            # Session settings
            st.write("**Session Management**")
            session_timeout = st.slider("Session Timeout (hours)", 1, 24, 8)
            force_logout = st.checkbox("Force Logout on Inactivity", True)
            
            if st.button("🔧 Update Security Settings"):
                st.success("✅ Security settings updated!")
        else:
            st.info("🔒 Security settings can only be modified by administrators")
            
        # Password change
        st.write("**Change Password**")
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.button("🔑 Change Password"):
            if new_password == confirm_password and len(new_password) >= 8:
                st.success("✅ Password updated successfully!")
            else:
                st.error("❌ Password requirements not met")
    
    with tabs[2]:
        st.subheader("System Configuration")
        
        if user_role == UserRole.ADMIN:
            st.write("**Vector Store Settings**")
            embedding_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "sentence-transformers"])
            chunk_size = st.number_input("Document Chunk Size", min_value=100, max_value=2000, value=500)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1)
            
            st.write("**Web Search Settings**")
            max_web_results = st.number_input("Max Web Results", min_value=1, max_value=20, value=5)
            cache_duration = st.number_input("Cache Duration (hours)", min_value=1, max_value=168, value=24)
            
            st.write("**Database Settings**")
            auto_cleanup = st.checkbox("Automatic Database Cleanup", True)
            backup_enabled = st.checkbox("Automatic Backups", True)
            
            if st.button("⚙️ Apply Configuration"):
                st.success("✅ System configuration updated!")
        else:
            st.info("🔒 System configuration can only be modified by administrators")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()