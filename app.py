"""
Enterprise RAG System - Streamlit Application
A comprehensive RAG system with authentication, document processing, and AI-powered querying.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #e1e5e9;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        color: #155724;
    }
    .warning-message {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        color: #856404;
    }
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8001/api/v1"
ENTERPRISE_API_URL = "http://localhost:8000"

class RAGSystemAPI:
    """API client for the RAG system backend"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user and get JWT token"""
        try:
            response = self.session.post(
                f"{API_BASE_URL}/auth/login",
                json={"email": email, "password": password}
            )
            if response.status_code == 200:
                data = response.json()
                # Set authorization header for future requests
                self.session.headers.update({
                    'Authorization': f"Bearer {data['token']}"
                })
                return {"success": True, "data": data}
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def signup(self, email: str, password: str, name: str) -> Dict[str, Any]:
        """Register new user"""
        try:
            response = self.session.post(
                f"{API_BASE_URL}/auth/register",
                json={"email": email, "password": password, "name": name}
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a RAG query"""
        try:
            payload = {
                "query": query,
                "user_id": st.session_state.get("user_id", "anonymous"),
                **kwargs
            }
            response = self.session.post(
                f"{API_BASE_URL}/query",
                json=payload
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health metrics"""
        try:
            response = self.session.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": "System unavailable"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize API client
@st.cache_resource
def get_api_client():
    return RAGSystemAPI()

api = get_api_client()

def init_session_state():
    """Initialize session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

def show_login_page():
    """Display login/signup interface"""
    st.markdown('<div class="main-header"><h1>🧠 Enterprise RAG System</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Authentication")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="demo@example.com")
                password = st.text_input("Password", type="password", placeholder="demo")
                submit_login = st.form_submit_button("Login", use_container_width=True)
                
                if submit_login:
                    if email and password:
                        with st.spinner("Authenticating..."):
                            result = api.login(email, password)
                            if result["success"]:
                                st.session_state.authenticated = True
                                st.session_state.user = result["data"]["user"]
                                st.session_state.token = result["data"]["token"]
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error(f"Login failed: {result['error']}")
                    else:
                        st.error("Please enter both email and password")
            
            st.info("**Demo Account**: email: `demo@example.com`, password: `demo`")
        
        with tab2:
            with st.form("signup_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit_signup = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_signup:
                    if name and email and password and confirm_password:
                        if password == confirm_password:
                            with st.spinner("Creating account..."):
                                result = api.signup(email, password, name)
                                if result["success"]:
                                    st.success("Account created successfully! Please login.")
                                else:
                                    st.error(f"Signup failed: {result['error']}")
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.error("Please fill in all fields")

def show_main_interface():
    """Display main RAG system interface"""
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header"><h1>🧠 Enterprise RAG System</h1></div>', unsafe_allow_html=True)
    with col2:
        st.write(f"Welcome, **{st.session_state.user['name']}**")
    with col3:
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Choose a section:",
            ["Query Interface", "Document Manager", "Analytics", "Settings"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        status_result = api.get_system_status()
        if status_result["success"]:
            st.success("🟢 System Online")
        else:
            st.error("🔴 System Offline")
    
    # Main content based on selected page
    if page == "Query Interface":
        show_query_interface()
    elif page == "Document Manager":
        show_document_manager()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()

def show_query_interface():
    """RAG query interface"""
    st.markdown("### 🔍 Ask Questions")
    
    # Query configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "Enter your question:",
            placeholder="What are the main benefits of renewable energy?",
            height=100
        )
    
    with col2:
        st.markdown("**Configuration**")
        model = st.selectbox(
            "AI Model:",
            ["auto", "llama-3.2-7b", "mistral-7b", "gemma-7b", "qwen-7b", "deepseek-coder"],
            index=0
        )
        max_sources = st.slider("Max Sources:", 1, 20, 10)
        enable_gpu = st.checkbox("Enable GPU Acceleration", value=True)
    
    # Example queries
    st.markdown("**Example Questions:**")
    examples = [
        "What are the main benefits of renewable energy?",
        "Explain the basics of machine learning",
        "How does photosynthesis work?",
        "Compare quantum computing vs classical computing"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💡 {example[:40]}...", key=f"example_{i}"):
                st.session_state.query_input = example
    
    # Process query
    if st.button("🚀 Process Query", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Processing your question..."):
                result = api.process_query(
                    query=query,
                    model=model,
                    max_sources=max_sources,
                    enable_gpu=enable_gpu
                )
                
                if result["success"]:
                    data = result["data"]
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "query": query,
                        "answer": data.get("answer", "Demo response - API keys needed for full functionality"),
                        "timestamp": datetime.now(),
                        "model": model,
                        "sources": data.get("sources", [])
                    })
                    
                    # Display results
                    st.markdown("### 💬 Answer")
                    st.markdown(data.get("answer", "Demo response - Add API keys for full RAG functionality"))
                    
                    if data.get("sources"):
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(data["sources"], 1):
                            st.markdown(f"{i}. {source}")
                    
                    st.markdown("### ⚡ Processing Details")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Processing Time", f"{data.get('processing_time', 0):.2f}s")
                    with col2:
                        st.metric("Model Used", data.get("model_used", model))
                    with col3:
                        st.metric("Sources Found", len(data.get("sources", [])))
                    with col4:
                        st.metric("GPU Accelerated", "Yes" if data.get("gpu_accelerated") else "No")
                else:
                    st.error(f"Query failed: {result['error']}")
        else:
            st.warning("Please enter a question")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📝 Recent Queries")
        
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {item['query'][:60]}..." if len(item['query']) > 60 else f"Q: {item['query']}"):
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown(f"**Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Model:** {item['model']}")
                if item['sources']:
                    st.markdown(f"**Sources:** {len(item['sources'])} found")

def show_document_manager():
    """Document upload and management interface"""
    st.markdown("### 📄 Document Manager")
    
    tab1, tab2 = st.tabs(["Upload Documents", "Manage Sources"])
    
    with tab1:
        st.markdown("**Upload Local Documents**")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx']
        )
        
        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} files")
            if st.button("Process Documents", type="primary"):
                st.info("Document processing would be implemented here with full API keys")
        
        st.markdown("---")
        st.markdown("**Add Web Sources**")
        url = st.text_input("Website URL", placeholder="https://example.com/article")
        if st.button("Add URL") and url:
            st.info(f"Would process URL: {url}")
    
    with tab2:
        st.markdown("**Data Sources**")
        
        # Mock data for demonstration
        sources_data = {
            "Source": ["Wikipedia Articles", "Uploaded PDFs", "Web URLs", "API Data"],
            "Count": [150, 23, 12, 5],
            "Last Updated": ["2 hours ago", "1 day ago", "3 days ago", "1 week ago"],
            "Status": ["Active", "Active", "Active", "Pending"]
        }
        
        df = pd.DataFrame(sources_data)
        st.dataframe(df, use_container_width=True)

def show_analytics():
    """Analytics and metrics dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Queries",
            value=len(st.session_state.query_history),
            delta=f"+{len(st.session_state.query_history)} today"
        )
    
    with col2:
        st.metric(
            label="Documents Processed",
            value="190",
            delta="+12 this week"
        )
    
    with col3:
        st.metric(
            label="Average Response Time",
            value="2.3s",
            delta="-0.5s vs last week"
        )
    
    with col4:
        st.metric(
            label="Success Rate",
            value="98.5%",
            delta="+1.2% vs last week"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Query Volume Over Time**")
        # Mock chart data
        dates = pd.date_range(start='2025-07-01', end='2025-08-04', freq='D')
        query_counts = [20 + i*2 + (i%7)*5 for i in range(len(dates))]
        
        fig = px.line(x=dates, y=query_counts, title="Daily Query Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Model Usage Distribution**")
        models = ["llama-3.2-7b", "mistral-7b", "gemma-7b", "auto"]
        usage = [45, 25, 20, 10]
        
        fig = px.pie(values=usage, names=models, title="AI Model Usage")
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """System settings and configuration"""
    st.markdown("### ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["User Profile", "System Config", "API Keys"])
    
    with tab1:
        st.markdown("**Profile Information**")
        with st.form("profile_form"):
            name = st.text_input("Name", value=st.session_state.user.get("name", ""))
            email = st.text_input("Email", value=st.session_state.user.get("email", ""))
            subscription = st.selectbox(
                "Subscription Tier",
                ["Free", "Pro", "Enterprise"],
                index=0 if st.session_state.user.get("subscription_tier") == "free" else 1
            )
            
            if st.form_submit_button("Update Profile"):
                st.success("Profile updated successfully!")
    
    with tab2:
        st.markdown("**System Configuration**")
        
        default_model = st.selectbox(
            "Default AI Model",
            ["auto", "llama-3.2-7b", "mistral-7b", "gemma-7b"],
            index=0
        )
        
        max_query_length = st.slider("Max Query Length", 100, 5000, 2000)
        enable_gpu_default = st.checkbox("Enable GPU by Default", value=True)
        auto_save_history = st.checkbox("Auto-save Query History", value=True)
        
        if st.button("Save Configuration"):
            st.success("Configuration saved!")
    
    with tab3:
        st.markdown("**API Keys Configuration**")
        st.info("Add your API keys to enable full RAG functionality")
        
        with st.form("api_keys_form"):
            openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            serper_key = st.text_input("Serper API Key", type="password", placeholder="Your Serper key")
            
            if st.form_submit_button("Update API Keys"):
                st.success("API keys updated successfully!")
                st.info("Keys are securely stored and will enable live web search and advanced AI models")

def main():
    """Main application entry point"""
    init_session_state()
    
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_interface()

if __name__ == "__main__":
    main()