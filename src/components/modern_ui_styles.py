"""
Modern UI Styles for Enterprise RAG System
Enhanced visual design components and styling
"""

import streamlit as st

def apply_modern_styling():
    """Apply modern, professional styling to the Streamlit app"""
    
    st.markdown("""
    <style>
    /* Global Styles - Clean and Simple */
    .stApp {
        background: #ffffff;
        min-height: 100vh;
    }
    
    /* Main container - minimal design */
    .main-container {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling - simple and clean */
    .header-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    /* Tab styling - clean and minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8f9fa;
        border-radius: 6px;
        padding: 0.25rem;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0px 20px;
        background: transparent;
        border-radius: 4px;
        border: none;
        color: #495057;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: #2c3e50;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #dee2e6;
    }
    
    /* Metric cards - clean design */
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border-color: #007bff;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Button styling - simple and clean */
    .stButton > button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
    }
    
    .stButton > button:hover {
        background: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    /* Select box styling */
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        color: white;
    }
    
    .status-info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    
    /* Chart containers - minimal styling */
    .chart-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Agent status cards */
    .agent-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px 0 rgba(31, 38, 135, 0.3);
    }
    
    .agent-active {
        border-left: 4px solid #4facfe;
        background: rgba(79, 172, 254, 0.1);
    }
    
    .agent-completed {
        border-left: 4px solid #00f2fe;
        background: rgba(0, 242, 254, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    /* Notification styles */
    .notification {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #4facfe;
        margin: 1rem 0;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Clean text styling */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* Make plotly charts blend better */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    .plotly .modebar {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a clean metric card"""
    delta_html = ""
    if delta:
        color = "#28a745" if delta_color == "normal" else "#dc3545" if delta_color == "inverse" else "#28a745"
        delta_html = f'<div style="color: {color}; font-size: 0.85rem; margin-top: 0.5rem; font-weight: 500;">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {delta_html}
    </div>
    """

def create_status_badge(text: str, status: str = "info"):
    """Create a status badge"""
    return f'<span class="status-badge status-{status}">{text}</span>'

def create_agent_card(agent_name: str, status: str, description: str, active: bool = False):
    """Create an agent status card"""
    status_class = "agent-active" if active else "agent-completed" if status == "completed" else ""
    
    return f"""
    <div class="agent-card {status_class} fade-in">
        <h4 style="margin: 0 0 0.5rem 0; color: white;">{agent_name}</h4>
        <p style="margin: 0 0 0.5rem 0; color: rgba(255, 255, 255, 0.8);">{description}</p>
        {create_status_badge(status, "success" if status == "completed" else "info")}
    </div>
    """

def create_notification(message: str, type: str = "info"):
    """Create a notification"""
    return f"""
    <div class="notification fade-in">
        <div style="color: white;">{message}</div>
    </div>
    """

def create_loading_indicator(text: str = "Processing..."):
    """Create a loading indicator"""
    return f"""
    <div style="display: flex; align-items: center; color: white; padding: 1rem;">
        <div class="loading-spinner" style="margin-right: 1rem;"></div>
        <span>{text}</span>
    </div>
    """

def create_hero_section(title: str, subtitle: str):
    """Create a hero section"""
    return f"""
    <div class="header-container fade-in">
        <h1 class="header-title">{title}</h1>
        <p class="header-subtitle">{subtitle}</p>
    </div>
    """

def wrap_in_container(content: str):
    """Wrap content in the main container"""
    return f"""
    <div class="main-container">
        {content}
    </div>
    """