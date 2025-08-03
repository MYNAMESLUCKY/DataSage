"""
Modern UI Styles for Enterprise RAG System
Enhanced visual design components and styling
"""

import streamlit as st

def apply_modern_styling():
    """Apply modern, professional styling to the Streamlit app"""
    
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Main container with glassmorphism effect */
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.2);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        font-weight: 300;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0px 24px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px 0 rgba(31, 38, 135, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.4);
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
    
    /* Chart containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
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
    
    /* Dark mode text */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: white !important;
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
    """Create a modern metric card"""
    delta_html = ""
    if delta:
        color = "#4facfe" if delta_color == "normal" else "#ff6b6b" if delta_color == "inverse" else "#4facfe"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</div>'
    
    return f"""
    <div class="metric-card fade-in">
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