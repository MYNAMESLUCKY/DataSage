"""
Enterprise RAG System - Main Application Entry Point
Redirects to enterprise_app.py for proper authentication flow
"""

import streamlit as st
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point that redirects to enterprise authentication system"""
    st.set_page_config(
        page_title="Enterprise RAG System",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Import and run the enterprise application
    try:
        # Import the enterprise app main function
        from enterprise_app import main as enterprise_main
        
        # Run the enterprise application with authentication
        enterprise_main()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please contact support if this issue persists.")
        
        # Fallback error information
        st.markdown("---")
        st.markdown("### System Information")
        st.write(f"Python version: {sys.version}")
        st.write(f"Working directory: {os.getcwd()}")

if __name__ == "__main__":
    main()