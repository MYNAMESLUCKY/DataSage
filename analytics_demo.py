"""
Enterprise Analytics Dashboard Demo
Standalone demo of the analytics dashboard capabilities
"""

import streamlit as st
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Analytics Dashboard Demo"""
    st.set_page_config(
        page_title="Enterprise Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        from src.analytics.dashboard import analytics_dashboard
        from src.analytics.system_monitor import system_monitor
        
        # Initialize monitoring if not already started
        if not system_monitor.monitoring:
            system_monitor.start_monitoring(interval=30)
        
        # Render the dashboard
        analytics_dashboard.render_dashboard()
        
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.info("Make sure all required packages are installed.")
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        
        # Fallback: Basic system info
        st.header("📊 Basic System Information")
        
        import psutil
        from datetime import datetime
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_percent = psutil.cpu_percent(interval=1)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
        with col2:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
        with col3:
            disk = psutil.disk_usage('/')
            st.metric("Disk Usage", f"{disk.percent:.1f}%")
            
        st.info("This is a basic fallback view. The full analytics dashboard provides much more detailed insights.")

if __name__ == "__main__":
    main()