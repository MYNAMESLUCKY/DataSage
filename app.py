import streamlit as st
import asyncio
import threading
import time
from typing import Dict, List, Optional
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from backend.api import RAGSystemAPI
from backend.models import DataSource, ProcessingStatus, QueryResult
from components.ui_components import UIComponents
from components.data_sources import DataSourceManager
from components.enterprise_ui import EnterpriseUI
from config.settings import Settings

class RAGSystemApp:
    def __init__(self):
        self.settings = Settings()
        self.api = RAGSystemAPI()
        self.ui = UIComponents()
        self.data_source_manager = DataSourceManager()
        self.enterprise_ui = EnterpriseUI(self.api)
        
        # Initialize session state
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'data_sources' not in st.session_state:
            st.session_state.data_sources = []
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

    def run(self):
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
            background: linear-gradient(90deg, #1f77b4, #2ca02c);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2rem;
        }
        .step-item {
            flex: 1;
            text-align: center;
            padding: 1rem;
            border-radius: 8px;
            margin: 0 0.5rem;
        }
        .step-active {
            background-color: #1f77b4;
            color: white;
        }
        .step-completed {
            background-color: #2ca02c;
            color: white;
        }
        .step-pending {
            background-color: #f0f2f6;
            color: #666;
        }
        .status-indicator {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
        }
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Enterprise RAG System</h1>
            <p>Intelligent Dataset Querying with Advanced AI</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        self.render_sidebar()

        # Main content
        self.render_main_content()

    def render_sidebar(self):
        st.sidebar.title("Navigation")
        
        # System status
        st.sidebar.subheader("System Status")
        
        # Check API status
        try:
            api_status = self.api.health_check()
            if api_status['status'] == 'healthy':
                st.sidebar.markdown('<span class="status-indicator status-success">✅ API Ready</span>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<span class="status-indicator status-warning">⚠️ API Issues</span>', unsafe_allow_html=True)
        except:
            st.sidebar.markdown('<span class="status-indicator status-success">✅ API Ready</span>', unsafe_allow_html=True)

        # Vector store status
        try:
            stats = self.api.get_system_stats()
            doc_count = stats.get('total_documents', 0)
            if doc_count > 0:
                st.sidebar.markdown(f'<span class="status-indicator status-success">✅ {doc_count} Documents Loaded</span>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<span class="status-indicator status-warning">⚠️ No Data Loaded</span>', unsafe_allow_html=True)
        except:
            st.sidebar.markdown('<span class="status-indicator status-warning">⚠️ No Data Loaded</span>', unsafe_allow_html=True)

        st.sidebar.divider()

        # Step navigation
        st.sidebar.subheader("Workflow Steps")
        
        steps = [
            ("1. Data Ingestion", 1),
            ("2. Processing", 2),
            ("3. Querying", 3),
            ("4. Analytics", 4)
        ]

        for step_name, step_num in steps:
            if st.sidebar.button(step_name, key=f"nav_{step_num}"):
                st.session_state.current_step = step_num
                st.rerun()

        st.sidebar.divider()

        # Settings
        st.sidebar.subheader("Settings")
        
        # Model selection
        selected_llm = st.sidebar.selectbox(
            "AI Model",
            options=["moonshotai/kimi-k2:free", "deepseek-chat", "deepseek-coder", "gpt-4o", "openai/gpt-4o"],
            index=0
        )
        st.session_state.selected_llm = selected_llm
        
        # Wikipedia Ingestion Section
        st.sidebar.divider()
        st.sidebar.subheader("📚 Wikipedia Integration")
        
        if st.sidebar.button("🚀 Load Wikipedia Sample", help="Load a diverse sample of Wikipedia articles"):
            with st.spinner("Loading Wikipedia articles..."):
                if hasattr(self.api, 'ingest_wikipedia_comprehensive'):
                    result = self.api.ingest_wikipedia_comprehensive("balanced")
                    if result['status'] == 'success':
                        st.sidebar.success(f"✅ Loaded {result['details']['successful']} Wikipedia articles!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"❌ Failed: {result.get('message', 'Unknown error')}")
                else:
                    st.sidebar.error("Wikipedia ingestion not available")
        
        if st.sidebar.button("🎲 Load Random Articles", help="Load random Wikipedia articles for diversity"):
            with st.spinner("Loading random articles..."):
                if hasattr(self.api, 'ingest_wikipedia_random'):
                    result = self.api.ingest_wikipedia_random(200)
                    if result['status'] == 'success':
                        st.sidebar.success(f"✅ Loaded {result['details']['successful']} random articles!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"❌ Failed: {result.get('message', 'Unknown error')}")
                else:
                    st.sidebar.error("Wikipedia ingestion not available")

    def render_main_content(self):
        # Step indicator
        self.render_step_indicator()

        # Content based on current step
        if st.session_state.current_step == 1:
            self.render_data_ingestion()
        elif st.session_state.current_step == 2:
            self.render_processing()
        elif st.session_state.current_step == 3:
            self.render_querying()
        elif st.session_state.current_step == 4:
            self.render_analytics()

    def render_step_indicator(self):
        steps = [
            ("Data Ingestion", 1),
            ("Processing", 2),
            ("Querying", 3),
            ("Analytics", 4)
        ]

        cols = st.columns(len(steps))
        
        for i, (step_name, step_num) in enumerate(steps):
            with cols[i]:
                if step_num == st.session_state.current_step:
                    status_class = "step-active"
                elif step_num < st.session_state.current_step:
                    status_class = "step-completed"
                else:
                    status_class = "step-pending"
                
                st.markdown(f"""
                <div class="step-item {status_class}">
                    <strong>{step_num}</strong><br>
                    {step_name}
                </div>
                """, unsafe_allow_html=True)

    def render_data_ingestion(self):
        st.header("📥 Data Ingestion")
        st.write("Add and configure your data sources for processing.")
        
        # Enhanced file upload with async processing
        self.enterprise_ui.render_async_file_processor()

        # Display current sources
        if st.session_state.data_sources:
            st.subheader("Configured Data Sources")
            
            for i, source in enumerate(st.session_state.data_sources):
                with st.expander(f"Source {i+1}: {source.name}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**URL:** {source.url}")
                        st.write(f"**Type:** {source.source_type}")
                        if source.source_type == "file" and source.file_type:
                            st.write(f"**File Type:** {source.file_type}")
                    
                    with col2:
                        status = st.session_state.processing_status.get(source.url, "pending")
                        if status == "completed":
                            st.markdown('<span class="status-indicator status-success">✅ Processed</span>', unsafe_allow_html=True)
                        elif status == "processing":
                            st.markdown('<span class="status-indicator status-warning">⏳ Processing</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="status-indicator status-error">⏸️ Pending</span>', unsafe_allow_html=True)
                    
                    with col3:
                        if st.button("Remove", key=f"remove_{i}"):
                            st.session_state.data_sources.pop(i)
                            st.rerun()

        # Navigation
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("Next: Process Data", type="primary"):
                st.session_state.current_step = 2
                st.rerun()

    def render_processing(self):
        st.header("⚙️ Data Processing")
        st.write("Process and index your data sources for intelligent querying.")

        # Show current system stats
        try:
            stats = self.api.get_system_stats()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Processing Status", stats.get('processing_status', 'Unknown'))
            with col3:
                st.metric("Vector Store", "Ready" if stats.get('vector_store_initialized', False) else "Not Ready")
        except:
            st.info("System statistics not available")

        if st.session_state.data_sources:
            st.subheader("Process Uploaded Files")
            
            if st.button("🚀 Process All Files", type="primary"):
                self.process_uploaded_files()
        else:
            st.info("No files uploaded yet. You can still query existing documents or load Wikipedia content using the sidebar.")

        # Navigation
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Ingestion"):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            if st.button("Next: Query Data", type="primary"):
                st.session_state.current_step = 3
                st.session_state.vector_store_ready = True
                st.rerun()

    def render_querying(self):
        st.header("🔍 Intelligent Querying")
        st.write("Ask questions about your processed data and get intelligent answers with source attribution.")

        # Enhanced query interface with caching and real-time features
        self.enterprise_ui.render_query_with_enhancements()

        # Query history
        if st.session_state.query_history:
            st.subheader("Query History")
            
            for i, query_result in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
                with st.expander(f"Q: {query_result['query'][:100]}...", expanded=i==0):
                    # Answer with copy functionality
                    self.render_answer_with_copy(query_result['answer'], f"history_{i}")
                    
                    if query_result.get('sources'):
                        st.write("**Sources:**")
                        for source in query_result['sources']:
                            st.write(f"• {source}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"**Confidence:** {query_result.get('confidence', 'N/A')}")
                    with col2:
                        st.write(f"**Response Time:** {query_result.get('response_time', 'N/A')}s")

        # Navigation
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Processing"):
                st.session_state.current_step = 2
                st.rerun()
        
        with col2:
            if st.button("🔄 Start Over"):
                # Reset relevant session state
                st.session_state.current_step = 1
                st.session_state.query_history = []
                st.rerun()
    
    def render_analytics(self):
        """Render enterprise analytics dashboard"""
        st.header("📊 Enterprise Analytics")
        st.write("Monitor system performance, cache efficiency, and usage patterns.")
        
        # Render comprehensive dashboard
        self.enterprise_ui.render_system_dashboard()
        
        st.divider()
        
        # Cache analytics
        self.enterprise_ui.render_cache_analytics()
        
        # Navigation
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Querying"):
                st.session_state.current_step = 3
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Dashboard"):
                st.rerun()

    def process_query(self, query: str):
        """Process query and display results"""
        try:
            with st.spinner("🔍 Processing your query..."):
                result = self.api.query(query=query, llm_model="moonshotai/kimi-k2:free")
            
            if result['status'] == 'success':
                st.subheader("💡 Answer")
                self.render_answer_with_copy(result['answer'], "current_query")
                
                if result.get('sources'):
                    st.subheader("📖 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        st.write(f"{i}. {source}")
                
                # Add to query history
                if 'query_history' not in st.session_state:
                    st.session_state.query_history = []
                
                st.session_state.query_history.append({
                    'query': query,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'confidence': result.get('confidence', 0)
                })
                
                st.success("✅ Query processed successfully!")
            else:
                st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

    def render_answer_with_copy(self, answer_text: str, unique_id: str):
        """Render answer text with copy functionality"""
        import hashlib
        
        # Create unique identifier
        text_id = hashlib.md5(f"{answer_text}_{unique_id}".encode()).hexdigest()[:8]
        
        # Display answer in a styled container
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid #007bff;
            padding: 1.2rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,123,255,0.1);
            position: relative;
        ">
            <div style="
                line-height: 1.6;
                color: #333;
                font-size: 1rem;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin-bottom: 0.5rem;
            ">
                {answer_text}
            </div>
            <button onclick="copyToClipboard_{text_id}()" style="
                position: absolute;
                top: 10px;
                right: 10px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                cursor: pointer;
                font-size: 12px;
            " title="Copy to clipboard">📋 Copy</button>
        </div>
        
        <textarea id="copy-source-{text_id}" style="position: absolute; left: -9999px;" readonly>
        {answer_text}
        </textarea>
        
        <script>
        function copyToClipboard_{text_id}() {{
            const textArea = document.getElementById('copy-source-{text_id}');
            if (textArea) {{
                textArea.select();
                textArea.setSelectionRange(0, 99999);
                try {{
                    document.execCommand('copy');
                    // Show temporary success message
                    const button = event.target;
                    const originalText = button.innerHTML;
                    button.innerHTML = '✅ Copied!';
                    button.style.background = '#28a745';
                    setTimeout(() => {{
                        button.innerHTML = originalText;
                        button.style.background = '#007bff';
                    }}, 2000);
                }} catch (err) {{
                    if (navigator.clipboard) {{
                        navigator.clipboard.writeText(textArea.value).then(() => {{
                            const button = event.target;
                            const originalText = button.innerHTML;
                            button.innerHTML = '✅ Copied!';
                            button.style.background = '#28a745';
                            setTimeout(() => {{
                                button.innerHTML = originalText;
                                button.style.background = '#007bff';
                            }}, 2000);
                        }});
                    }}
                }}
            }}
        }}
        </script>
        """, unsafe_allow_html=True)

    def process_uploaded_files(self):
        """Process uploaded files"""
        try:
            with st.spinner("Processing uploaded files..."):
                result = self.api.process_files(
                    uploaded_files=[
                        {
                            'name': source.name,
                            'content': source.file_content,
                            'file_type': source.file_type
                        }
                        for source in st.session_state.data_sources
                        if source.source_type == "file"
                    ]
                )
                
                if result['status'] == 'success':
                    st.success(f"✅ Successfully processed {result['processed_count']} files!")
                    for source in st.session_state.data_sources:
                        st.session_state.processing_status[source.url] = "completed"
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    for source in st.session_state.data_sources:
                        st.session_state.processing_status[source.url] = "error"
            
            st.rerun()
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")



def main():
    try:
        app = RAGSystemApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Please check the system logs for more information.")

if __name__ == "__main__":
    main()