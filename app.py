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
from config.settings import Settings

class RAGSystemApp:
    def __init__(self):
        self.settings = Settings()
        self.api = RAGSystemAPI()
        self.ui = UIComponents()
        self.data_source_manager = DataSourceManager()
        
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
        .query-result {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .source-attribution {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.9rem;
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
        except Exception as e:
            st.sidebar.markdown('<span class="status-indicator status-error">❌ API Offline</span>', unsafe_allow_html=True)

        # Vector store status
        if st.session_state.vector_store_ready:
            st.sidebar.markdown('<span class="status-indicator status-success">✅ Vector Store Ready</span>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="status-indicator status-warning">⚠️ No Data Loaded</span>', unsafe_allow_html=True)

        st.sidebar.divider()

        # Step navigation
        st.sidebar.subheader("Workflow Steps")
        
        steps = [
            ("1. Data Ingestion", 1),
            ("2. Processing", 2),
            ("3. Querying", 3)
        ]

        for step_name, step_num in steps:
            if st.sidebar.button(step_name, key=f"nav_{step_num}"):
                st.session_state.current_step = step_num

        st.sidebar.divider()

        # Settings
        st.sidebar.subheader("Settings")
        
        # Model selection
        selected_llm = st.sidebar.selectbox(
            "LLM Model",
            options=["gpt-4o", "gpt-3.5-turbo", "huggingface"],
            index=0
        )
        st.session_state.selected_llm = selected_llm

        # Embedding model
        selected_embedding = st.sidebar.selectbox(
            "Embedding Model",
            options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "distilbert-base-nli-mean-tokens"],
            index=0
        )
        st.session_state.selected_embedding = selected_embedding

        # Chunk size
        chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 512)
        st.session_state.chunk_size = chunk_size

        # Overlap
        chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 50)
        st.session_state.chunk_overlap = chunk_overlap

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

    def render_step_indicator(self):
        steps = [
            ("Data Ingestion", 1),
            ("Processing", 2),
            ("Querying", 3)
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

        # Data source input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            source_url = st.text_input(
                "Data Source URL",
                placeholder="Enter URL to scrape data from...",
                help="Supports web pages, APIs, and other online data sources"
            )
        
        with col2:
            if st.button("Add Source", type="primary", disabled=not source_url):
                if source_url:
                    try:
                        # Validate and add source
                        source = DataSource(
                            url=source_url,
                            source_type="web",
                            name=f"Source {len(st.session_state.data_sources) + 1}"
                        )
                        st.session_state.data_sources.append(source)
                        st.success(f"Added source: {source_url}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding source: {str(e)}")

        # Display current sources
        if st.session_state.data_sources:
            st.subheader("Configured Data Sources")
            
            for i, source in enumerate(st.session_state.data_sources):
                with st.expander(f"Source {i+1}: {source.name}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**URL:** {source.url}")
                        st.write(f"**Type:** {source.source_type}")
                    
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

        # Suggested sources
        st.subheader("Suggested Data Sources")
        suggestions = self.data_source_manager.get_suggested_sources()
        
        suggestion_cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i % 3]:
                if st.button(f"Add {suggestion['name']}", key=f"suggest_{i}"):
                    source = DataSource(
                        url=suggestion['url'],
                        source_type=suggestion['type'],
                        name=suggestion['name']
                    )
                    st.session_state.data_sources.append(source)
                    st.rerun()

        # Navigation
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("Next: Process Data", type="primary", disabled=not st.session_state.data_sources):
                st.session_state.current_step = 2
                st.rerun()

    def render_processing(self):
        st.header("⚙️ Data Processing")
        st.write("Process and index your data sources for intelligent querying.")

        if not st.session_state.data_sources:
            st.warning("No data sources configured. Please go back to Data Ingestion.")
            return

        # Processing configuration
        with st.expander("Processing Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Chunk Size:** {st.session_state.get('chunk_size', 512)}")
                st.write(f"**Chunk Overlap:** {st.session_state.get('chunk_overlap', 50)}")
            
            with col2:
                st.write(f"**Embedding Model:** {st.session_state.get('selected_embedding', 'all-MiniLM-L6-v2')}")
                st.write(f"**Total Sources:** {len(st.session_state.data_sources)}")

        # Process all sources
        if st.button("🚀 Start Processing", type="primary"):
            self.process_all_sources()

        # Show processing status
        if st.session_state.processing_status:
            st.subheader("Processing Status")
            
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            completed = sum(1 for status in st.session_state.processing_status.values() if status == "completed")
            total = len(st.session_state.data_sources)
            progress = completed / total if total > 0 else 0
            
            progress_bar.progress(progress)
            status_placeholder.write(f"Processed {completed}/{total} sources")

            # Detailed status for each source
            for source in st.session_state.data_sources:
                status = st.session_state.processing_status.get(source.url, "pending")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{source.name}** - {source.url}")
                
                with col2:
                    if status == "completed":
                        st.success("✅ Done")
                    elif status == "processing":
                        st.info("⏳ Processing...")
                    elif status == "error":
                        st.error("❌ Error")
                    else:
                        st.warning("⏸️ Pending")

        # Navigation
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Ingestion"):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            processing_complete = all(
                st.session_state.processing_status.get(source.url) == "completed"
                for source in st.session_state.data_sources
            )
            if st.button("Next: Query Data", type="primary", disabled=not processing_complete):
                st.session_state.current_step = 3
                st.session_state.vector_store_ready = True
                st.rerun()

    def render_querying(self):
        st.header("🔍 Intelligent Querying")
        st.write("Ask questions about your processed data and get intelligent answers with source attribution.")

        if not st.session_state.vector_store_ready:
            st.warning("No processed data available. Please complete the processing step first.")
            return

        # Query interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Ask a question about your data:",
                placeholder="What insights can you provide from the processed data?",
                key="query_input"
            )
        
        with col2:
            if st.button("🔍 Query", type="primary", disabled=not query):
                if query:
                    self.process_query(query)

        # Suggested queries
        st.subheader("Suggested Queries")
        suggested_queries = [
            "What are the main topics covered in the data?",
            "Summarize the key insights from all sources",
            "What trends or patterns can you identify?",
            "Compare different perspectives from the sources"
        ]

        query_cols = st.columns(2)
        for i, suggested_query in enumerate(suggested_queries):
            with query_cols[i % 2]:
                if st.button(suggested_query, key=f"suggest_query_{i}"):
                    self.process_query(suggested_query)

        # Query history
        if st.session_state.query_history:
            st.subheader("Query History")
            
            for i, query_result in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"Q: {query_result['query'][:100]}...", expanded=i==0):
                    st.markdown(f"""
                    <div class="query-result">
                        <h4>Answer:</h4>
                        <p>{query_result['answer']}</p>
                        
                        <div class="source-attribution">
                            <h5>Sources:</h5>
                            <ul>
                    """, unsafe_allow_html=True)
                    
                    for source in query_result.get('sources', []):
                        st.markdown(f"<li>{source}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div></div>", unsafe_allow_html=True)
                    
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
                # Reset all session state
                for key in list(st.session_state.keys()):
                    if key not in ['selected_llm', 'selected_embedding', 'chunk_size', 'chunk_overlap']:
                        del st.session_state[key]
                st.session_state.current_step = 1
                st.rerun()

    def process_all_sources(self):
        """Process all configured data sources"""
        try:
            for source in st.session_state.data_sources:
                st.session_state.processing_status[source.url] = "processing"
            
            # Start processing in background
            with st.spinner("Processing data sources..."):
                result = self.api.process_sources(
                    sources=st.session_state.data_sources,
                    chunk_size=st.session_state.get('chunk_size', 512),
                    chunk_overlap=st.session_state.get('chunk_overlap', 50),
                    embedding_model=st.session_state.get('selected_embedding', 'all-MiniLM-L6-v2')
                )
                
                if result['status'] == 'success':
                    for source in st.session_state.data_sources:
                        st.session_state.processing_status[source.url] = "completed"
                    st.success("✅ All sources processed successfully!")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    for source in st.session_state.data_sources:
                        st.session_state.processing_status[source.url] = "error"
            
            st.rerun()
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            for source in st.session_state.data_sources:
                st.session_state.processing_status[source.url] = "error"

    def process_query(self, query: str):
        """Process a user query and display results"""
        try:
            start_time = time.time()
            
            with st.spinner("Searching and generating answer..."):
                result = self.api.query(
                    query=query,
                    llm_model=st.session_state.get('selected_llm', 'gpt-4o'),
                    max_results=5
                )
                
                response_time = round(time.time() - start_time, 2)
                
                if result['status'] == 'success':
                    query_result = {
                        'query': query,
                        'answer': result['answer'],
                        'sources': result.get('sources', []),
                        'confidence': result.get('confidence', 'N/A'),
                        'response_time': response_time
                    }
                    
                    st.session_state.query_history.append(query_result)
                    st.success("✅ Query processed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    app = RAGSystemApp()
    app.run()
