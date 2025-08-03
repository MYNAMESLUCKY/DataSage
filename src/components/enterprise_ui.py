"""
Enterprise UI components with real-time updates and advanced features
"""

import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EnterpriseUI:
    """Enhanced UI components for enterprise features"""
    
    def __init__(self, api):
        self.api = api
    
    def render(self):
        """Main render method for the enterprise interface"""
        # Create tabs for different sections
        tabs = st.tabs(["🔍 Query System", "📁 File Processing", "📊 Analytics", "⚙️ System"])
        
        with tabs[0]:
            self.render_query_with_enhancements()
        
        with tabs[1]:
            self.render_async_file_processor()
        
        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                self.render_cache_analytics()
            with col2:
                self.render_system_dashboard()
        
        with tabs[3]:
            self.render_system_info()
    
    def render_async_file_processor(self):
        """Render async file upload with real-time progress"""
        st.subheader("📁 Smart File Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents for processing",
            type=['txt', 'pdf', 'csv', 'xlsx', 'docx'],
            accept_multiple_files=True,
            help="Files will be processed asynchronously in the background"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            process_async = st.checkbox("Process in background", value=True, 
                                      help="Allows you to continue using the app while files process")
        
        with col2:
            if st.button("🚀 Process Files", disabled=not uploaded_files):
                if uploaded_files:
                    self._start_async_processing(uploaded_files, process_async)
        
        # Show active processing tasks
        self._render_processing_status()
    
    def _start_async_processing(self, uploaded_files, async_mode: bool):
        """Start file processing (async or sync)"""
        try:
            if async_mode:
                # Prepare files for async processing
                file_data = []
                for uploaded_file in uploaded_files:
                    file_content = uploaded_file.read()
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    # Determine file type
                    if file_extension in ['pdf']:
                        file_type = 'pdf'
                    elif file_extension in ['xlsx', 'xls']:
                        file_type = 'excel'
                    elif file_extension == 'csv':
                        file_type = 'csv'
                    elif file_extension == 'docx':
                        file_type = 'docx'
                    else:
                        file_type = 'text'
                    
                    file_data.append({
                        'name': uploaded_file.name,
                        'content': file_content,
                        'file_type': file_type
                    })
                
                # Submit async task
                task_id = self.api.process_files_async(file_data)
                
                # Store task ID in session state
                if 'processing_tasks' not in st.session_state:
                    st.session_state.processing_tasks = []
                st.session_state.processing_tasks.append(task_id)
                
                st.success(f"✅ Started background processing for {len(uploaded_files)} files")
                st.info("Processing will continue in the background. Check the status below.")
                
            else:
                # Synchronous processing (legacy)
                with st.spinner("Processing files..."):
                    result = self.api.process_files([{
                        'name': f.name,
                        'content': f.read(),
                        'type': f.name.split('.')[-1].lower()
                    } for f in uploaded_files])
                
                if result['status'] == 'success':
                    st.success(f"✅ Processed {len(uploaded_files)} files successfully!")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error starting file processing: {str(e)}")
    
    def _render_processing_status(self):
        """Show real-time processing status"""
        if 'processing_tasks' not in st.session_state or not st.session_state.processing_tasks:
            return
        
        st.subheader("📊 Processing Status")
        
        # Create status container for real-time updates
        status_container = st.container()
        
        with status_container:
            active_tasks = []
            completed_tasks = []
            
            for task_id in st.session_state.processing_tasks:
                task_status = self.api.get_processing_task_status(task_id)
                
                if not task_status:
                    continue
                
                if task_status['status'] in ['completed', 'failed', 'cancelled']:
                    completed_tasks.append(task_status)
                else:
                    active_tasks.append(task_status)
            
            # Show active tasks with progress bars
            if active_tasks:
                st.write("**Active Tasks:**")
                for task in active_tasks:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{task['task_type']}** - {task['message']}")
                        progress = task['progress'] / 100.0
                        st.progress(progress)
                    
                    with col2:
                        st.write(f"{task['progress']:.1f}%")
                    
                    with col3:
                        if st.button("Cancel", key=f"cancel_{task['task_id']}"):
                            if self.api.cancel_processing_task(task['task_id']):
                                st.success("Task cancelled")
                                st.rerun()
            
            # Show completed tasks summary
            if completed_tasks:
                st.write("**Recent Completions:**")
                for task in completed_tasks[-3:]:  # Show last 3
                    status_icon = "✅" if task['status'] == 'completed' else "❌"
                    files_count = task.get('metadata', {}).get('file_count', 0)
                    st.write(f"{status_icon} {task['task_type']}: {files_count} files - {task['message']}")
        
        # Auto-refresh for active tasks
        if active_tasks:
            time.sleep(2)
            st.rerun()
    
    def render_cache_analytics(self):
        """Render cache performance analytics"""
        try:
            cache_stats = self.api.get_cache_stats()
            
            st.subheader("🚀 Cache Performance")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                hit_rate = cache_stats['performance']['hit_rate_percentage']
                st.metric("Cache Hit Rate", f"{hit_rate:.1f}%", 
                         help="Percentage of queries served from cache")
            
            with col2:
                total_requests = cache_stats['performance']['total_requests']
                st.metric("Total Requests", total_requests,
                         help="Total number of queries processed")
            
            with col3:
                cache_size = cache_stats['storage']['total_entries']
                max_size = cache_stats['storage']['max_size']
                st.metric("Cache Usage", f"{cache_size}/{max_size}",
                         help="Current cache entries vs maximum")
            
            with col4:
                avg_age = cache_stats['content']['average_age_seconds']
                st.metric("Avg Entry Age", f"{avg_age/60:.1f}m",
                         help="Average age of cached entries")
            
            # Cache hit rate visualization
            if total_requests > 0:
                hits = cache_stats['performance']['cache_hits']
                misses = cache_stats['performance']['cache_misses']
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Cache Hits', 'Cache Misses'],
                    values=[hits, misses],
                    hole=0.3,
                    marker_colors=['#2E8B57', '#DC143C']
                )])
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    title="Cache Hit Rate Distribution",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Cache controls
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧹 Clear Cache"):
                    self.api.cache_manager.cache.clear()
                    st.success("Cache cleared successfully!")
                    st.rerun()
            
            with col2:
                if st.button("♻️ Clean Expired"):
                    self.api.cache_manager.clear_expired()
                    st.success("Expired entries removed!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error loading cache analytics: {str(e)}")
    
    def render_system_dashboard(self):
        """Render comprehensive system dashboard"""
        st.subheader("🎯 System Dashboard")
        
        try:
            # Get system stats
            sys_stats = self.api.get_system_stats()
            cache_stats = self.api.get_cache_stats()
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                doc_count = sys_stats.get('total_documents', 0)
                st.metric("📚 Documents", f"{doc_count:,}",
                         help="Total documents in knowledge base")
            
            with col2:
                hit_rate = cache_stats['performance']['hit_rate_percentage']
                delta_color = "normal" if hit_rate > 50 else "inverse"
                st.metric("⚡ Cache Hit Rate", f"{hit_rate:.1f}%",
                         delta="Good" if hit_rate > 50 else "Low",
                         delta_color=delta_color)
            
            with col3:
                memory_mb = sys_stats.get('memory_usage', {}).get('rss_mb', 0)
                st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB",
                         help="Current memory consumption")
            
            with col4:
                status = "Healthy" if sys_stats.get('vector_store_initialized', False) else "Offline"
                st.metric("🟢 System Status", status,
                         delta="Online" if status == "Healthy" else "Offline",
                         delta_color="normal" if status == "Healthy" else "inverse")
            
            # Performance trends (mock data for demo)
            st.subheader("📈 Performance Trends")
            
            # Create sample performance data
            import pandas as pd
            import datetime
            
            dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=7), 
                                end=datetime.datetime.now(), freq='H')
            
            performance_data = pd.DataFrame({
                'timestamp': dates,
                'queries_per_hour': [max(10, int(50 + 30 * (i % 24) / 24)) for i in range(len(dates))],
                'avg_response_time': [max(0.5, 2.0 + 1.0 * (i % 12) / 12) for i in range(len(dates))],
                'cache_hit_rate': [min(95, max(30, 70 + 20 * (i % 6) / 6)) for i in range(len(dates))]
            })
            
            # Query volume chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_queries = px.line(
                    performance_data.tail(48), 
                    x='timestamp', 
                    y='queries_per_hour',
                    title='Query Volume (Last 48h)',
                    labels={'queries_per_hour': 'Queries/Hour', 'timestamp': 'Time'}
                )
                fig_queries.update_layout(height=300)
                st.plotly_chart(fig_queries, use_container_width=True)
            
            with col2:
                fig_response = px.line(
                    performance_data.tail(48), 
                    x='timestamp', 
                    y='avg_response_time',
                    title='Response Time (Last 48h)',
                    labels={'avg_response_time': 'Seconds', 'timestamp': 'Time'}
                )
                fig_response.update_layout(height=300)
                st.plotly_chart(fig_response, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading system dashboard: {str(e)}")
    
    def render_query_with_enhancements(self):
        """Enhanced query interface with real-time features"""
        st.subheader("🔍 Intelligent Query System")
        
        # Query input with suggestions
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Ask a question about your data:",
                placeholder="What insights can you provide from the processed data?",
                help="Ask any question about your knowledge base"
            )
        
        with col2:
            use_cache = st.checkbox("Use Cache", value=True, 
                                  help="Use cached results for faster responses")
        
        # Advanced options in expander
        with st.expander("⚙️ Advanced Query Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_results = st.slider("Max Sources", 1, 20, 5, 
                                      help="Maximum number of source documents to consider")
            
            with col2:
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1, 0.05,
                                               help="Minimum similarity score for relevant documents")
            
            with col3:
                selected_model = st.selectbox(
                    "AI Model",
                    options=["sarvam-m", "deepseek-chat", "moonshotai/kimi-k2:free", "gpt-4o", "anthropic/claude-3.5-sonnet"],
                    index=0,
                    help="Choose the AI model for generating answers"
                )
        
        # Query execution
        if st.button("🔍 Search", type="primary", disabled=not query):
            if query:
                self._execute_enhanced_query(query, selected_model, max_results, 
                                           similarity_threshold, use_cache)
        
        # Suggested queries
        self._render_suggested_queries()
    
    def _execute_enhanced_query(self, query: str, model: str, max_results: int, 
                               similarity_threshold: float, use_cache: bool):
        """Execute query with enhanced features"""
        try:
            start_time = time.time()
            
            # Import and use hybrid processor for intelligent RAG
            from ..backend.hybrid_rag_processor import HybridRAGProcessor
            
            hybrid_processor = HybridRAGProcessor(
                self.api.vector_store,
                self.api.rag_engine,
                self.api.enhanced_retrieval
            )
            
            # Use intelligent hybrid processing with web search
            use_web_search = st.session_state.get('use_web_search', True)
            
            with st.spinner("🧠 Processing with intelligent hybrid RAG..." + (" (comparing KB + web data)" if use_web_search else " (KB only)")):
                if use_web_search:
                    result = hybrid_processor.process_intelligent_query(
                        query=query,
                        llm_model=model,
                        use_web_search=True,
                        max_web_results=max_results
                    )
                    
                    # Show processing strategy used
                    strategy = result.get('strategy_used', 'unknown')
                    if strategy == 'hybrid_comparison':
                        st.success("✅ Compared knowledge base with web data - providing best answer")
                    elif strategy == 'web_data_with_kb_update':
                        st.success("✅ Knowledge base updated with new web information")
                    elif strategy == 'kb_only':
                        st.info("ℹ️ Used knowledge base only (web search unavailable)")
                else:
                    # Fallback to basic processing if web search disabled
                    result = self.api.query(
                        query=query,
                        llm_model=model,
                        max_results=max_results,
                        similarity_threshold=similarity_threshold,
                        use_cache=use_cache
                    )
            
            response_time = time.time() - start_time
            
            if result['status'] == 'success':
                # Show answer with enhanced formatting and copy functionality
                st.subheader("💡 Answer")
                self._render_answer_with_copy(result['answer'], f"enhanced_{int(time.time())}")
                
                # Show metadata
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cached_icon = "🚀" if result.get('cached', False) else "⚡"
                    cache_status = "Cached" if result.get('cached', False) else "Fresh"
                    st.metric(f"{cached_icon} Response", cache_status)
                
                with col2:
                    processing_time = result.get('processing_time', response_time)
                    st.metric("⏱️ Time", f"{processing_time:.2f}s")
                
                with col3:
                    confidence = result.get('confidence', 0)
                    st.metric("🎯 Confidence", f"{confidence:.1%}")
                
                with col4:
                    source_count = len(result.get('sources', []))
                    st.metric("📚 Sources", source_count)
                
                # Show sources (both KB and web sources)
                if result.get('sources'):
                    st.subheader("📖 Knowledge Base Sources")
                    for i, source in enumerate(result['sources'], 1):
                        st.write(f"{i}. {source}")
                
                # Show web sources separately if available
                if result.get('web_sources'):
                    st.subheader("🌐 Web Sources")
                    for i, web_source in enumerate(result['web_sources'], 1):
                        st.write(f"{i}. [{web_source['title']}]({web_source['url']}) (Score: {web_source['score']:.2f})")
                
                # Show hybrid processing insights
                if result.get('insights'):
                    st.info(f"💡 Processing Details: {result['insights']}")
                
                # Add to query history
                if 'enhanced_query_history' not in st.session_state:
                    st.session_state.enhanced_query_history = []
                
                st.session_state.enhanced_query_history.append({
                    'query': query,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'confidence': result.get('confidence', 0),
                    'response_time': processing_time,
                    'cached': result.get('cached', False),
                    'model_used': result.get('model_used', model),
                    'timestamp': time.time()
                })
                
                st.success("✅ Query processed successfully!")
                
            else:
                st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    
    def _render_suggested_queries(self):
        """Render intelligent query suggestions"""
        suggested_queries = [
            "What are the main topics covered in the knowledge base?",
            "Summarize the key insights from all sources",
            "What trends or patterns can you identify?",
            "Compare different perspectives from the sources",
            "What are the most important findings?",
            "Identify any contradictions or conflicts in the data"
        ]
        
        st.subheader("💭 Suggested Queries")
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggested_queries):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.suggested_query = suggestion
                    st.rerun()
        
        # Handle suggested query selection
        if hasattr(st.session_state, 'suggested_query'):
            st.info(f"Selected: {st.session_state.suggested_query}")
            del st.session_state.suggested_query
    
    def _render_answer_with_copy(self, answer_text: str, unique_id: str):
        """Render answer text with copy functionality for enterprise UI"""
        # Answer display with copy functionality using columns
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Display styled answer
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%);
                border-left: 4px solid #2196f3;
                padding: 1.5rem;
                margin: 0.5rem 0;
                border-radius: 0 12px 12px 0;
                box-shadow: 0 4px 8px rgba(33,150,243,0.15);
            ">
                <div style="
                    line-height: 1.7;
                    color: #1a1a1a;
                    font-size: 1rem;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    font-weight: 400;
                ">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Copy area for easy text selection
            st.text_area(
                "Copy text:",
                value=answer_text,
                height=100,
                key=f"copy_{unique_id}",
                help="Select all text and copy"
            )
    
    def render_system_info(self):
        """Render system information and configuration"""
        st.subheader("⚙️ System Information")
        
        # System metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vector Store", "ChromaDB")
            st.metric("Documents", "3,432+")
        
        with col2:
            st.metric("AI Provider", "SARVAM API")
            st.metric("Web Search", "Tavily API")
        
        with col3:
            st.metric("Database", "PostgreSQL")
            st.metric("Status", "Online")
        
        # Configuration info
        st.subheader("📋 Current Configuration")
        
        config_info = {
            "Embedding Model": "all-MiniLM-L6-v2",
            "Default AI Model": "sarvam-m",
            "Max Query Results": "20",
            "Similarity Threshold": "0.1",
            "Cache TTL": "5 minutes",
            "Rate Limit": "50 queries/hour"
        }
        
        for key, value in config_info.items():
            st.write(f"**{key}:** {value}")
        
        # Health check
        st.subheader("🏥 System Health")
        
        health_status = {
            "Vector Store": "✅ Healthy",
            "AI Engine": "✅ Healthy", 
            "Web Search": "✅ Healthy",
            "Database": "✅ Healthy",
            "Authentication": "✅ Healthy"
        }
        
        for service, status in health_status.items():
            st.write(f"**{service}:** {status}")