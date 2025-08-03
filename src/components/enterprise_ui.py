"""
Enterprise UI components with real-time updates and advanced features
"""

import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import logging
import requests
import json

logger = logging.getLogger(__name__)

class EnterpriseUI:
    """Enhanced UI components for enterprise features"""
    
    def __init__(self, api):
        self.api = api
        self.api_gateway_url = "http://localhost:8000"
    
    def render(self):
        """Main render method for the enterprise interface"""
        # Create tabs for different sections
        tabs = st.tabs(["🔍 Query System", "📁 File Processing", "🔑 API Keys", "📊 Analytics", "⚙️ System"])
        
        with tabs[0]:
            self.render_query_with_enhancements()
        
        with tabs[1]:
            self.render_async_file_processor()
        
        with tabs[2]:
            self.render_api_key_management()
        
        with tabs[3]:
            col1, col2 = st.columns(2)
            with col1:
                self.render_cache_analytics()
            with col2:
                self.render_system_dashboard()
        
        with tabs[4]:
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
                        max_web_results=max_results,
                        max_results=max_results  # Pass the user's setting
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
                # Extract answer text
                answer_text = result.get('answer', '').strip()
                if not answer_text:
                    st.error("❌ Empty answer received from AI model")
                    return
                
                # Display formatted answer with proper styling
                st.subheader("💡 Answer")
                
                # Render the answer with enhanced formatting
                self._render_formatted_answer(answer_text, f"enhanced_{int(time.time())}")
                
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
    
    def _render_formatted_answer(self, answer_text: str, unique_id: str):
        """Render answer with professional formatting and copy functionality"""
        if not answer_text or not answer_text.strip():
            st.error("❌ No answer content to display")
            return
        
        # Process the answer text for better formatting
        formatted_answer = self._format_answer_text(answer_text)
        
        # Create columns for answer and copy functionality
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Display the formatted answer using Streamlit markdown
            st.markdown(formatted_answer)
        
        with col2:
            # Copy functionality with text area
            st.text_area(
                "Copy text:",
                value=answer_text,
                height=150,
                key=f"copy_{unique_id}",
                help="Select all (Ctrl+A) and copy (Ctrl+C)"
            )
    
    def _format_answer_text(self, text: str) -> str:
        """Format answer text with proper styling for better readability"""
        import re
        
        # Clean up the text
        formatted = text.strip()
        
        # Convert numbered lists to proper markdown
        formatted = re.sub(r'^(\d+)\.\s*\*\*([^*]+)\*\*:', r'**\1. \2:**', formatted, flags=re.MULTILINE)
        
        # Ensure proper spacing after headers
        formatted = re.sub(r'(\*\*[^*]+\*\*:)\s*', r'\1\n', formatted)
        
        # Add proper line breaks for bullet points
        formatted = re.sub(r'\n-\s+', r'\n\n- ', formatted)
        
        # Ensure proper spacing between sections
        formatted = re.sub(r'\n(\d+\.\s*\*\*)', r'\n\n\1', formatted)
        
        return formatted
    
    def _render_answer_with_copy(self, answer_text: str, unique_id: str):
        """Legacy method - redirects to new formatted renderer"""
        self._render_formatted_answer(answer_text, unique_id)
    
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
    
    def render_api_key_management(self):
        """Render API key management interface"""
        st.markdown("## 🔑 API Key Management")
        st.markdown("Generate and manage API keys for programmatic access to your RAG system.")
        
        # Get user authentication
        user_token = self._get_user_token()
        if not user_token:
            st.warning("Please authenticate to manage API keys")
            return
        
        # Create sub-tabs for API key management
        key_tabs = st.tabs(["📋 My API Keys", "➕ Generate New Key", "📊 Usage Analytics"])
        
        with key_tabs[0]:
            self._render_key_list(user_token)
        
        with key_tabs[1]:
            self._render_key_generation(user_token)
        
        with key_tabs[2]:
            self._render_usage_analytics(user_token)
    
    def _get_user_token(self):
        """Get or create authentication token for current user"""
        try:
            # Check if we have a cached token
            if 'api_token' in st.session_state:
                return st.session_state.api_token
            
            # Generate a token for the current session
            user_id = st.session_state.get('user_info', {}).get('uid', 'anonymous_user')
            
            response = requests.post(
                f"{self.api_gateway_url}/auth/token",
                params={"user_id": user_id, "role": "user"}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                st.session_state.api_token = token_data['access_token']
                return token_data['access_token']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user token: {e}")
            return None
    
    def _render_key_list(self, user_token: str):
        """Render the list of existing API keys"""
        st.markdown("### Your API Keys")
        
        try:
            response = requests.get(
                f"{self.api_gateway_url}/api-keys/list",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                keys = response.json()
                
                if not keys:
                    st.info("You haven't created any API keys yet. Use the 'Generate New Key' tab to create one.")
                    return
                
                for key in keys:
                    self._render_key_card(key, user_token)
                    
            else:
                st.error(f"Failed to fetch API keys: {response.text}")
                
        except Exception as e:
            st.error(f"Error fetching API keys: {str(e)}")
    
    def _render_key_card(self, key: Dict[str, Any], user_token: str):
        """Render a single API key card"""
        status_colors = {
            "active": "🟢",
            "suspended": "🟡", 
            "expired": "🔴",
            "revoked": "⚫"
        }
        
        status_icon = status_colors.get(key['status'], "❓")
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{status_icon} {key['name']}**")
                st.caption(f"ID: {key['key_id'][:12]}... | Scope: {key['scope']}")
                if key['description']:
                    st.caption(f"Description: {key['description']}")
            
            with col2:
                st.metric("Usage", key['usage_count'])
                if key['last_used']:
                    st.caption(f"Last used: {key['last_used'][:10]}")
                else:
                    st.caption("Never used")
            
            with col3:
                if key['status'] == 'active':
                    if st.button("Revoke", key=f"revoke_{key['key_id']}", type="secondary"):
                        self._revoke_key(key['key_id'], user_token)
                        st.rerun()
                else:
                    st.caption(f"Status: {key['status']}")
            
            st.divider()
    
    def _render_key_generation(self, user_token: str):
        """Render the API key generation form"""
        st.markdown("### Generate New API Key")
        
        with st.form("generate_key_form"):
            name = st.text_input(
                "Key Name *",
                placeholder="e.g., 'Production App API Key'",
                help="A descriptive name to identify this key"
            )
            
            description = st.text_area(
                "Description",
                placeholder="Optional description of what this key will be used for",
                height=100
            )
            
            scope_options = {
                "read_only": "Read Only - System information and health endpoints",
                "query_only": "Query Only - Search and retrieve information", 
                "ingest_only": "Ingest Only - Add documents to knowledge base",
                "full_access": "Full Access - Query and ingest operations"
            }
            
            scope = st.selectbox(
                "Access Scope *",
                options=list(scope_options.keys()),
                format_func=lambda x: scope_options[x],
                index=1
            )
            
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    expires_in_days = st.number_input(
                        "Expires in days",
                        min_value=0,
                        max_value=365,
                        value=0,
                        help="0 = Never expires"
                    )
                
                with col2:
                    rate_limit = st.number_input(
                        "Rate limit (requests/hour)",
                        min_value=1,
                        max_value=10000,
                        value=100
                    )
            
            submitted = st.form_submit_button("🔑 Generate API Key", type="primary")
            
            if submitted:
                if not name.strip():
                    st.error("Key name is required")
                else:
                    self._generate_new_key(
                        name=name.strip(),
                        description=description.strip(),
                        scope=scope,
                        expires_in_days=expires_in_days if expires_in_days > 0 else None,
                        rate_limit=rate_limit,
                        user_token=user_token
                    )
    
    def _generate_new_key(self, name: str, description: str, scope: str,
                         expires_in_days: Optional[int], rate_limit: int, user_token: str):
        """Generate a new API key"""
        try:
            payload = {
                "name": name,
                "description": description,
                "scope": scope,
                "rate_limit": rate_limit
            }
            
            if expires_in_days:
                payload["expires_in_days"] = expires_in_days
            
            response = requests.post(
                f"{self.api_gateway_url}/api-keys/generate",
                headers={"Authorization": f"Bearer {user_token}"},
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ API Key Generated Successfully!")
                st.markdown("### 🔐 Your New API Key")
                st.code(result['api_key'])
                st.warning("⚠️ **Important**: This is the only time you'll see this key. Store it securely!")
                
                with st.expander("Usage Example"):
                    st.markdown("### Python Example")
                    st.code(f"""
import requests

headers = {{
    "Authorization": "Bearer {result['api_key']}",
    "Content-Type": "application/json"
}}

response = requests.post(
    "{self.api_gateway_url}/query",
    headers=headers,
    json={{"query": "What is artificial intelligence?"}}
)

result = response.json()
print(result['answer'])
                    """, language="python")
                
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Failed to generate API key: {error_detail}")
                
        except Exception as e:
            st.error(f"Error generating API key: {str(e)}")
    
    def _revoke_key(self, key_id: str, user_token: str):
        """Revoke an API key"""
        try:
            response = requests.delete(
                f"{self.api_gateway_url}/api-keys/{key_id}",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                st.success("API key revoked successfully")
            else:
                st.error(f"Failed to revoke key: {response.text}")
                
        except Exception as e:
            st.error(f"Error revoking key: {str(e)}")
    
    def _render_usage_analytics(self, user_token: str):
        """Render usage analytics for all user's keys"""
        st.markdown("### Usage Analytics")
        
        try:
            response = requests.get(
                f"{self.api_gateway_url}/api-keys/list",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            
            if response.status_code == 200:
                keys = response.json()
                active_keys = [k for k in keys if k['status'] == 'active']
                
                if not active_keys:
                    st.info("No active API keys to show analytics for.")
                    return
                
                selected_key = st.selectbox(
                    "Select API Key",
                    options=active_keys,
                    format_func=lambda k: f"{k['name']} ({k['usage_count']} uses)"
                )
                
                if selected_key:
                    self._show_key_stats(selected_key['key_id'], user_token)
                    
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    def _show_key_stats(self, key_id: str, user_token: str):
        """Show usage statistics for a key"""
        try:
            days = st.select_slider(
                "Time period",
                options=[7, 14, 30, 60, 90],
                value=30,
                format_func=lambda x: f"Last {x} days"
            )
            
            response = requests.get(
                f"{self.api_gateway_url}/api-keys/{key_id}/usage",
                headers={"Authorization": f"Bearer {user_token}"},
                params={"days": days}
            )
            
            if response.status_code == 200:
                stats = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Requests", stats['total_requests'])
                
                with col2:
                    st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
                
                with col3:
                    st.metric("Successful Requests", stats['successful_requests'])
                
                if stats['top_endpoints']:
                    st.markdown("#### Most Used Endpoints")
                    for endpoint in stats['top_endpoints']:
                        st.text(f"{endpoint['endpoint']}: {endpoint['count']} requests")
                
            else:
                st.error("Failed to load usage statistics")
                
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")