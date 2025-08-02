import streamlit as st
import time
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class UIComponents:
    """
    Reusable UI components for the RAG system
    """
    
    @staticmethod
    def render_status_badge(status: str, text: Optional[str] = None) -> str:
        """Render a status badge with appropriate styling"""
        if text is None:
            text = status.title()
            
        status_colors = {
            'success': '#28a745',
            'warning': '#ffc107', 
            'error': '#dc3545',
            'info': '#17a2b8',
            'pending': '#6c757d'
        }
        
        color = status_colors.get(status.lower(), '#6c757d')
        
        return f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-weight: bold;
            text-transform: uppercase;
        ">{text}</span>
        """
    
    @staticmethod
    def render_metric_card(title: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
        """Render a metric card with optional delta and help text"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
    
    @staticmethod
    def render_progress_bar(current: int, total: int, label: str = "Progress"):
        """Render a progress bar with current/total information"""
        if total == 0:
            progress = 0
        else:
            progress = current / total
        
        st.progress(progress)
        st.write(f"{label}: {current}/{total} ({progress:.1%})")
    
    @staticmethod
    def render_data_source_card(source: Dict[str, Any], index: int):
        """Render a data source card with status and actions"""
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                background: white;
            ">
                <h4 style="margin: 0 0 0.5rem 0;">{source.get('name', f'Source {index + 1}')}</h4>
                <p style="margin: 0 0 0.5rem 0; color: #666;">
                    <strong>URL:</strong> {source['url'][:60]}{'...' if len(source['url']) > 60 else ''}
                </p>
                <p style="margin: 0; color: #666;">
                    <strong>Type:</strong> {source.get('type', 'web')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_query_result_card(result: Dict[str, Any]):
        """Render a query result with answer and sources"""
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        ">
            <h4 style="color: #007bff; margin: 0 0 1rem 0;">Answer</h4>
            <p style="margin: 0 0 1rem 0; line-height: 1.6;">
                {result['answer']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if result.get('sources'):
            st.markdown("**Sources:**")
            for i, source in enumerate(result['sources'], 1):
                st.markdown(f"{i}. {source}")
        
        # Show metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Confidence: {result.get('confidence', 'N/A')}")
        with col2:
            st.caption(f"Response Time: {result.get('response_time', 'N/A')}s")
        with col3:
            st.caption(f"Model: {result.get('model_used', 'N/A')}")
    
    @staticmethod
    def render_system_health_dashboard(health_data: Dict[str, Any]):
        """Render system health dashboard"""
        st.subheader("System Health")
        
        # Overall status
        overall_status = "Healthy" if health_data.get('is_healthy', False) else "Issues Detected"
        status_color = "success" if health_data.get('is_healthy', False) else "error"
        
        st.markdown(
            f"**Overall Status:** {UIComponents.render_status_badge(status_color, overall_status)}", 
            unsafe_allow_html=True
        )
        
        # Component status
        components = health_data.get('components', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**API Components:**")
            for component, status in components.items():
                status_text = "✅ Online" if status else "❌ Offline"
                st.write(f"- {component.title()}: {status_text}")
        
        with col2:
            # Memory usage
            memory = health_data.get('memory_usage', {})
            if memory and 'rss_mb' in memory:
                st.markdown("**Memory Usage:**")
                st.write(f"- RSS: {memory['rss_mb']:.1f} MB")
                st.write(f"- Usage: {memory.get('percent', 0):.1f}%")
    
    @staticmethod
    def render_processing_timeline(processing_history: List[Dict[str, Any]]):
        """Render processing timeline visualization"""
        if not processing_history:
            st.info("No processing history available")
            return
        
        st.subheader("Processing Timeline")
        
        # Create timeline data
        timeline_data = []
        for event in processing_history:
            timeline_data.append({
                'timestamp': datetime.fromtimestamp(event.get('timestamp', time.time())),
                'event': event.get('event', 'Unknown'),
                'status': event.get('status', 'unknown'),
                'details': event.get('details', '')
            })
        
        # Display as a simple list for now (could be enhanced with plotly timeline)
        for event in reversed(timeline_data[-10:]):  # Show last 10 events
            status_color = {
                'success': '🟢',
                'warning': '🟡', 
                'error': '🔴',
                'info': '🔵'
            }.get(event['status'], '⚪')
            
            st.write(f"{status_color} **{event['timestamp'].strftime('%H:%M:%S')}** - {event['event']}")
            if event['details']:
                st.caption(event['details'])
    
    @staticmethod
    def render_statistics_dashboard(stats: Dict[str, Any]):
        """Render statistics dashboard with charts"""
        st.subheader("System Statistics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            UIComponents.render_metric_card(
                "Total Documents", 
                str(stats.get('total_documents', 0))
            )
        
        with col2:
            UIComponents.render_metric_card(
                "Unique Sources", 
                str(stats.get('unique_sources', 0))
            )
        
        with col3:
            UIComponents.render_metric_card(
                "Index Size", 
                f"{stats.get('index_size_mb', 0):.1f} MB"
            )
        
        with col4:
            UIComponents.render_metric_card(
                "Avg Doc Length", 
                f"{stats.get('avg_document_length', 0):.0f} chars"
            )
        
        # Document length distribution (if data available)
        if 'document_lengths' in stats:
            st.subheader("Document Length Distribution")
            
            fig = px.histogram(
                x=stats['document_lengths'],
                title="Distribution of Document Lengths",
                labels={'x': 'Document Length (characters)', 'y': 'Frequency'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_query_suggestions(suggestions: List[str]):
        """Render query suggestions as clickable buttons"""
        st.subheader("Suggested Questions")
        
        cols = st.columns(2)
        
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    return suggestion
        
        return None
    
    @staticmethod
    def render_loading_spinner(message: str = "Processing..."):
        """Render a loading spinner with message"""
        with st.spinner(message):
            # Simulate some processing time for visual effect
            time.sleep(0.1)
    
    @staticmethod
    def render_error_message(error: str, details: Optional[str] = None):
        """Render an error message with optional details"""
        st.error(f"❌ {error}")
        
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def render_success_message(message: str, details: Optional[str] = None):
        """Render a success message with optional details"""
        st.success(f"✅ {message}")
        
        if details:
            st.info(details)
    
    @staticmethod
    def render_info_panel(title: str, content: str, expanded: bool = False):
        """Render an expandable info panel"""
        with st.expander(title, expanded=expanded):
            st.markdown(content)
    
    @staticmethod
    def render_configuration_panel(config: Dict[str, Any]) -> Dict[str, Any]:
        """Render configuration panel and return updated config"""
        st.subheader("Configuration")
        
        updated_config = config.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            updated_config['chunk_size'] = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=config.get('chunk_size', 512),
                step=50,
                help="Size of text chunks for processing"
            )
            
            updated_config['similarity_threshold'] = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config.get('similarity_threshold', 0.7),
                step=0.05,
                help="Minimum similarity score for document retrieval"
            )
        
        with col2:
            updated_config['chunk_overlap'] = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=config.get('chunk_overlap', 50),
                step=10,
                help="Overlap between consecutive chunks"
            )
            
            updated_config['max_results'] = st.slider(
                "Max Results",
                min_value=1,
                max_value=20,
                value=config.get('max_results', 5),
                help="Maximum number of results to return"
            )
        
        return updated_config
