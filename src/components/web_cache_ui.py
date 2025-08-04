import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from ..backend.web_cache_db import WebCacheDatabase
from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class WebCacheUI:
    """
    UI components for web cache management and analytics
    """
    
    def __init__(self):
        self.web_cache = WebCacheDatabase()
    
    def render_cache_status(self):
        """Render cache connection status"""
        if self.web_cache.is_connected:
            st.success("✅ Database caching enabled")
        else:
            st.warning("⚠️ Database caching disabled")
    
    def render_cache_statistics(self):
        """Render cache usage statistics"""
        stats = self.web_cache.get_cache_statistics()
        
        if stats.get("status") == "disconnected":
            st.error("Database not connected")
            return
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Cached Searches",
                stats.get("total_cached_searches", 0),
                help="Total number of cached search queries"
            )
        
        with col2:
            st.metric(
                "Cached Content",
                stats.get("total_cached_content", 0),
                help="Total number of cached web content items"
            )
        
        with col3:
            st.metric(
                "Recent Searches",
                stats.get("recent_searches_24h", 0),
                help="Searches cached in the last 24 hours"
            )
        
        with col4:
            st.metric(
                "Cache Hits",
                stats.get("total_cache_hits", 0),
                help="Total number of times cached results were used"
            )
    
    def render_cache_management(self):
        """Render cache management controls"""
        st.subheader("🗄️ Cache Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Statistics"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clean Old Cache (7+ days)"):
                if self.web_cache.is_connected:
                    cleaned = self.web_cache.cleanup_old_cache(days_old=7)
                    st.success(f"Cleaned {cleaned} old cache entries")
                    st.rerun()
                else:
                    st.error("Database not connected")
        
        with col3:
            if st.button("🗑️ Clean Old Cache (3+ days)"):
                if self.web_cache.is_connected:
                    cleaned = self.web_cache.cleanup_old_cache(days_old=3)
                    st.success(f"Cleaned {cleaned} old cache entries")
                    st.rerun()
                else:
                    st.error("Database not connected")
    
    def render_cache_settings(self):
        """Render cache configuration settings"""
        st.subheader("⚙️ Cache Settings")
        
        # Cache age settings
        cache_age = st.slider(
            "Cache Age (hours)",
            min_value=1,
            max_value=72,
            value=24,
            help="How long to keep cached results before fetching fresh data"
        )
        st.session_state.cache_age_hours = cache_age
        
        # Quality threshold
        quality_threshold = st.slider(
            "Content Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Minimum quality score for cached content to be used"
        )
        st.session_state.quality_threshold = quality_threshold
        
        # Auto-cleanup settings
        auto_cleanup = st.checkbox(
            "Enable Auto-cleanup",
            value=True,
            help="Automatically clean old cache entries during queries"
        )
        st.session_state.auto_cleanup = auto_cleanup
        
        if auto_cleanup:
            cleanup_days = st.number_input(
                "Auto-cleanup after (days)",
                min_value=1,
                max_value=30,
                value=7,
                help="Automatically remove cache entries older than this many days"
            )
            st.session_state.cleanup_days = cleanup_days
    
    def render_recent_cached_queries(self):
        """Render list of recent cached queries"""
        st.subheader("🕒 Recent Cached Queries")
        
        if not self.web_cache.is_connected:
            st.warning("Database not connected - cannot show cached queries")
            return
        
        # This would require additional database query method
        # For now, show placeholder
        st.info("Recent query history will be displayed here")
        
        # Example of what it could look like:
        sample_queries = [
            {"query": "Current AI trends 2024", "cached_at": "2 hours ago", "hits": 3},
            {"query": "Python best practices", "cached_at": "5 hours ago", "hits": 1},
            {"query": "Machine learning algorithms", "cached_at": "1 day ago", "hits": 7}
        ]
        
        for i, query_info in enumerate(sample_queries):
            with st.expander(f"🔍 {query_info['query']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Cached:** {query_info['cached_at']}")
                with col2:
                    st.write(f"**Cache Hits:** {query_info['hits']}")
                with col3:
                    if st.button(f"Clear Cache", key=f"clear_{i}"):
                        st.success("Cache cleared for this query")
    
    def render_cache_performance_chart(self):
        """Render cache performance visualization"""
        st.subheader("📈 Cache Performance")
        
        if not self.web_cache.is_connected:
            st.warning("Database not connected - cannot show performance metrics")
            return
        
        # Sample data for demonstration
        # In reality, this would query the database for actual metrics
        sample_data = {
            "date": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
            "cache_hits": [15, 23, 18, 31, 27, 22, 29],
            "fresh_queries": [25, 18, 22, 14, 18, 23, 16]
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sample_data["date"],
            y=sample_data["cache_hits"],
            mode='lines+markers',
            name='Cache Hits',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=sample_data["date"],
            y=sample_data["fresh_queries"],
            mode='lines+markers',
            name='Fresh Queries',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Cache Hit Rate vs Fresh Queries",
            xaxis_title="Time Period",
            yaxis_title="Number of Queries",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cache efficiency metrics
        cache_hits = sum(sample_data["cache_hits"])
        fresh_queries = sum(sample_data["fresh_queries"])
        total_queries = cache_hits + fresh_queries
        cache_efficiency = (cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        st.metric(
            "Cache Efficiency",
            f"{cache_efficiency:.1f}%",
            help=f"Percentage of queries served from cache ({cache_hits}/{total_queries})"
        )
    
    def render_full_dashboard(self):
        """Render complete web cache dashboard"""
        st.header("🌐 Web Search Cache Dashboard")
        
        # Status and basic stats
        self.render_cache_status()
        st.divider()
        
        # Statistics
        self.render_cache_statistics()
        st.divider()
        
        # Performance chart
        self.render_cache_performance_chart()
        st.divider()
        
        # Management controls
        self.render_cache_management()
        st.divider()
        
        # Settings
        self.render_cache_settings()
        st.divider()
        
        # Recent queries
        self.render_recent_cached_queries()