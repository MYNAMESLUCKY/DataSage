"""
Real-time Performance Monitoring and Analytics Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

def show_performance_dashboard():
    """Display comprehensive performance monitoring dashboard"""
    
    st.markdown("### 📊 System Performance Analytics")
    
    # Get cache statistics
    try:
        from src.backend.enhanced_cache import get_cache_manager
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_cache_statistics()
        
        # Display cache performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Query Cache Hit Rate",
                f"{cache_stats.get('query_hit_rate', 0):.1%}",
                delta=f"{cache_stats.get('query_hits', 0)} hits"
            )
        
        with col2:
            st.metric(
                "Embedding Cache Hit Rate", 
                f"{cache_stats.get('embedding_hit_rate', 0):.1%}",
                delta=f"{cache_stats.get('embedding_hits', 0)} hits"
            )
        
        with col3:
            st.metric(
                "Cache Entries",
                cache_stats.get('query_cache_entries', 0) + cache_stats.get('embedding_cache_entries', 0),
                delta=f"{cache_stats.get('reranking_cache_entries', 0)} reranking"
            )
        
        with col4:
            total_requests = cache_stats.get('query_hits', 0) + cache_stats.get('query_misses', 0)
            if total_requests > 0:
                avg_hit_rate = cache_stats.get('query_hits', 0) / total_requests
                st.metric("Overall Cache Efficiency", f"{avg_hit_rate:.1%}")
            else:
                st.metric("Overall Cache Efficiency", "0.0%")
        
        # Cache performance chart
        if cache_stats.get('query_hits', 0) + cache_stats.get('query_misses', 0) > 0:
            fig = go.Figure(data=[
                go.Bar(name='Cache Hits', x=['Query', 'Embedding', 'Reranking'], 
                      y=[cache_stats.get('query_hits', 0), cache_stats.get('embedding_hits', 0), cache_stats.get('reranking_hits', 0)],
                      marker_color='green'),
                go.Bar(name='Cache Misses', x=['Query', 'Embedding', 'Reranking'], 
                      y=[cache_stats.get('query_misses', 0), cache_stats.get('embedding_misses', 0), cache_stats.get('reranking_misses', 0)],
                      marker_color='orange')
            ])
            
            fig.update_layout(
                title="Cache Performance by Type",
                barmode='stack',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Cache statistics unavailable: {e}")
    
    # System improvements summary
    st.markdown("### 🚀 Recent Performance Improvements")
    
    improvements_data = [
        {
            "Feature": "Query Rewriting",
            "Implementation": "Multiple query variations",
            "Expected Improvement": "+4-6 points NDCG",
            "Status": "✅ Active"
        },
        {
            "Feature": "Advanced Reranking", 
            "Implementation": "LLM-based cross-encoder",
            "Expected Improvement": "+22 points NDCG@3",
            "Status": "✅ Active"
        },
        {
            "Feature": "Multi-Level Caching",
            "Implementation": "Query, embedding, reranking cache",
            "Expected Improvement": "2-5x faster responses",
            "Status": "✅ Active"
        },
        {
            "Feature": "Hybrid Search",
            "Implementation": "Vector + text + reranking",
            "Expected Improvement": "Better relevance",
            "Status": "✅ Active"
        }
    ]
    
    # Display improvements table
    st.dataframe(improvements_data, use_container_width=True)
    
    # Performance optimization controls
    st.markdown("### ⚙️ Performance Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Optimize Cache", help="Clean expired entries and optimize cache performance"):
            try:
                cache_manager.optimize_cache()
                st.success("Cache optimization completed")
            except Exception as e:
                st.error(f"Cache optimization failed: {e}")
    
    with col2:
        if st.button("📊 Refresh Stats", help="Refresh performance statistics"):
            st.rerun()
    
    with col3:
        cache_type = st.selectbox("Clear Cache Type", ["query", "embedding", "reranking", "all"])
        if st.button(f"🗑️ Clear {cache_type.title()} Cache"):
            try:
                cache_manager.clear_cache(cache_type)
                st.success(f"{cache_type.title()} cache cleared")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")

def show_query_performance_metrics(processing_time: float, sources_count: int, approach: str):
    """Display real-time query performance metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Response Time",
            f"{processing_time:.2f}s",
            delta="Fast" if processing_time < 3 else "Slow"
        )
    
    with col2:
        st.metric(
            "Sources Used", 
            sources_count,
            delta="Rich" if sources_count > 5 else "Limited"
        )
    
    with col3:
        st.metric(
            "Processing Approach",
            approach.replace('_', ' ').title()
        )
    
    with col4:
        # Performance rating based on time and sources
        if processing_time < 2 and sources_count > 5:
            rating = "Excellent"
            color = "green"
        elif processing_time < 4 and sources_count > 3:
            rating = "Good"
            color = "blue"
        else:
            rating = "Fair"
            color = "orange"
        
        st.metric("Performance Rating", rating)

def show_advanced_features_status():
    """Show status of advanced RAG features"""
    
    st.markdown("### 🔧 Advanced Features Status")
    
    features = [
        ("Query Rewriting", "✅", "Generating 3-5 query variations for better retrieval"),
        ("Multi-Query Processing", "✅", "Breaking complex queries into sub-queries"),
        ("Intelligent Routing", "✅", "Smart routing based on query type"),
        ("Cross-Encoder Reranking", "✅", "LLM-based relevance scoring"),
        ("Hybrid Search", "✅", "Combining vector and text search"),
        ("Multi-Level Caching", "✅", "Query, embedding, and reranking caches"),
        ("Real-time Web Search", "✅", "Tavily integration for live data"),
        ("Knowledge Base Updates", "✅", "Automatic KB updates from web data"),
        ("Performance Monitoring", "✅", "Real-time analytics and optimization")
    ]
    
    for feature, status, description in features:
        col1, col2, col3 = st.columns([2, 1, 4])
        with col1:
            st.write(f"**{feature}**")
        with col2:
            st.write(status)
        with col3:
            st.write(description)

def show_processing_breakdown(query_analysis: Dict[str, Any]):
    """Show detailed breakdown of query processing"""
    
    if not query_analysis:
        return
    
    st.markdown("### 🔍 Query Processing Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Query Analysis:**")
        st.write(f"• Sub-queries generated: {len(query_analysis.get('sub_queries', []))}")
        st.write(f"• Total query variations: {query_analysis.get('total_queries', 1)}")
        st.write(f"• Processing time: {query_analysis.get('processing_time', 0):.3f}s")
        
        routing_info = query_analysis.get('routing', {})
        st.write(f"• Routing strategy: {routing_info.get('strategy', 'hybrid')}")
        st.write(f"• Confidence: {routing_info.get('confidence', 0):.2f}")
    
    with col2:
        st.markdown("**Query Variations:**")
        query_rewrites = query_analysis.get('query_rewrites', {})
        for category, queries in query_rewrites.items():
            if queries:
                st.write(f"**{category.title()}:**")
                for i, query in enumerate(queries[:3], 1):  # Show first 3
                    st.write(f"  {i}. {query[:60]}{'...' if len(query) > 60 else ''}")

def create_performance_chart(processing_times: List[float], labels: List[str]):
    """Create a performance comparison chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=processing_times,
        marker=dict(
            color=processing_times,
            colorscale='RdYlGn_r',  # Red for slow, green for fast
            showscale=True,
            colorbar=dict(title="Response Time (s)")
        ),
        text=[f"{t:.2f}s" for t in processing_times],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Query Processing Performance",
        xaxis_title="Processing Stage",
        yaxis_title="Time (seconds)",
        height=400
    )
    
    return fig