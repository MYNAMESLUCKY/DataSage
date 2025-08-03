"""
Training UI Components for RAG System
====================================

This module provides UI components for the training and feedback system.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
import time

def render_training_dashboard(api):
    """Render the comprehensive training dashboard"""
    st.header("🎯 RAG Training Dashboard")
    
    # Get training insights
    insights_result = api.get_training_insights()
    
    if insights_result['status'] == 'error':
        st.error(f"Error loading training insights: {insights_result['error']}")
        return
    
    insights = insights_result['insights']
    
    # Performance metrics overview
    _render_performance_overview(insights.get('performance_metrics', {}))
    
    # Recommendations section
    _render_recommendations(insights.get('recommendations', []))
    
    # Query analysis
    _render_query_analysis(insights)
    
    # User feedback section
    _render_feedback_section(api, insights)
    
    # Training data export
    _render_export_section(insights)

def _render_performance_overview(metrics: Dict[str, Any]):
    """Render performance metrics overview"""
    st.subheader("📊 Performance Metrics (Last 30 Days)")
    
    if not metrics:
        st.info("No performance data available yet. Start asking questions to generate insights!")
        return
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries",
            metrics.get('total_queries', 0),
            help="Number of queries processed"
        )
    
    with col2:
        avg_confidence = metrics.get('avg_confidence', 0)
        confidence_color = "normal" if avg_confidence > 0.7 else "inverse"
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.1%}",
            delta=None,
            help="Average confidence score of responses"
        )
    
    with col3:
        response_time = metrics.get('avg_response_time', 0)
        time_color = "normal" if response_time < 5 else "inverse"
        st.metric(
            "Avg Response Time",
            f"{response_time:.1f}s",
            help="Average time to generate responses"
        )
    
    with col4:
        fallback_rate = metrics.get('fallback_usage_rate', 0)
        fallback_color = "normal" if fallback_rate < 0.2 else "inverse"
        st.metric(
            "Fallback Usage",
            f"{fallback_rate:.1%}",
            help="Percentage of queries requiring external search"
        )
    
    # User satisfaction if available
    if metrics.get('user_satisfaction_rate') is not None:
        satisfaction = metrics['user_satisfaction_rate']
        st.metric(
            "User Satisfaction",
            f"{satisfaction:.1%}",
            help="Percentage of positive user feedback"
        )

def _render_recommendations(recommendations: list):
    """Render improvement recommendations"""
    st.subheader("🎯 Improvement Recommendations")
    
    if not recommendations:
        st.success("🎉 No critical issues detected! Your RAG system is performing well.")
        return
    
    # Group by priority
    high_priority = [r for r in recommendations if r.get('priority') == 'high']
    medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
    low_priority = [r for r in recommendations if r.get('priority') == 'low']
    
    # High priority recommendations
    if high_priority:
        st.error("🚨 High Priority Issues")
        for rec in high_priority:
            with st.expander(f"⚠️ {rec.get('issue', 'Unknown issue')}", expanded=True):
                st.write(f"**Type:** {rec.get('type', 'Unknown')}")
                st.write(f"**Suggestion:** {rec.get('suggestion', 'No suggestion')}")
                if 'affected_queries' in rec:
                    st.write(f"**Affected Queries:** {rec['affected_queries']}")
    
    # Medium priority recommendations
    if medium_priority:
        st.warning("⚡ Medium Priority Improvements")
        for rec in medium_priority:
            with st.expander(f"💡 {rec.get('issue', 'Unknown issue')}"):
                st.write(f"**Type:** {rec.get('type', 'Unknown')}")
                st.write(f"**Suggestion:** {rec.get('suggestion', 'No suggestion')}")
                if 'affected_queries' in rec:
                    st.write(f"**Affected Queries:** {rec['affected_queries']}")
    
    # Low priority recommendations
    if low_priority:
        st.info("📝 Low Priority Suggestions")
        for rec in low_priority:
            with st.expander(f"ℹ️ {rec.get('issue', 'Unknown issue')}"):
                st.write(f"**Type:** {rec.get('type', 'Unknown')}")
                st.write(f"**Suggestion:** {rec.get('suggestion', 'No suggestion')}")

def _render_query_analysis(insights: Dict[str, Any]):
    """Render query type analysis"""
    st.subheader("🔍 Query Analysis")
    
    # Query type distribution
    query_dist = insights.get('query_type_distribution', {})
    if query_dist:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Query Type Distribution**")
            # Create pie chart
            fig = px.pie(
                values=list(query_dist.values()),
                names=list(query_dist.keys()),
                title="Query Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Query Type Performance**")
            # Show which types need improvement
            recommendations = insights.get('recommendations', [])
            type_issues = [r for r in recommendations if r.get('type') == 'query_type_performance']
            
            if type_issues:
                for issue in type_issues:
                    st.warning(f"**{issue.get('query_type')}:** {issue.get('issue')}")
            else:
                st.success("All query types performing well!")
    
    # Low confidence examples
    low_conf_examples = insights.get('low_confidence_examples', [])
    if low_conf_examples:
        st.write("**Low Confidence Query Examples**")
        df = pd.DataFrame(low_conf_examples)
        st.dataframe(df, use_container_width=True)

def _render_feedback_section(api, insights: Dict[str, Any]):
    """Render user feedback interface"""
    st.subheader("💬 User Feedback")
    
    # Feedback stats
    feedback_count = insights.get('feedback_entries', 0)
    if feedback_count > 0:
        st.info(f"Total feedback entries: {feedback_count}")
    
    # Add feedback form
    with st.expander("📝 Add Feedback for Recent Query"):
        st.write("Help improve the system by providing feedback on recent answers:")
        
        query_input = st.text_input("Query you want to provide feedback for:")
        
        col1, col2 = st.columns(2)
        with col1:
            satisfied = st.radio(
                "Were you satisfied with the answer?",
                ["Yes", "No"],
                key="feedback_satisfaction"
            )
        
        with col2:
            suggestions = st.text_area(
                "Suggestions for improvement (optional):",
                height=100,
                key="feedback_suggestions"
            )
        
        if st.button("Submit Feedback", type="primary"):
            if query_input.strip():
                user_satisfied = satisfied == "Yes"
                feedback_result = api.add_user_feedback(
                    query_input, 
                    user_satisfied, 
                    suggestions.strip() or None
                )
                
                if feedback_result['status'] == 'success':
                    st.success("Thank you! Your feedback has been recorded.")
                    st.rerun()
                else:
                    st.error(f"Error recording feedback: {feedback_result['error']}")
            else:
                st.warning("Please enter the query you want to provide feedback for.")

def _render_export_section(insights: Dict[str, Any]):
    """Render training data export section"""
    st.subheader("📤 Export Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Options**")
        if st.button("Download Training Insights", type="secondary"):
            # Convert insights to JSON for download
            import json
            insights_json = json.dumps(insights, indent=2, default=str)
            
            st.download_button(
                label="📁 Download JSON",
                data=insights_json,
                file_name=f"rag_training_insights_{int(time.time())}.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**Training Summary**")
        st.metric("Queries Analyzed", insights.get('total_queries_analyzed', 0))
        st.metric("Feedback Entries", insights.get('feedback_entries', 0))
        
        generated_at = insights.get('generated_at', time.time())
        st.caption(f"Data generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(generated_at))}")

def render_8_point_improvement_status():
    """Render status of the 8-point improvement plan"""
    st.subheader("✅ 8-Point RAG Improvement Status")
    
    improvements = [
        {
            "point": "1. Document Chunking Strategy",
            "status": "✅ Implemented",
            "description": "Intelligent chunking with semantic awareness and content type detection",
            "color": "green"
        },
        {
            "point": "2. Aggressive Source Filtering", 
            "status": "✅ Implemented",
            "description": "Quality-based filtering removes low-value sources before retrieval",
            "color": "green"
        },
        {
            "point": "3. Better Embeddings",
            "status": "✅ Implemented", 
            "description": "Advanced embedding manager with caching and query optimization",
            "color": "green"
        },
        {
            "point": "4. Enhanced Retrieval Logic",
            "status": "✅ Implemented",
            "description": "Multi-strategy search with query expansion and hybrid approaches",
            "color": "green"
        },
        {
            "point": "5. Result Reranking",
            "status": "✅ Implemented",
            "description": "Intelligent document ranking based on relevance and content quality",
            "color": "green"
        },
        {
            "point": "6. Metadata Filtering",
            "status": "✅ Implemented",
            "description": "Content type and source authority-based filtering",
            "color": "green"
        },
        {
            "point": "7. Retrieval Pipeline Audit",
            "status": "✅ Implemented",
            "description": "Comprehensive monitoring and performance analytics",
            "color": "green"
        },
        {
            "point": "8. Search API Fallback",
            "status": "✅ Implemented",
            "description": "Wikipedia and DuckDuckGo fallback for missing knowledge",
            "color": "green"
        }
    ]
    
    for improvement in improvements:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{improvement['point']}**")
                st.caption(improvement['description'])
            
            with col2:
                if improvement['color'] == 'green':
                    st.success(improvement['status'])
                elif improvement['color'] == 'orange':
                    st.warning(improvement['status'])
                else:
                    st.info(improvement['status'])
    
    st.success("🎉 All 8 improvements successfully implemented! Your RAG system now has enterprise-grade quality and performance.")