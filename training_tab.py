"""
Training Tab for RAG System
===========================

Standalone training dashboard component.
"""

import streamlit as st
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from backend.api import RAGSystemAPI

def render_training_tab():
    """Render the complete training dashboard"""
    st.header("🎯 RAG Training & Improvement Dashboard")
    
    # Initialize API
    if 'api' not in st.session_state:
        st.session_state.api = RAGSystemAPI()
    
    api = st.session_state.api
    
    # 8-Point Improvement Status
    st.subheader("✅ 8-Point RAG Improvement Implementation Status")
    
    improvements = [
        {
            "point": "1. Document Chunking Strategy",
            "status": "✅ Implemented",
            "description": "Intelligent chunking with semantic awareness and content type detection"
        },
        {
            "point": "2. Aggressive Source Filtering", 
            "status": "✅ Implemented",
            "description": "Quality-based filtering removes low-value sources before retrieval"
        },
        {
            "point": "3. Better Embeddings",
            "status": "✅ Implemented", 
            "description": "Advanced embedding manager with caching and query optimization"
        },
        {
            "point": "4. Enhanced Retrieval Logic",
            "status": "✅ Implemented",
            "description": "Multi-strategy search with query expansion and hybrid approaches"
        },
        {
            "point": "5. Result Reranking",
            "status": "✅ Implemented",
            "description": "Intelligent document ranking based on relevance and content quality"
        },
        {
            "point": "6. Metadata Filtering",
            "status": "✅ Implemented",
            "description": "Content type and source authority-based filtering"
        },
        {
            "point": "7. Retrieval Pipeline Audit",
            "status": "✅ Implemented",
            "description": "Comprehensive monitoring and performance analytics"
        },
        {
            "point": "8. Search API Fallback",
            "status": "✅ Implemented",
            "description": "Wikipedia and DuckDuckGo fallback for missing knowledge"
        }
    ]
    
    for improvement in improvements:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{improvement['point']}**")
                st.caption(improvement['description'])
            
            with col2:
                st.success(improvement['status'])
    
    st.success("🎉 All 8 improvements successfully implemented! Your RAG system now has enterprise-grade quality and performance.")
    
    st.divider()
    
    # Performance Metrics
    st.subheader("📊 Performance Overview")
    
    try:
        insights_result = api.get_training_insights()
        
        if insights_result['status'] == 'success':
            insights = insights_result['insights']
            metrics = insights.get('performance_metrics', {})
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", metrics.get('total_queries', 0))
                
                with col2:
                    avg_confidence = metrics.get('avg_confidence', 0)
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                with col3:
                    response_time = metrics.get('avg_response_time', 0)
                    st.metric("Avg Response Time", f"{response_time:.1f}s")
                
                with col4:
                    fallback_rate = metrics.get('fallback_usage_rate', 0)
                    st.metric("Fallback Usage", f"{fallback_rate:.1%}")
                    
                # Recommendations
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    st.subheader("🎯 Improvement Recommendations")
                    
                    high_priority = [r for r in recommendations if r.get('priority') == 'high']
                    if high_priority:
                        st.error("🚨 High Priority Issues")
                        for rec in high_priority:
                            st.write(f"• **{rec.get('issue', 'Unknown issue')}**")
                            st.write(f"  *Suggestion: {rec.get('suggestion', 'No suggestion')}*")
                    else:
                        st.success("✅ No critical issues detected!")
            else:
                st.info("No performance data available yet. Start asking questions to generate insights!")
        else:
            st.error(f"Error loading training insights: {insights_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Training system error: {str(e)}")
    
    st.divider()
    
    # User Feedback Section
    st.subheader("💬 Provide Feedback")
    
    with st.expander("📝 Help Improve the System"):
        st.write("Your feedback helps train the system to provide better answers:")
        
        query_input = st.text_input("Query you want to provide feedback for:")
        
        col1, col2 = st.columns(2)
        with col1:
            satisfied = st.radio("Were you satisfied with the answer?", ["Yes", "No"])
        
        with col2:
            suggestions = st.text_area("Suggestions for improvement (optional):", height=100)
        
        if st.button("Submit Feedback", type="primary"):
            if query_input.strip():
                try:
                    user_satisfied = satisfied == "Yes"
                    feedback_result = api.add_user_feedback(
                        query_input, 
                        user_satisfied, 
                        suggestions.strip() or None
                    )
                    
                    if feedback_result['status'] == 'success':
                        st.success("Thank you! Your feedback has been recorded.")
                    else:
                        st.error(f"Error recording feedback: {feedback_result['error']}")
                except Exception as e:
                    st.error(f"Feedback submission error: {str(e)}")
            else:
                st.warning("Please enter the query you want to provide feedback for.")
    
    # Training Features Summary
    st.divider()
    st.subheader("🔧 Enhanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🧠 Intelligent Processing:**
        - Smart document chunking
        - Advanced embedding models
        - Multi-strategy retrieval
        - Quality-based filtering
        """)
    
    with col2:
        st.markdown("""
        **📈 Continuous Improvement:**
        - Performance monitoring
        - User feedback integration
        - Automatic optimization
        - Search fallback system
        """)

if __name__ == "__main__":
    render_training_tab()