"""
Wikipedia Ingestion Panel for Streamlit UI
Provides interface for Wikipedia data ingestion
"""

import streamlit as st
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class WikipediaPanel:
    """Panel for Wikipedia data ingestion controls"""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def render(self):
        """Render the Wikipedia ingestion panel"""
        st.header("📚 Wikipedia Data Ingestion")
        st.write("Ingest knowledge from Wikipedia to create a comprehensive knowledge base")
        
        # Warning about data volume
        with st.expander("⚠️ Important Information", expanded=True):
            st.warning("""
            **Wikipedia Ingestion Information:**
            - Wikipedia contains millions of articles - full ingestion is not practical
            - This system uses smart sampling to get diverse, high-quality content
            - Processing large amounts of data may take significant time
            - Each article is chunked into smaller pieces for better search results
            """)
        
        # Ingestion strategies
        st.subheader("Ingestion Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Choose Ingestion Strategy",
                options=[
                    "balanced",
                    "category_focused", 
                    "random_diverse"
                ],
                format_func=lambda x: {
                    "balanced": "🎯 Balanced (Categories + Random)",
                    "category_focused": "📖 Educational Focus", 
                    "random_diverse": "🎲 Random Diverse Sample"
                }[x],
                help="Different strategies for selecting Wikipedia content"
            )
        
        with col2:
            if strategy == "balanced":
                st.info("**Balanced Strategy**\n- 20 major categories (30 articles each)\n- 200 random articles for diversity\n- ~800 total articles")
            elif strategy == "category_focused":
                st.info("**Educational Focus**\n- Science, Math, History, Technology\n- ~600 articles from educational topics")
            elif strategy == "random_diverse":
                st.info("**Random Diverse**\n- 1000 completely random articles\n- Maximum topic diversity")
        
        # Custom category selection
        st.subheader("Custom Categories (Optional)")
        
        # Predefined categories
        popular_categories = [
            "Category:Science", "Category:Technology", "Category:History",
            "Category:Mathematics", "Category:Physics", "Category:Computer_science",
            "Category:Biology", "Category:Chemistry", "Category:Geography",
            "Category:Philosophy", "Category:Economics", "Category:Literature",
            "Category:Art", "Category:Music", "Category:Sports", "Category:Medicine"
        ]
        
        selected_categories = st.multiselect(
            "Select specific categories to include:",
            options=popular_categories,
            help="Choose specific Wikipedia categories to focus on"
        )
        
        articles_per_category = st.slider(
            "Articles per category:",
            min_value=10,
            max_value=100,
            value=25,
            step=5,
            help="Number of articles to fetch from each selected category"
        )
        
        # Random articles option
        st.subheader("Random Articles")
        random_count = st.slider(
            "Number of random articles:",
            min_value=0,
            max_value=500,
            value=100,
            step=25,
            help="Additional random articles for diversity"
        )
        
        # Processing options
        st.subheader("Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.selectbox(
                "Chunk Size",
                options=[500, 750, 1000, 1250],
                index=2,
                help="Size of text chunks for vector storage"
            )
        
        with col2:
            max_concurrent = st.selectbox(
                "Concurrent Processing",
                options=[1, 2, 3, 4],
                index=2,
                help="Number of articles to process simultaneously"
            )
        
        # Start ingestion button
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Wikipedia Ingestion", type="primary", use_container_width=True):
                self._start_ingestion(
                    strategy=strategy,
                    custom_categories=selected_categories,
                    articles_per_category=articles_per_category,
                    random_count=random_count,
                    chunk_size=chunk_size,
                    max_concurrent=max_concurrent
                )
        
        # Status display
        if hasattr(st.session_state, 'wikipedia_ingestion_status'):
            self._display_ingestion_status()
    
    def _start_ingestion(self, strategy: str, custom_categories: List[str], 
                        articles_per_category: int, random_count: int,
                        chunk_size: int, max_concurrent: int):
        """Start the Wikipedia ingestion process"""
        
        # Initialize status
        st.session_state.wikipedia_ingestion_status = {
            'status': 'starting',
            'progress': 0,
            'current_task': 'Initializing...',
            'results': None
        }
        
        # Create progress display
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            with progress_placeholder.container():
                st.info("🔄 Starting Wikipedia ingestion...")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Update status
            status_text.text("Connecting to Wikipedia API...")
            progress_bar.progress(10)
            
            if custom_categories:
                # Use custom categories
                status_text.text(f"Processing {len(custom_categories)} custom categories...")
                progress_bar.progress(30)
                
                result = self.api_client.ingest_wikipedia_categories(
                    categories=custom_categories,
                    articles_per_category=articles_per_category
                )
                
                progress_bar.progress(70)
                
                # Add random articles if specified
                if random_count > 0:
                    status_text.text(f"Adding {random_count} random articles...")
                    random_result = self.api_client.ingest_wikipedia_random(random_count)
                    
                    # Combine results
                    if result['status'] == 'success' and random_result['status'] == 'success':
                        combined_details = {
                            'successful': result['details']['successful'] + random_result['details']['successful'],
                            'failed': result['details']['failed'] + random_result['details']['failed'],
                            'documents_created': result['details']['documents_created'] + random_result['details']['documents_created']
                        }
                        result['details'] = combined_details
                        result['message'] = f"Processed {combined_details['successful']} total articles"
                
            else:
                # Use strategy-based ingestion
                status_text.text(f"Executing {strategy} strategy...")
                progress_bar.progress(30)
                
                result = self.api_client.ingest_wikipedia_comprehensive(strategy)
                progress_bar.progress(70)
            
            progress_bar.progress(100)
            status_text.text("Ingestion completed!")
            
            # Store results
            st.session_state.wikipedia_ingestion_status = {
                'status': 'completed',
                'progress': 100,
                'current_task': 'Completed',
                'results': result
            }
            
            # Display results
            if result['status'] == 'success':
                st.success(f"✅ {result['message']}")
                
                # Show detailed stats
                details = result.get('details', {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Articles Processed", details.get('successful', 0))
                with col2:
                    st.metric("Documents Created", details.get('documents_created', 0))
                with col3:
                    st.metric("Failed Articles", details.get('failed', 0))
                
                if details.get('failed', 0) > 0:
                    with st.expander("Failed Articles"):
                        failed_articles = details.get('failed_articles', [])
                        for article in failed_articles[:10]:  # Show first 10
                            st.text(f"• {article}")
                        if len(failed_articles) > 10:
                            st.text(f"... and {len(failed_articles) - 10} more")
            else:
                st.error(f"❌ Ingestion failed: {result.get('message', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Error during Wikipedia ingestion: {e}")
            st.error(f"❌ Error during ingestion: {str(e)}")
            
            st.session_state.wikipedia_ingestion_status = {
                'status': 'error',
                'progress': 0,
                'current_task': f'Error: {str(e)}',
                'results': None
            }
    
    def _display_ingestion_status(self):
        """Display current ingestion status"""
        status = st.session_state.wikipedia_ingestion_status
        
        if status['status'] == 'completed' and status['results']:
            st.success("✅ Wikipedia ingestion completed successfully!")
            
            # Reset button
            if st.button("🔄 Start New Ingestion"):
                del st.session_state.wikipedia_ingestion_status
                st.rerun()
        
        elif status['status'] == 'error':
            st.error(f"❌ Ingestion failed: {status['current_task']}")
            
            # Reset button
            if st.button("🔄 Try Again"):
                del st.session_state.wikipedia_ingestion_status
                st.rerun()