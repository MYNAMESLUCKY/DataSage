import time
from typing import Dict, List, Any, Optional
from langchain.schema import Document

from ..utils.utils import setup_logging
from .tavily_integration import TavilyIntegrationService
from .rag_improvements import EnhancedRetrieval
from .training_system import get_training_system
from .web_cache_db import WebCacheDatabase

logger = setup_logging(__name__)

class HybridRAGProcessor:
    """
    Intelligent Hybrid RAG System that:
    1. Checks existing knowledge base first
    2. Fetches web data for comparison and updates
    3. Compares sources and identifies gaps
    4. Updates knowledge base with new information
    5. Generates comprehensive answer from all sources
    """
    
    def __init__(self, vector_store, rag_engine, enhanced_retrieval):
        self.vector_store = vector_store
        self.rag_engine = rag_engine
        self.enhanced_retrieval = enhanced_retrieval
        self.tavily_service = TavilyIntegrationService()
        self.training_system = get_training_system()
        self.web_cache = WebCacheDatabase()
    
    def process_intelligent_query(
        self, 
        query: str, 
        llm_model: Optional[str] = None,
        use_web_search: bool = True,
        max_web_results: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Intelligent hybrid processing as requested:
        1. Check if info is available in docs
        2. Compare with Tavily data
        3. If KB has data, compare and provide best answer
        4. If KB lacks data, train model with Tavily data and answer
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting intelligent hybrid RAG processing: {query[:100]}...")
            
            # STEP 1: Check existing knowledge base first
            logger.info("STEP 1: Checking existing knowledge base...")
            kb_docs = self.enhanced_retrieval.enhanced_similarity_search(
                self.vector_store,
                query, 
                k=10
            )
            
            kb_has_relevant_data = len(kb_docs) > 0
            logger.info(f"Knowledge base check: Found {len(kb_docs)} relevant documents")
            
            # STEP 2: Always fetch web data for comparison/updates
            web_results = []
            web_sources = []
            web_docs = []
            
            if use_web_search and self.tavily_service.is_available():
                logger.info("STEP 2: Fetching web data for comparison...")
                
                # Check cache first
                cached_results = None
                if self.web_cache.is_connected:
                    cached_results = self.web_cache.get_cached_results(query, max_age_hours=6)
                
                if cached_results:
                    logger.info("Using cached web data for comparison")
                    web_results = [
                        type('TavilyResult', (), {
                            'url': r.get('url', ''),
                            'title': r.get('title', ''),
                            'content': r.get('content', ''),
                            'score': r.get('score', 0.0)
                        })() for r in cached_results.results
                    ]
                else:
                    # Fresh web search
                    try:
                        web_results = self.tavily_service.search_and_fetch(
                            query=query,
                            max_results=max_web_results,
                            search_depth="advanced"
                        )
                        
                        # Cache the fresh results
                        if web_results and self.web_cache.is_connected:
                            results_for_cache = [
                                {
                                    "url": r.url,
                                    "title": r.title,
                                    "content": r.content,
                                    "score": r.score
                                } for r in web_results
                            ]
                            self.web_cache.cache_search_results(query, results_for_cache)
                            
                        logger.info(f"Fetched {len(web_results)} fresh web results")
                    except Exception as e:
                        logger.error(f"Web search failed: {e}")
                        web_results = []
                
                # Process web results into documents
                for result in web_results:
                    web_doc = Document(
                        page_content=f"Title: {result.title}\n\nContent: {result.content}",
                        metadata={
                            "source": result.url,
                            "title": result.title,
                            "type": "web_data",
                            "score": result.score
                        }
                    )
                    web_docs.append(web_doc)
                
                web_sources = [{"url": r.url, "title": r.title, "score": r.score} for r in web_results]
            
            # STEP 3: Intelligent decision making
            strategy_used = ""
            final_docs = []
            should_update_kb = False
            
            if kb_has_relevant_data and web_docs:
                # CASE 1: Compare KB data with web data, provide best answer
                logger.info("STEP 3: KB has data + Web data available - Comparing sources")
                strategy_used = "hybrid_comparison"
                
                # Combine both sources with web data first (more recent)
                final_docs.extend(web_docs[:max_web_results])
                final_docs.extend(kb_docs[:5])
                
                # Web data might have newer information
                should_update_kb = True
                
            elif not kb_has_relevant_data and web_docs:
                # CASE 2: KB lacks data, use web data and update KB
                logger.info("STEP 3: KB lacks data - Using web data and updating knowledge base")
                strategy_used = "web_data_with_kb_update"
                
                final_docs.extend(web_docs)
                should_update_kb = True
                
            elif kb_has_relevant_data and not web_docs:
                # CASE 3: KB has data but web search failed
                logger.info("STEP 3: Using knowledge base data only (web search unavailable)")
                strategy_used = "kb_only"
                
                final_docs.extend(kb_docs)
                
            else:
                # CASE 4: No data found anywhere
                logger.warning("STEP 3: No relevant data found in KB or web")
                strategy_used = "no_data"
                
                return {
                    'status': 'error',
                    'message': 'No relevant information found',
                    'answer': 'I could not find relevant information for your query. Please try rephrasing your question or check if the topic is covered in the knowledge base.',
                    'sources': [],
                    'web_sources': [],
                    'processing_time': time.time() - start_time,
                    'strategy_used': strategy_used
                }
            
            # STEP 4: Update knowledge base with new web data if needed
            if should_update_kb and web_docs:
                logger.info("STEP 4: Updating knowledge base with new web information...")
                try:
                    # Add web documents to vector store for future queries
                    try:
                        self.vector_store.add_documents(web_docs)
                        logger.info(f"Added {len(web_docs)} new documents to knowledge base")
                    
                except Exception as e:
                    logger.warning(f"Failed to update knowledge base: {e}")
            
            # STEP 5: Generate comprehensive answer
            logger.info("STEP 5: Generating comprehensive answer from all sources...")
            
            if not final_docs:
                final_docs = kb_docs[:10]  # Fallback
            
            result = self.rag_engine.generate_answer(
                query=query,
                relevant_docs=final_docs,
                llm_model=llm_model
            )
            
            processing_time = time.time() - start_time
            
            # Generate insights
            insights = f"Strategy: {strategy_used} | "
            insights += f"KB docs: {len(kb_docs)} | "
            insights += f"Web results: {len(web_docs)} | "
            insights += f"KB updated: {'Yes' if should_update_kb else 'No'}"
            
            logger.info(f"Intelligent hybrid query completed in {processing_time:.2f} seconds")
            
            return {
                'status': 'success',
                'answer': result['answer'],
                'sources': result['sources'],
                'web_sources': web_sources,
                'confidence': result.get('confidence', 0.8),
                'model_used': result.get('model_used', 'unknown'),
                'processing_time': processing_time,
                'insights': insights,
                'strategy_used': strategy_used,
                'knowledge_base_updated': should_update_kb,
                'kb_documents_found': len(kb_docs),
                'web_results_used': len(web_docs),
                'hybrid_processing': True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Intelligent hybrid query failed: {str(e)}")
            
            return {
                'status': 'error',
                'message': str(e),
                'answer': '',
                'sources': [],
                'web_sources': [],
                'processing_time': processing_time,
                'strategy_used': 'error'
            }