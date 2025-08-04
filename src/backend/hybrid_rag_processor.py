import time
from typing import Dict, List, Any, Optional
from langchain.schema import Document

from ..utils.utils import setup_logging
from .tavily_integration import TavilyIntegrationService
from .rag_improvements import EnhancedRetrieval
from .training_system import get_training_system
from .web_cache_db import WebCacheDatabase
from .query_processor import AdvancedQueryProcessor
from .reranker import AdvancedReranker
from .enhanced_cache import get_cache_manager

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
        

        # Initialize advanced processing components
        self.query_processor = AdvancedQueryProcessor(rag_engine)
        self.reranker = AdvancedReranker(rag_engine)
        self.cache_manager = get_cache_manager()
        
        # Speed optimization settings
        self.fast_mode = True  # Enable fast mode by default
        self.max_processing_time = 20.0  # Target max processing time in seconds
    
    def process_intelligent_query(
        self, 
        query: str, 
        llm_model: Optional[str] = None,
        use_web_search: bool = True,
        max_web_results: int = 10,  # Increased default to support more sources
        max_results: int = 10,      # Added max_results parameter
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
            logger.info(f"Starting advanced hybrid RAG processing: {query[:100]}...")
            
            # STEP -1: Check cache first
            cache_key_params = {
                'use_web_search': use_web_search,
                'max_web_results': max_web_results,
                'max_results': max_results,
                'llm_model': llm_model
            }
            
            cached_result = self.cache_manager.get_cached_query_result(query, cache_key_params)
            if cached_result:
                # Skip cached errors - force fresh processing for better accuracy
                if (cached_result.get('status') == 'error' or 
                    'error' in str(cached_result.get('answer', '')).lower() or
                    'nonetype' in str(cached_result.get('answer', '')).lower()):
                    logger.info("Cached result contains error - forcing fresh processing for accuracy")
                else:
                    logger.info("Retrieved valid result from cache")
                    cached_result['from_cache'] = True
                    cached_result['processing_time'] = time.time() - start_time
                    return cached_result
            
            # STEP 0: Fast Query Processing (optimized for speed)
            logger.info("STEP 0: Fast processing mode enabled...")
            
            # Skip complex processing for simple queries to save time
            if len(query) < 50:
                # Simple query - use original only
                query_rewrites = {'main': [query]}
                sub_queries = [query]
                total_variations = 1
            else:
                # Complex query - limited processing for speed
                query_analysis = self.query_processor.process_query_comprehensive(query)
                sub_queries = query_analysis.get('sub_queries', [query])[:2]  # Limit to 2 sub-queries
                query_rewrites = {'main': [query] + query_analysis.get('query_rewrites', {}).get('main', [])[:1]}  # Only 1 rewrite
                total_variations = len(query_rewrites['main'])
            
            logger.info(f"Fast query analysis: {len(sub_queries)} sub-queries, {total_variations} total variations")
            
            # STEP 1: Enhanced knowledge base search with query variations
            logger.info("STEP 1: Enhanced knowledge base search with query variations...")
            
            all_kb_docs = []
            # Search with limited query variations for speed
            search_count = 0
            max_searches = 3  # Limit total searches for speed
            for query_set_name, queries in query_rewrites.items():
                for variant_query in queries[:2]:  # Only use first 2 queries per set
                    if search_count >= max_searches:
                        break
                    kb_docs = self.enhanced_retrieval.enhanced_similarity_search(
                        self.vector_store,
                        variant_query, 
                        k=max(6, max_results)  # Reduced from 8 for speed
                    )
                    all_kb_docs.extend(kb_docs)
                    search_count += 1
                if search_count >= max_searches:
                    break
            
            # Remove duplicates while preserving order
            seen_docs = set()
            unique_kb_docs = []
            for doc in all_kb_docs:
                doc_id = getattr(doc, 'page_content', str(doc))[:100]  # Use content snippet as ID
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    unique_kb_docs.append(doc)
            
            # Fast reranking with reduced scope
            if unique_kb_docs:
                # Limit reranking candidates for speed
                rerank_candidates = unique_kb_docs[:20]  # Reduced from unlimited to 20
                logger.info(f"Fast reranking {len(rerank_candidates)} documents...")
                reranked_kb = self.reranker.rerank_documents(query, rerank_candidates, top_k=max_results)
                kb_docs = [doc for doc, score in reranked_kb]
            else:
                kb_docs = []
            
            kb_has_relevant_data = len(kb_docs) > 0
            logger.info(f"Enhanced knowledge base search: Found {len(kb_docs)} relevant documents after reranking")
            
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
                
                # Process web results into documents with safe attribute access
                for result in web_results:
                    # Safely extract attributes with fallbacks
                    title = str(getattr(result, 'title', 'Web Result'))
                    content = str(getattr(result, 'content', ''))
                    url = str(getattr(result, 'url', ''))
                    score = float(getattr(result, 'score', 0.0))
                    
                    web_doc = Document(
                        page_content=f"Title: {title}\n\nContent: {content}",
                        metadata={
                            "source": url,
                            "title": title,
                            "type": "web_data",
                            "score": score
                        }
                    )
                    web_docs.append(web_doc)
                
                web_sources = [{"url": str(getattr(r, 'url', '')), "title": str(getattr(r, 'title', '')), "score": float(getattr(r, 'score', 0.0))} for r in web_results]
            
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
                    self.vector_store.add_documents(web_docs)
                    logger.info(f"Added {len(web_docs)} new documents to knowledge base")
                    
                except Exception as e:
                    logger.warning(f"Failed to update knowledge base: {e}")
            
            # STEP 5: Generate comprehensive answer
            logger.info("STEP 5: Generating comprehensive answer from all sources...")
            
            if not final_docs:
                final_docs = kb_docs[:10]  # Fallback
            
            # Always ensure we can provide an answer with available data
            try:
                result = self.rag_engine.generate_answer(
                    query=query,
                    relevant_docs=final_docs,
                    llm_model=llm_model
                )
                
                # Check if result indicates failure but we have documents - use KB fallback
                if (result.get('status') == 'error' and final_docs and 
                    'rate limit' in result.get('answer', '').lower()):
                    
                    logger.warning("API failed but we have relevant documents - using knowledge base fallback")
                    
                    # Force generate answer using knowledge base
                    context = self.rag_engine._prepare_context(final_docs)
                    fallback_result = self.rag_engine._generate_enhanced_fallback_answer(query, context)
                    
                    result = {
                        'answer': fallback_result['answer'],
                        'confidence': fallback_result['confidence'],
                        'status': 'success_kb_fallback',
                        'model_used': f"{llm_model} (knowledge base fallback)",
                        'sources': [doc.metadata.get('source', f'Document {i+1}') for i, doc in enumerate(final_docs[:5])],
                        'fallback_reason': 'Rate limit exceeded - used knowledge base data'
                    }
                    
            except Exception as e:
                logger.error(f"Error in answer generation: {e}")
                
                # Emergency fallback - always provide something if we have docs
                if final_docs:
                    logger.info("Emergency fallback - extracting direct information from documents")
                    context = self.rag_engine._prepare_context(final_docs)
                    fallback_result = self.rag_engine._generate_enhanced_fallback_answer(query, context)
                    
                    result = {
                        'answer': fallback_result['answer'],
                        'confidence': fallback_result['confidence'],
                        'status': 'emergency_fallback',
                        'model_used': 'Knowledge Base Direct Access',
                        'sources': [doc.metadata.get('source', f'Document {i+1}') for i, doc in enumerate(final_docs[:5])],
                        'fallback_reason': f'Complete API failure - emergency knowledge base access: {str(e)}'
                    }
                else:
                    result = {
                        'answer': "I encountered technical difficulties and no relevant information is available in the knowledge base for your query.",
                        'confidence': 0.1,
                        'status': 'no_data_error',
                        'model_used': 'None',
                        'sources': [],
                        'fallback_reason': f'No documents available and API failed: {str(e)}'
                    }
            
            processing_time = time.time() - start_time
            
            # Generate insights
            insights = f"Strategy: {strategy_used} | "
            insights += f"KB docs: {len(kb_docs)} | "
            insights += f"Web results: {len(web_docs)} | "
            insights += f"KB updated: {'Yes' if should_update_kb else 'No'}"
            
            logger.info(f"Intelligent hybrid query completed in {processing_time:.2f} seconds")
            
            final_result = {
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
                'hybrid_processing': True,
                'query_processing': {
                    'sub_queries': len(sub_queries),
                    'query_variations': query_analysis.get('total_queries', 1),
                    'routing_strategy': routing_info.get('strategy', 'hybrid'),
                    'processing_time': query_analysis.get('processing_time', 0)
                },
                'reranking_applied': True,
                'from_cache': False
            }
            
            # Cache the complete result for future queries
            self.cache_manager.cache_query_result(query, final_result, cache_key_params)
            
            return final_result
            
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