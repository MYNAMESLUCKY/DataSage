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
from .complexity_classifier import classify_query_complexity, ComplexityLevel
from .gpu_processor import gpu_processor

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
        
        # GPU processing for complex queries
        self.gpu_processor = gpu_processor
        
        # Advanced mathematical and physics-enhanced processing
        from src.backend.advanced_mathematics import advanced_math_processor
        from src.backend.physics_enhanced_search import physics_search
        from src.backend.advanced_optimization import advanced_rag_optimizer
        self.advanced_math = advanced_math_processor
        self.physics_search = physics_search
        self.advanced_optimizer = advanced_rag_optimizer
    
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
            
            # STEP -2: Speed optimization for simple queries
            from src.backend.speed_optimizer import speed_optimizer
            
            if speed_optimizer.should_use_fast_path(query):
                logger.info("Query eligible for speed optimization - using knowledge base optimized processing")
                
                fast_response = speed_optimizer.get_optimized_response(query, self)
                if fast_response and fast_response.get('optimization_path') in ['basic_definition_cache', 'lightweight_processing']:
                    processing_time = time.time() - start_time
                    
                    # Format as hybrid processor response
                    return {
                        'status': 'success',
                        'answer': fast_response['answer'],
                        'sources': fast_response.get('sources', []),
                        'web_sources': fast_response.get('web_sources', []),
                        'confidence': fast_response.get('confidence', 0.8),
                        'model_used': f"{llm_model} (KB + web optimized)",
                        'processing_time': processing_time,
                        'optimization_used': True,
                        'optimization_path': fast_response['optimization_path'],
                        'fast_response_time_ms': fast_response.get('total_processing_time_ms', 0),
                        'web_cache_used': fast_response.get('web_cache_used', False),
                        'web_processing_time_ms': fast_response.get('web_processing_time_ms', 0),
                        'strategy': 'hybrid_speed_optimized',
                        'insights': f"Hybrid speed optimization - KB + fast web search ({fast_response['optimization_path']})",
                        'cache_type': fast_response.get('cache_type', 'hybrid')
                    }
            
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
            
            # STEP 0: Classify query complexity for GPU processing decision
            complexity_analysis = classify_query_complexity(query)
            logger.info(f"Query complexity: {complexity_analysis.score:.3f} ({complexity_analysis.level.value}) - GPU recommended: {complexity_analysis.gpu_recommended}")
            
            # STEP 0.5: Advanced Query Processing
            logger.info("STEP 0.5: Processing query with advanced techniques...")
            query_analysis = self.query_processor.process_query_comprehensive(query)
            
            # Extract processing results
            sub_queries = query_analysis.get('sub_queries', [query])
            query_rewrites = query_analysis.get('query_rewrites', {'main': [query]})
            routing_info = query_analysis.get('routing', {})
            
            logger.info(f"Query analysis: {len(sub_queries)} sub-queries, {query_analysis.get('total_queries', 1)} total variations")
            
            # STEP 1: Enhanced knowledge base search with query variations
            logger.info("STEP 1: Enhanced knowledge base search with query variations...")
            
            all_kb_docs = []
            # Search with all query variations
            for query_set_name, queries in query_rewrites.items():
                for variant_query in queries:
                    kb_docs = self.enhanced_retrieval.enhanced_similarity_search(
                        self.vector_store,
                        variant_query, 
                        k=max(8, max_results // len(query_rewrites))  # Scale with max_results
                    )
                    all_kb_docs.extend(kb_docs)
            
            # Remove duplicates while preserving order
            seen_docs = set()
            unique_kb_docs = []
            for doc in all_kb_docs:
                doc_id = getattr(doc, 'page_content', str(doc))[:100]  # Use content snippet as ID
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    unique_kb_docs.append(doc)
            
            # Advanced physics-enhanced reranking
            if unique_kb_docs:
                logger.info(f"Applying advanced mathematical and physics-enhanced reranking to {len(unique_kb_docs)} documents...")
                
                # Standard reranking first
                reranked_kb = self.reranker.rerank_documents(query, unique_kb_docs, top_k=max_results)
                
                # Extract embeddings for advanced processing
                doc_embeddings = []
                doc_texts = []
                for doc in unique_kb_docs:
                    # Create embeddings if not available (simplified)
                    doc_text = getattr(doc, 'page_content', str(doc))
                    doc_texts.append(doc_text)
                    
                    # Simple embedding from text (in production, use proper embedding model)
                    char_values = [ord(c) for c in doc_text.lower()[:100] if c.isalnum()]
                    while len(char_values) < 100:
                        char_values.append(0)
                    doc_embeddings.append(np.array(char_values[:100]))
                
                if doc_embeddings:
                    # Physics-enhanced ranking
                    query_chars = [ord(c) for c in query.lower()[:100] if c.isalnum()]
                    while len(query_chars) < 100:
                        query_chars.append(0)
                    query_embedding = np.array(query_chars[:100])
                    
                    # Apply multiple physics-based ranking algorithms
                    gravitational_ranking = self.physics_search.gravitational_ranking(
                        [{"embedding": emb} for emb in doc_embeddings], query_embedding
                    )
                    
                    wave_interference_scores = self.physics_search.wave_interference_ranking(
                        query_embedding, doc_embeddings
                    )
                    
                    # Combine standard + physics-based scores
                    enhanced_scores = []
                    for i, (doc, standard_score) in enumerate(reranked_kb):
                        if i < len(wave_interference_scores):
                            physics_score = wave_interference_scores[i]
                            combined_score = 0.7 * standard_score + 0.3 * physics_score
                            enhanced_scores.append((doc, combined_score))
                        else:
                            enhanced_scores.append((doc, standard_score))
                    
                    # Re-sort by enhanced scores
                    enhanced_scores.sort(key=lambda x: x[1], reverse=True)
                    kb_docs = [doc for doc, score in enhanced_scores]
                    
                    logger.info("Applied physics-enhanced ranking with gravitational and wave interference algorithms")
                else:
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
            
            # STEP 5: Determine processing strategy based on complexity
            logger.info("STEP 5: Determining processing strategy based on query complexity...")
            
            if not final_docs:
                final_docs = kb_docs[:10]  # Fallback
            
            # Check if GPU processing is recommended for this complex query
            if complexity_analysis.gpu_recommended and complexity_analysis.score >= 0.6:
                logger.info(f"Complex query detected (score: {complexity_analysis.score:.3f}) - attempting GPU processing")
                
                try:
                    # Prepare context for GPU processing
                    context = self.rag_engine._prepare_context(final_docs)
                    
                    # Use async GPU processing (simulate with sync call for now)
                    import asyncio
                    
                    async def gpu_process():
                        async with self.gpu_processor as gpu:
                            return await gpu.process_on_gpu(
                                query=query,
                                context=context,
                                complexity_score=complexity_analysis.score
                            )
                    
                    # Execute GPU processing
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    gpu_response = loop.run_until_complete(gpu_process())
                    
                    if gpu_response.success:
                        logger.info(f"GPU processing successful in {gpu_response.compute_time:.2f}s")
                        
                        result = {
                            'answer': gpu_response.result,
                            'confidence': gpu_response.confidence,
                            'status': 'success_gpu',
                            'model_used': f"GPU-{gpu_response.provider.value if gpu_response.provider else 'distributed'}",
                            'sources': [doc.metadata.get('source', f'Document {i+1}') for i, doc in enumerate(final_docs[:5])],
                            'gpu_processing': True,
                            'gpu_compute_time': gpu_response.compute_time,
                            'complexity_analysis': {
                                'score': complexity_analysis.score,
                                'level': complexity_analysis.level.value,
                                'reasoning': complexity_analysis.reasoning
                            }
                        }
                    else:
                        logger.warning(f"GPU processing failed: {gpu_response.error}")
                        raise Exception(f"GPU processing failed: {gpu_response.error}")
                        
                except Exception as gpu_error:
                    logger.warning(f"GPU processing failed, falling back to standard processing: {gpu_error}")
                    
                    # Fallback to standard processing
                    result = self.rag_engine.generate_answer(
                        query=query,
                        relevant_docs=final_docs,
                        llm_model=llm_model
                    )
                    
                    # Add complexity info to standard result
                    result['gpu_processing'] = False
                    result['gpu_fallback_reason'] = str(gpu_error)
                    result['complexity_analysis'] = {
                        'score': complexity_analysis.score,
                        'level': complexity_analysis.level.value,
                        'gpu_recommended': True,
                        'gpu_failed': True
                    }
            
            else:
                # Standard processing for simple/moderate queries
                logger.info("Standard processing for simple/moderate complexity query")
                
                try:
                    result = self.rag_engine.generate_answer(
                        query=query,
                        relevant_docs=final_docs,
                        llm_model=llm_model
                    )
                    
                    # Add complexity info
                    result['gpu_processing'] = False
                    result['complexity_analysis'] = {
                        'score': complexity_analysis.score,
                        'level': complexity_analysis.level.value,
                        'gpu_recommended': False
                    }
                    
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
            
            # Add advanced mathematical analysis to results
            if final_docs:
                try:
                    # Advanced mathematical analysis
                    doc_texts = [getattr(doc, 'page_content', str(doc)) for doc in final_docs[:5]]
                    
                    # Harmonic analysis
                    harmonic_analysis = self.advanced_math.harmonic_analysis(doc_texts)
                    
                    # Thermodynamic information theory
                    thermo_analysis = self.physics_search.thermodynamic_information_theory(doc_texts)
                    
                    # Apply advanced optimization techniques
                    optimization_metrics = self.advanced_optimizer.get_optimization_metrics()
                    
                    # Add to result metadata
                    result['advanced_analysis'] = {
                        'harmonic_resonance_frequency': harmonic_analysis.resonance_frequency,
                        'information_entropy': harmonic_analysis.entropy_measure,
                        'phase_coherence': harmonic_analysis.phase_coherence,
                        'thermodynamic_temperature': thermo_analysis.get('temperature', 0.0),
                        'information_density': thermo_analysis.get('information_density', 0.0),
                        'physics_enhanced': True,
                        'optimization_applied': True,
                        'compression_ratio': optimization_metrics.compression_ratio,
                        'search_speedup': optimization_metrics.search_speedup,
                        'memory_reduction': optimization_metrics.memory_reduction,
                        'accuracy_retention': optimization_metrics.accuracy_retention
                    }
                    
                    logger.info("Applied advanced mathematical and thermodynamic analysis to results")
                    
                except Exception as e:
                    logger.warning(f"Advanced analysis failed: {e}")
                    result['advanced_analysis'] = {'physics_enhanced': False, 'error': str(e)}
            
            # Generate insights
            insights = f"Strategy: {strategy_used} | "
            insights += f"KB docs: {len(kb_docs)} | "
            insights += f"Web results: {len(web_docs)} | "
            insights += f"KB updated: {'Yes' if should_update_kb else 'No'} | "
            insights += f"Physics Enhanced: {'Yes' if result.get('advanced_analysis', {}).get('physics_enhanced') else 'No'} | "
            insights += f"Advanced Optimization: {'Yes' if result.get('advanced_analysis', {}).get('optimization_applied') else 'No'}"
            
            logger.info(f"Intelligent hybrid query completed in {processing_time:.2f} seconds")
        
        # Record query metrics for analytics
        try:
            from src.analytics.system_monitor import system_monitor, QueryMetrics
            import uuid
            
            # Calculate estimated cost (rough estimate based on model and tokens)
            estimated_cost = 0.0
            token_count = len(str(result.get('answer', '')).split()) * 1.3  # Rough token estimate
            
            if llm_model == 'sarvam-m':
                estimated_cost = token_count * 0.00001  # Example rate
            elif 'deepseek' in llm_model:
                estimated_cost = token_count * 0.000005
            elif 'openai' in llm_model:
                estimated_cost = token_count * 0.00003
                
            # Calculate quality score based on result
            quality_score = 4.0  # Default good quality
            if result.get('status') == 'success':
                quality_score = 4.5
            elif result.get('status') == 'error':
                quality_score = 2.0
            elif 'fallback' in result.get('status', ''):
                quality_score = 3.5
                
            query_metrics = QueryMetrics(
                query_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                query_text=query[:200],  # Truncate long queries
                processing_time=processing_time,
                model_used=llm_model,
                success=result.get('status') != 'error',
                token_count=int(token_count),
                cost_estimate=estimated_cost,
                user_id=getattr(st.session_state, 'user_id', 'anonymous') if 'st' in globals() else 'anonymous',
                response_quality_score=quality_score
            )
            
            system_monitor.record_query_metrics(query_metrics)
            
        except Exception as analytics_error:
            logger.warning(f"Failed to record analytics: {analytics_error}")
            
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