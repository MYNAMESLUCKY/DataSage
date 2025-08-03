import time
from typing import Dict, List, Any, Optional
from langchain.schema import Document

from ..utils.utils import setup_logging
from .tavily_integration import TavilyIntegrationService
from .rag_improvements import EnhancedRetrieval
from .training_system import get_training_system
from .web_cache_db import WebCacheDatabase

logger = setup_logging(__name__)

class WebRAGProcessor:
    """
    Processor that combines existing knowledge base with real-time web search using Tavily
    """
    
    def __init__(self, vector_store, rag_engine, enhanced_retrieval):
        self.vector_store = vector_store
        self.rag_engine = rag_engine
        self.enhanced_retrieval = enhanced_retrieval
        self.tavily_service = TavilyIntegrationService()
        self.training_system = get_training_system()
        self.web_cache = WebCacheDatabase()
    
    def process_query_with_web(
        self, 
        query: str, 
        llm_model: Optional[str] = None,
        use_web_search: bool = True,
        max_web_results: int = 5,
        prioritize_web: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query with intelligent web search and database caching
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query with intelligent web integration: {query[:100]}...")
            
            web_sources = []
            relevant_docs = []
            web_content_added = False
            cache_hit = False
            
            # Step 1: Check cache first for web results
            if use_web_search and self.web_cache.is_connected:
                cached_results = self.web_cache.get_cached_results(query, max_age_hours=24)
                if cached_results:
                    logger.info(f"Using cached web results for query: {query[:50]}...")
                    web_results = [
                        type('TavilySearchResult', (), {
                            'url': result.get('url', ''),
                            'title': result.get('title', ''),
                            'content': result.get('content', ''),
                            'score': result.get('score', 0.0)
                        })() for result in cached_results.results
                    ]
                    cache_hit = True
                    web_content_added = True
                else:
                    web_results = []
            
            # Step 2: If no cache hit and web search enabled, fetch fresh results
            if use_web_search and self.tavily_service.is_available() and not cache_hit:
                try:
                    logger.info(f"Fetching fresh web results for: {query[:50]}...")
                    
                    # For any question, search the web directly without limiting to existing docs
                    web_results = self.tavily_service.search_and_fetch(
                        query=query,
                        max_results=max_web_results,
                        search_depth="advanced"
                    )
                    
                    if web_results:
                        # Cache the results for future use
                        results_for_cache = [
                            {
                                "url": result.url,
                                "title": result.title,
                                "content": result.content,
                                "score": result.score,
                                "published_date": result.published_date
                            } for result in web_results
                        ]
                        
                        self.web_cache.cache_search_results(
                            query=query,
                            results=results_for_cache,
                            relevance_score=sum(r.score for r in web_results) / len(web_results)
                        )
                        
                        # Also cache individual processed content
                        for result in web_results:
                            self.web_cache.cache_processed_content(
                                url=result.url,
                                title=result.title,
                                content=result.content,
                                source_query=query,
                                quality_score=result.score
                            )
                        
                        web_content_added = True
                        logger.info(f"Fetched and cached {len(web_results)} fresh web results")
                
                except Exception as e:
                    logger.warning(f"Fresh web search failed, checking cached content: {e}")
                    # Fallback to cached content search
                    cached_content = self.web_cache.search_cached_content(query, limit=max_web_results)
                    if cached_content:
                        web_results = [
                            type('TavilySearchResult', (), {
                                'url': content['url'],
                                'title': content['title'],
                                'content': content['content'],
                                'score': content['quality_score']
                            })() for content in cached_content
                        ]
                        web_content_added = True
                        logger.info(f"Used {len(web_results)} cached content items as fallback")
            
            # Step 3: Add web content to documents for processing
            if web_results and web_content_added:
                for result in web_results:
                    web_doc = Document(
                        page_content=result.content,
                        metadata={
                            "source": result.url,
                            "title": result.title,
                            "type": "web_search",
                            "search_score": result.score,
                            "cached": cache_hit
                        }
                    )
                    relevant_docs.append(web_doc)
                
                web_sources = [{"url": result.url, "title": result.title, "score": result.score} for result in web_results]
            
            # Step 4: Only use existing knowledge base if no web results or as supplement
            if not prioritize_web or (not web_content_added and not use_web_search):
                kb_docs = self.enhanced_retrieval.retrieve_documents(
                    query, 
                    self.vector_store,
                    max_docs=5 if web_content_added else 15,
                    similarity_threshold=0.1
                )
                relevant_docs.extend(kb_docs)
            
            # If we have no content at all, that's an error
            if not relevant_docs:
                return {
                    'status': 'error',
                    'message': 'No relevant information found in knowledge base or web search',
                    'answer': 'I could not find any relevant information to answer your question. Please try rephrasing your query or check if web search is enabled.',
                    'sources': [],
                    'web_sources': [],
                    'processing_time': time.time() - start_time
                }
            
            # Step 5: Generate comprehensive answer
            result = self.rag_engine.generate_answer(
                query=query,
                relevant_docs=relevant_docs,
                llm_model=llm_model
            )
            
            processing_time = time.time() - start_time
            
            # Step 6: Analyze performance and provide insights
            insights = self.training_system.analyze_query(query, result, processing_time)
            if web_content_added:
                insights += f" | Web search {'(cached)' if cache_hit else '(fresh)'}"
            if self.web_cache.is_connected:
                insights += " | Database caching enabled"
            
            logger.info(f"Web-enhanced query executed in {processing_time:.2f} seconds")
            
            return {
                'status': 'success',
                'answer': result['answer'],
                'sources': result['sources'],
                'web_sources': web_sources,
                'confidence': result.get('confidence', 0.8),
                'model_used': result.get('model_used', 'unknown'),
                'processing_time': processing_time,
                'insights': insights,
                'documents_found': len([doc for doc in relevant_docs if doc.metadata.get('type') != 'web_search']),
                'web_results_used': len(web_sources),
                'cache_hit': cache_hit,
                'web_search_enabled': use_web_search and self.tavily_service.is_available(),
                'database_caching': self.web_cache.is_connected
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Web-enhanced query failed: {str(e)}")
            
            return {
                'status': 'error',
                'message': str(e),
                'answer': '',
                'sources': [],
                'web_sources': [],
                'processing_time': processing_time
            }
                    
                    if web_results:
                        # Add web content to relevant docs for this query
                        for result in web_results:
                            web_doc = Document(
                                page_content=result.content,
                                metadata={
                                    "source": result.url,
                                    "title": result.title,
                                    "type": "web_search",
                                    "search_score": result.score,
                                    "published_date": result.published_date
                                }
                            )
                            relevant_docs.append(web_doc)
                        
                        web_sources = [{"url": result.url, "title": result.title, "score": result.score} for result in web_results]
                        web_content_added = True
                        logger.info(f"Added {len(web_results)} web search results to context")
                
                except Exception as e:
                    logger.warning(f"Web search failed, continuing with existing docs: {e}")
            
            # Step 3: Generate comprehensive answer
            result = self.rag_engine.generate_answer(
                query=query,
                relevant_docs=relevant_docs,
                llm_model=llm_model
            )
            
            processing_time = time.time() - start_time
            
            # Step 4: Analyze performance and provide insights
            insights = self.training_system.analyze_query(query, result, processing_time)
            if web_content_added:
                insights += " | Web search enhanced context"
            
            logger.info(f"Web-enhanced query executed in {processing_time:.2f} seconds")
            
            return {
                'status': 'success',
                'answer': result['answer'],
                'sources': result['sources'],
                'web_sources': web_sources,
                'confidence': result.get('confidence', 0.8),
                'model_used': result.get('model_used', 'unknown'),
                'processing_time': processing_time,
                'insights': insights,
                'documents_found': len([doc for doc in relevant_docs if doc.metadata.get('type') != 'web_search']),
                'web_results_used': len(web_sources),
                'web_search_enabled': use_web_search and self.tavily_service.is_available()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Web-enhanced query failed: {str(e)}")
            
            return {
                'status': 'error',
                'message': str(e),
                'answer': '',
                'sources': [],
                'web_sources': [],
                'processing_time': processing_time
            }
    
    def ingest_web_search_permanently(
        self, 
        query: str, 
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search web content and permanently add it to the knowledge base
        """
        start_time = time.time()
        
        try:
            if not self.tavily_service.is_available():
                return {
                    'status': 'error',
                    'message': 'Tavily service not available',
                    'documents_added': 0
                }
            
            logger.info(f"Permanently ingesting web content for query: {query}")
            
            # Search for web content
            web_results = self.tavily_service.search_and_fetch(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )
            
            if not web_results:
                return {
                    'status': 'warning',
                    'message': 'No web results found',
                    'documents_added': 0
                }
            
            # Convert to data sources and process them
            data_sources = self.tavily_service.create_data_sources_from_results(
                web_results, query
            )
            
            # Add to vector store permanently
            documents_added = 0
            for data_source in data_sources:
                try:
                    # Process through normal ingestion pipeline
                    if hasattr(self.vector_store, 'add_document'):
                        self.vector_store.add_document(
                            content=data_source.content,
                            metadata={
                                "source": data_source.url,
                                "title": data_source.title,
                                "type": "web_search_permanent",
                                "search_query": query,
                                "ingested_at": data_source.metadata.get("fetched_at"),
                                "search_score": data_source.metadata.get("search_score", 0.0)
                            }
                        )
                        documents_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add document to vector store: {e}")
            
            processing_time = time.time() - start_time
            
            logger.info(f"Permanently ingested {documents_added} web documents in {processing_time:.2f} seconds")
            
            return {
                'status': 'success',
                'message': f'Successfully added {documents_added} web documents to knowledge base',
                'documents_added': documents_added,
                'processing_time': processing_time,
                'sources': [{"url": result.url, "title": result.title} for result in web_results[:documents_added]]
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Permanent web ingestion failed: {str(e)}")
            
            return {
                'status': 'error',
                'message': str(e),
                'documents_added': 0,
                'processing_time': processing_time
            }