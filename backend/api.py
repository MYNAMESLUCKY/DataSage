import asyncio
import threading
import time
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import os

from .rag_engine import RAGEngine
from .data_ingestion import DataIngestionService
from .vector_store import VectorStoreManager
from .wikipedia_ingestion import WikipediaIngestionService
from .models import DataSource, QueryResult, ProcessingStatus
from .utils import setup_logging, performance_monitor
from .cache_manager import get_cache_manager, ContextManager
from .async_processor import get_async_processor, ProcessingTask
from .rag_improvements import EnhancedRetrieval, RetrievalAuditor
from .search_fallback import get_search_fallback_service
from .training_system import get_training_system

logger = setup_logging(__name__)

class RAGSystemAPI:
    """
    Main API class that orchestrates the RAG system components
    """
    
    def __init__(self):
        self.rag_engine = RAGEngine()
        self.data_ingestion = DataIngestionService()
        self.vector_store = VectorStoreManager()
        self.wikipedia_ingestion = None  # Initialize after vector store
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_ready = False
        
        # Enterprise features
        self.cache_manager = get_cache_manager()
        self.context_manager = ContextManager()
        self.async_processor = get_async_processor()
        
        # Enhanced RAG features
        self.enhanced_retrieval = EnhancedRetrieval()
        self.retrieval_auditor = RetrievalAuditor()
        self.search_fallback = get_search_fallback_service()
        self.training_system = get_training_system()
        
        # Initialize components
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system components"""
        try:
            logger.info("Initializing RAG System API...")
            
            # Initialize vector store
            self.vector_store.initialize()
            
            # Initialize RAG engine
            self.rag_engine.initialize()
            
            # Initialize Wikipedia ingestion service
            if hasattr(self.vector_store, 'chroma_store') and self.vector_store.chroma_store:
                self.wikipedia_ingestion = WikipediaIngestionService(self.vector_store.chroma_store)
            else:
                logger.warning("ChromaDB vector store not available for Wikipedia ingestion")
            
            self.is_ready = True
            logger.info("RAG System API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG System API: {str(e)}")
            self.is_ready = False
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the system"""
        try:
            status = {
                'status': 'healthy' if self.is_ready else 'unhealthy',
                'components': {
                    'rag_engine': self.rag_engine.is_ready,
                    'vector_store': self.vector_store.is_initialized,
                    'data_ingestion': True  # Always available
                },
                'timestamp': time.time()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    @performance_monitor
    def process_sources(
        self, 
        sources: List[DataSource], 
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> Dict[str, Any]:
        """
        Process multiple data sources and add them to the vector store
        """
        try:
            logger.info(f"Processing {len(sources)} data sources...")
            
            all_documents = []
            processed_sources = []
            
            for source in sources:
                try:
                    logger.info(f"Processing source: {source.url}")
                    
                    # Ingest data from source based on type
                    if source.source_type == "file" and source.file_content:
                        documents = self.data_ingestion.ingest_from_file(
                            file_content=source.file_content,
                            filename=source.name or source.url,
                            file_type=source.file_type or "text",
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    else:
                        documents = self.data_ingestion.ingest_from_url(
                            url=source.url,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    
                    if documents:
                        all_documents.extend(documents)
                        processed_sources.append(source.url)
                        logger.info(f"Successfully processed {len(documents)} chunks from {source.url}")
                    else:
                        logger.warning(f"No content extracted from {source.url}")
                        
                except Exception as e:
                    logger.error(f"Failed to process source {source.url}: {str(e)}")
                    continue
            
            if not all_documents:
                return {
                    'status': 'error',
                    'error': 'No documents could be processed from the provided sources'
                }
            
            # Update vector store with new documents
            logger.info(f"Adding {len(all_documents)} documents to vector store...")
            self.vector_store.add_documents(
                documents=all_documents,
                embedding_model=embedding_model
            )
            
            # Update RAG engine with new vector store
            self.rag_engine.update_vector_store(self.vector_store)
            
            return {
                'status': 'success',
                'processed_sources': processed_sources,
                'total_documents': len(all_documents),
                'failed_sources': [s.url for s in sources if s.url not in processed_sources]
            }
            
        except Exception as e:
            logger.error(f"Error processing sources: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @performance_monitor
    def query(
        self, 
        query: str, 
        llm_model: str = "gpt-4o",
        max_results: int = 5,
        similarity_threshold: float = 0.1,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query with intelligent caching and return answers with source attribution
        """
        try:
            start_time = time.time()
            logger.info(f"Processing query: {query[:100]}...")
            
            if not self.is_ready:
                return {
                    'status': 'error',
                    'error': 'RAG system is not ready. Please ensure data sources are processed first.',
                    'processing_time': time.time() - start_time,
                    'cached': False
                }
            
            # Generate context fingerprint for cache key
            context_fingerprint = self.context_manager.get_current_fingerprint()
            
            # Check cache first if enabled
            if use_cache:
                model_for_cache = llm_model or "default"
                cached_result = self.cache_manager.get(query, model_for_cache, context_fingerprint)
                
                if cached_result:
                    cached_result['cached'] = True
                    cached_result['processing_time'] = time.time() - start_time
                    logger.info(f"Returned cached result for query: {query[:50]}...")
                    return cached_result
            
            # Use enhanced retrieval for better document selection
            relevant_docs = self.enhanced_retrieval.enhanced_similarity_search(
                vector_store=self.vector_store,
                query=query,
                k=max_results,
                threshold=similarity_threshold
            )
            
            if not relevant_docs or len(relevant_docs) < 2:
                # Try fallback search
                logger.info("Insufficient relevant documents found, trying fallback search...")
                fallback_results = self.search_fallback.search_fallback(query, min_results=1)
                
                if fallback_results:
                    fallback_response = self.search_fallback.format_fallback_response(query, fallback_results)
                    result = {
                        'status': 'success',
                        'answer': fallback_response['answer'],
                        'sources': fallback_response['sources'],
                        'confidence': fallback_response['confidence'],
                        'processing_time': time.time() - start_time,
                        'cached': False,
                        'fallback_used': True
                    }
                    
                    # Record fallback usage
                    self.training_system.record_query(
                        query=query,
                        answer=result['answer'],
                        sources=result['sources'],
                        confidence=result['confidence'],
                        response_time=result['processing_time'],
                        fallback_used=True
                    )
                    
                    return result
                else:
                    result = {
                        'status': 'success',
                        'answer': "I couldn't find relevant information in the knowledge base or external sources to answer your question. Please try rephrasing your query with more specific terms, or ensure your data sources contain relevant content.",
                        'sources': [],
                        'confidence': 0.0,
                        'processing_time': time.time() - start_time,
                        'cached': False,
                        'fallback_used': True
                    }
                    return result
            
            # Generate answer using RAG engine
            answer_result = self.rag_engine.generate_answer(
                query=query,
                relevant_docs=relevant_docs,
                llm_model=llm_model
            )
            
            processing_time = time.time() - start_time
            logger.info(f"query executed in {processing_time:.2f} seconds")
            
            result = {
                'status': 'success',
                'answer': answer_result['answer'],
                'sources': answer_result['sources'],
                'confidence': answer_result.get('confidence', 0.8),
                'relevant_docs_count': len(relevant_docs),
                'processing_time': processing_time,
                'model_used': answer_result.get('model_used', llm_model),
                'cached': False
            }
            
            # Audit the retrieval for performance monitoring
            audit_data = self.retrieval_auditor.audit_retrieval(query, relevant_docs, result['answer'])
            result['audit_data'] = audit_data
            
            # Record query for training system
            self.training_system.record_query(
                query=query,
                answer=result['answer'],
                sources=result['sources'],
                confidence=result['confidence'],
                response_time=processing_time,
                fallback_used=False,
                audit_data=audit_data
            )
            
            # Cache the result if caching is enabled
            if use_cache:
                model_for_cache = result['model_used']
                self.cache_manager.set(query, model_for_cache, result, context_fingerprint)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
                'cached': False
            }
    
    def add_user_feedback(self, query: str, user_satisfied: bool, suggestions: str = None) -> Dict[str, Any]:
        """Add user feedback for training system improvement"""
        try:
            self.training_system.add_feedback(query, user_satisfied, suggestions)
            return {
                'status': 'success',
                'message': 'Feedback recorded successfully'
            }
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_training_insights(self) -> Dict[str, Any]:
        """Get comprehensive training insights and recommendations"""
        try:
            return {
                'status': 'success',
                'insights': self.training_system.export_training_insights()
            }
        except Exception as e:
            logger.error(f"Error getting training insights: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @performance_monitor
    def process_files(
        self,
        uploaded_files: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> Dict[str, Any]:
        """
        Process uploaded files and add them to the vector store
        """
        try:
            logger.info(f"Processing {len(uploaded_files)} uploaded files...")
            
            all_documents = []
            processed_files = []
            
            for file_data in uploaded_files:
                try:
                    filename = file_data.get('name', 'unknown_file')
                    file_content = file_data.get('content')
                    file_type = file_data.get('type', 'text')
                    
                    # Ensure file_content is bytes
                    if not isinstance(file_content, bytes):
                        if file_content is None:
                            logger.warning(f"No content found for file {filename}")
                            continue
                        if isinstance(file_content, str):
                            file_content = file_content.encode('utf-8')
                    
                    logger.info(f"Processing file: {filename} (type: {file_type})")
                    
                    # Process the file
                    documents = self.data_ingestion.ingest_from_file(
                        file_content=file_content,
                        filename=filename,
                        file_type=file_type,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    if documents:
                        all_documents.extend(documents)
                        processed_files.append(filename)
                        logger.info(f"Successfully processed {len(documents)} chunks from {filename}")
                    else:
                        logger.warning(f"No content extracted from {filename}")
                        
                except Exception as e:
                    logger.error(f"Failed to process file {file_data.get('name', 'unknown')}: {str(e)}")
                    continue
            
            if not all_documents:
                return {
                    'status': 'error',
                    'error': 'No documents could be processed from the uploaded files'
                }
            
            # Update vector store with new documents
            logger.info(f"Adding {len(all_documents)} documents to vector store...")
            self.vector_store.add_documents(
                documents=all_documents,
                embedding_model=embedding_model
            )
            
            # Update RAG engine with new vector store
            self.rag_engine.update_vector_store(self.vector_store)
            
            return {
                'status': 'success',
                'processed_files': processed_files,
                'total_documents': len(all_documents),
                'processed_count': len(all_documents)
            }
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def process_files_async(self, uploaded_files: List[Dict[str, Any]]) -> str:
        """
        Process uploaded files asynchronously and return task ID for tracking
        """
        try:
            def file_processor(file_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
                """Process a single file"""
                file_name = file_data.get('name', 'unknown')
                file_content = file_data.get('content')
                file_type = file_data.get('file_type', 'text')
                
                if not file_content:
                    raise ValueError(f"No content provided for file: {file_name}")
                
                # Process with data ingestion service
                documents = self.data_ingestion.ingest_from_file(
                    file_content=file_content,
                    filename=file_name,
                    file_type=file_type
                )
                
                if documents:
                    # Add to vector store
                    self.vector_store.add_documents(documents)
                    return {
                        'document_count': len(documents),
                        'file_name': file_name,
                        'status': 'success'
                    }
                else:
                    return {
                        'document_count': 0,
                        'file_name': file_name,
                        'status': 'no_content'
                    }
            
            # Submit to async processor
            task_id = self.async_processor.submit_file_processing(
                files=uploaded_files,
                processing_func=file_processor
            )
            
            logger.info(f"Submitted async file processing task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit file processing: {str(e)}")
            raise
    
    def get_processing_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of async processing task"""
        task = self.async_processor.get_task_status(task_id)
        if not task:
            return None
        
        return {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'status': task.status.value,
            'progress': task.progress,
            'message': task.message,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'result': task.result,
            'error': task.error,
            'metadata': task.metadata
        }
    
    def cancel_processing_task(self, task_id: str) -> bool:
        """Cancel an async processing task"""
        return self.async_processor.cancel_task(task_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.cache_manager.get_stats()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            vector_stats = self.vector_store.get_stats()
            memory_usage = self._get_memory_usage()
            
            return {
                'total_documents': vector_stats.get('total_documents', 0),
                'total_sources': len(getattr(self, '_sources', [])),
                'processing_status': 'ready',
                'embedding_model': vector_stats.get('embedding_model', 'N/A'),
                'memory_usage': memory_usage,
                'vector_store_initialized': self.vector_store.is_initialized
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                'total_documents': 0,
                'total_sources': 0,
                'processing_status': 'error',
                'error': str(e)
            }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            'is_ready': self.is_ready,
            'vector_store_size': self.vector_store.get_size(),
            'available_models': self.rag_engine.get_available_models(),
            'system_info': {
                'memory_usage': self._get_memory_usage(),
                'uptime': time.time() - getattr(self, '_start_time', time.time())
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0}
    
    def ingest_wikipedia_categories(self, categories: List[str], articles_per_category: int = 25) -> Dict[str, Any]:
        """Ingest Wikipedia articles from specific categories"""
        try:
            if not self.wikipedia_ingestion:
                return {'status': 'error', 'message': 'Wikipedia ingestion service not available'}
            
            logger.info(f"Starting Wikipedia ingestion from categories: {categories}")
            result = self.wikipedia_ingestion.ingest_wikipedia_by_categories(categories, articles_per_category)
            
            return {
                'status': 'success',
                'message': f'Ingested {result["successful"]} articles from {len(categories)} categories',
                'details': result
            }
            
        except Exception as e:
            logger.error(f"Error ingesting Wikipedia categories: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def ingest_wikipedia_random(self, count: int = 100) -> Dict[str, Any]:
        """Ingest random Wikipedia articles"""
        try:
            if not self.wikipedia_ingestion:
                return {'status': 'error', 'message': 'Wikipedia ingestion service not available'}
            
            logger.info(f"Starting random Wikipedia ingestion of {count} articles")
            result = self.wikipedia_ingestion.ingest_random_wikipedia_sample(count)
            
            return {
                'status': 'success',
                'message': f'Ingested {result["successful"]} random articles',
                'details': result
            }
            
        except Exception as e:
            logger.error(f"Error ingesting random Wikipedia articles: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def ingest_wikipedia_comprehensive(self, strategy: str = "balanced") -> Dict[str, Any]:
        """Comprehensive Wikipedia ingestion with different strategies"""
        try:
            if not self.wikipedia_ingestion:
                return {'status': 'error', 'message': 'Wikipedia ingestion service not available'}
            
            total_results = {'successful': 0, 'failed': 0, 'documents_created': 0}
            
            if strategy == "balanced":
                # Get major categories
                categories = self.wikipedia_ingestion.get_wikipedia_categories(20)
                logger.info(f"Using balanced strategy with {len(categories)} categories")
                
                # Ingest from categories
                category_result = self.wikipedia_ingestion.ingest_wikipedia_by_categories(categories, 30)
                
                # Add random articles for diversity
                random_result = self.wikipedia_ingestion.ingest_random_wikipedia_sample(200)
                
                # Combine results
                total_results['successful'] = category_result['successful'] + random_result['successful']
                total_results['failed'] = category_result['failed'] + random_result['failed']
                total_results['documents_created'] = category_result['documents_created'] + random_result['documents_created']
                
            elif strategy == "random_diverse":
                # Large random sample for maximum diversity
                random_result = self.wikipedia_ingestion.ingest_random_wikipedia_sample(1000)
                total_results = random_result
                
            elif strategy == "category_focused":
                # Focus on educational categories
                edu_categories = [
                    "Category:Science", "Category:Mathematics", "Category:Physics",
                    "Category:Computer_science", "Category:History", "Category:Geography",
                    "Category:Biology", "Category:Chemistry", "Category:Technology",
                    "Category:Philosophy", "Category:Economics", "Category:Literature"
                ]
                category_result = self.wikipedia_ingestion.ingest_wikipedia_by_categories(edu_categories, 50)
                total_results = category_result
            
            return {
                'status': 'success',
                'strategy': strategy,
                'message': f'Comprehensive ingestion completed: {total_results["successful"]} articles processed',
                'details': total_results
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive Wikipedia ingestion: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def reset_system(self) -> Dict[str, Any]:
        """Reset the entire RAG system"""
        try:
            logger.info("Resetting RAG system...")
            
            # Clear vector store
            self.vector_store.reset()
            
            # Reinitialize components
            self._initialize()
            
            return {'status': 'success', 'message': 'System reset successfully'}
            
        except Exception as e:
            logger.error(f"Error resetting system: {str(e)}")
            return {'status': 'error', 'error': str(e)}
