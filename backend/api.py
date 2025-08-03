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
        similarity_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Process a query and return intelligent answers with source attribution
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            if not self.is_ready:
                return {
                    'status': 'error',
                    'error': 'RAG system is not ready. Please ensure data sources are processed first.'
                }
            
            # Get relevant documents from vector store
            relevant_docs = self.vector_store.similarity_search(
                query=query,
                k=max_results,
                threshold=similarity_threshold
            )
            
            if not relevant_docs:
                return {
                    'status': 'success',
                    'answer': "I couldn't find relevant information in the processed data to answer your question. Please try rephrasing your query or ensure your data sources contain relevant content.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Generate answer using RAG engine
            result = self.rag_engine.generate_answer(
                query=query,
                relevant_docs=relevant_docs,
                llm_model=llm_model
            )
            
            return {
                'status': 'success',
                'answer': result['answer'],
                'sources': result['sources'],
                'confidence': result.get('confidence', 0.8),
                'relevant_docs_count': len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
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
