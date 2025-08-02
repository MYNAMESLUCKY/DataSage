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
                    
                    # Ingest data from source
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
        similarity_threshold: float = 0.7
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
