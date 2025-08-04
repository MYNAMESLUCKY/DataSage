"""
Hybrid GPU-Accelerated RAG Processor
Integrates GPU infrastructure with existing RAG pipeline for sub-second processing
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .hybrid_rag_processor import HybridRAGProcessor
from .gpu_accelerator import get_gpu_manager
from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class HybridGPUProcessor:
    """
    Enhanced RAG processor with GPU acceleration for sub-second query processing
    Intelligently offloads heavy computations to GPU infrastructure
    """
    
    def __init__(self, hybrid_processor: HybridRAGProcessor):
        self.hybrid_processor = hybrid_processor
        self.gpu_manager = get_gpu_manager()
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Performance targets
        self.target_response_time = 3.0  # seconds
        self.gpu_threshold_time = 1.0    # Use GPU if local processing > 1s
        
    async def process_accelerated_query(
        self, 
        query: str, 
        llm_model: Optional[str] = None,
        max_results: int = 5,
        use_gpu_acceleration: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query with GPU acceleration for maximum speed
        
        Args:
            query: User query
            llm_model: Model to use for generation
            max_results: Maximum results to return
            use_gpu_acceleration: Whether to use GPU acceleration
        """
        start_time = time.time()
        gpu_tasks_completed = 0
        
        try:
            logger.info(f"Starting GPU-accelerated processing for: {query[:100]}...")
            
            # Step 1: Parallel GPU-accelerated embedding generation and vector search
            if use_gpu_acceleration:
                embedding_future = self._async_generate_embeddings(query)
                search_future = self._async_vector_search(query, max_results)
                
                # Wait for both to complete
                query_embedding, initial_results = await asyncio.gather(
                    embedding_future,
                    search_future,
                    return_exceptions=True
                )
                
                if isinstance(query_embedding, Exception):
                    logger.warning(f"GPU embedding failed: {query_embedding}")
                    query_embedding = None
                
                if isinstance(initial_results, Exception):
                    logger.warning(f"GPU search failed: {initial_results}")
                    initial_results = []
                else:
                    gpu_tasks_completed += 1
            else:
                # Fallback to local processing
                initial_results = []
                query_embedding = None
            
            # Step 2: Enhanced knowledge base search (potentially GPU-accelerated)
            kb_docs = await self._accelerated_kb_search(query, max_results, use_gpu_acceleration)
            
            # Step 3: GPU-accelerated reranking if we have enough documents
            if len(kb_docs) > 3 and use_gpu_acceleration:
                try:
                    reranked_docs = await self._async_rerank_documents(query, kb_docs, max_results)
                    kb_docs = reranked_docs
                    gpu_tasks_completed += 1
                except Exception as e:
                    logger.warning(f"GPU reranking failed: {e}")
                    # Keep original docs if GPU reranking fails
            
            # Step 4: Parallel web search and final processing
            web_results = []
            if kwargs.get('use_web_search', True):
                web_future = self._async_web_search(query, kwargs.get('max_web_results', 5))
                web_results = await web_future
            
            # Step 5: Final answer generation (use fastest available model)
            final_answer = await self._accelerated_answer_generation(
                query, kb_docs, web_results, llm_model, use_gpu_acceleration
            )
            
            processing_time = time.time() - start_time
            
            # Determine if we met our performance targets
            performance_level = self._assess_performance(processing_time, gpu_tasks_completed)
            
            result = {
                'status': 'success',
                'answer': final_answer['answer'],
                'sources': [doc.metadata.get('source', f'Document {i+1}') for i, doc in enumerate(kb_docs)],
                'web_sources': [r.get('url', '') for r in web_results] if web_results else [],
                'confidence': final_answer.get('confidence', 0.85),
                'model_used': final_answer.get('model_used', llm_model),
                'processing_time': processing_time,
                'gpu_acceleration': {
                    'enabled': use_gpu_acceleration,
                    'tasks_completed': gpu_tasks_completed,
                    'performance_level': performance_level,
                    'infrastructure_status': self.gpu_manager.get_infrastructure_status()
                },
                'performance_metrics': {
                    'target_time': self.target_response_time,
                    'actual_time': processing_time,
                    'speed_ratio': self.target_response_time / processing_time if processing_time > 0 else 1.0,
                    'sub_second_achieved': processing_time < 1.0
                }
            }
            
            logger.info(f"GPU-accelerated query completed in {processing_time:.2f}s (target: {self.target_response_time}s)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"GPU-accelerated processing failed after {processing_time:.2f}s: {e}")
            
            # Ultimate fallback to standard hybrid processing
            return await self._fallback_to_standard_processing(query, llm_model, max_results, **kwargs)
    
    async def _async_generate_embeddings(self, query: str) -> Optional[List[float]]:
        """Generate embeddings using GPU acceleration"""
        try:
            result = await self.gpu_manager.process_heavy_computation(
                task_type="embedding",
                data={'texts': [query]},
                priority="urgent"
            )
            
            if result['status'] == 'success' and result['gpu_accelerated']:
                return result['result'][0]  # First embedding
            
        except Exception as e:
            logger.warning(f"GPU embedding generation failed: {e}")
        
        return None
    
    async def _async_vector_search(self, query: str, max_results: int) -> List[Any]:
        """Perform vector search using GPU acceleration"""
        try:
            # Get query embedding first
            embedding = await self._async_generate_embeddings(query)
            
            if embedding:
                result = await self.gpu_manager.process_heavy_computation(
                    task_type="similarity_search",
                    data={
                        'query_embedding': embedding,
                        'vectors': [],  # Would contain actual vector database
                        'k': max_results
                    },
                    priority="urgent"
                )
                
                if result['status'] == 'success':
                    return result['result']
        
        except Exception as e:
            logger.warning(f"GPU vector search failed: {e}")
        
        return []
    
    async def _accelerated_kb_search(self, query: str, max_results: int, use_gpu: bool) -> List[Any]:
        """Perform accelerated knowledge base search"""
        try:
            if use_gpu:
                # Try GPU-accelerated search first
                gpu_results = await self._async_vector_search(query, max_results * 2)
                if gpu_results:
                    return gpu_results[:max_results]
            
            # Fallback to local enhanced search
            loop = asyncio.get_event_loop()
            kb_docs = await loop.run_in_executor(
                self.executor,
                self.hybrid_processor.enhanced_retrieval.enhanced_similarity_search,
                self.hybrid_processor.vector_store,
                query,
                max_results
            )
            
            return kb_docs
            
        except Exception as e:
            logger.error(f"Accelerated KB search failed: {e}")
            return []
    
    async def _async_rerank_documents(self, query: str, documents: List[Any], top_k: int) -> List[Any]:
        """Rerank documents using GPU acceleration"""
        try:
            result = await self.gpu_manager.process_heavy_computation(
                task_type="reranking",
                data={
                    'query': query,
                    'documents': [{'content': getattr(doc, 'page_content', str(doc))} for doc in documents]
                },
                priority="normal"
            )
            
            if result['status'] == 'success' and result['gpu_accelerated']:
                # Convert GPU results back to document format
                gpu_results = result['result']
                reranked_docs = []
                
                for gpu_result in gpu_results[:top_k]:
                    # Find corresponding original document
                    original_doc = documents[gpu_result.get('rank', 1) - 1]
                    reranked_docs.append(original_doc)
                
                return reranked_docs
        
        except Exception as e:
            logger.warning(f"GPU reranking failed: {e}")
        
        # Fallback to local reranking
        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            self.executor,
            self.hybrid_processor.reranker.rerank_documents,
            query,
            documents,
            top_k
        )
        
        return [doc for doc, score in reranked]
    
    async def _async_web_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform async web search"""
        try:
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(
                self.executor,
                self._sync_web_search,
                query,
                max_results
            )
            return web_results
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []
    
    def _sync_web_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Synchronous web search helper"""
        try:
            if self.hybrid_processor.tavily_service.is_available():
                results = self.hybrid_processor.tavily_service.search(query, max_results=max_results)
                return [{'url': r.url, 'title': r.title, 'content': r.content} for r in results]
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
        
        return []
    
    async def _accelerated_answer_generation(
        self, 
        query: str, 
        kb_docs: List[Any], 
        web_results: List[Dict[str, Any]], 
        llm_model: Optional[str],
        use_gpu: bool
    ) -> Dict[str, Any]:
        """Generate final answer with acceleration"""
        try:
            # Prepare context from all sources
            context_parts = []
            
            # Add KB context
            for doc in kb_docs[:5]:  # Top 5 documents
                content = getattr(doc, 'page_content', str(doc))
                context_parts.append(content[:500])  # Truncate for speed
            
            # Add web context
            for result in web_results[:3]:  # Top 3 web results
                context_parts.append(result.get('content', '')[:300])
            
            context = '\n\n'.join(context_parts)
            
            # Use fastest available generation method
            if use_gpu and len(context) > 1000:
                # For large contexts, consider GPU acceleration
                # This would integrate with GPU providers that support text generation
                pass
            
            # Generate answer using standard RAG engine (optimized for speed)
            loop = asyncio.get_event_loop()
            answer_result = await loop.run_in_executor(
                self.executor,
                self._generate_fast_answer,
                query,
                context,
                llm_model
            )
            
            return answer_result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': 'Unable to generate answer due to processing error.',
                'confidence': 0.1,
                'model_used': 'error_fallback'
            }
    
    def _generate_fast_answer(self, query: str, context: str, llm_model: Optional[str]) -> Dict[str, Any]:
        """Generate answer optimized for speed"""
        try:
            # Use the RAG engine with speed optimizations
            response = self.hybrid_processor.rag_engine.generate_answer(
                query=query,
                context=context,
                llm_model=llm_model,
                max_tokens=150,  # Shorter responses for speed
                temperature=0.1  # Lower temperature for consistency
            )
            
            return {
                'answer': response,
                'confidence': 0.85,
                'model_used': llm_model or 'sarvam-m'
            }
            
        except Exception as e:
            logger.error(f"Fast answer generation failed: {e}")
            return {
                'answer': 'Processing error occurred during answer generation.',
                'confidence': 0.1,
                'model_used': 'error'
            }
    
    def _assess_performance(self, processing_time: float, gpu_tasks_completed: int) -> str:
        """Assess overall performance level"""
        if processing_time < 1.0:
            return "excellent"  # Sub-second response
        elif processing_time < self.target_response_time:
            return "good"
        elif processing_time < self.target_response_time * 2:
            return "acceptable"
        else:
            return "slow"
    
    async def _fallback_to_standard_processing(
        self, 
        query: str, 
        llm_model: Optional[str], 
        max_results: int, 
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to standard hybrid processing"""
        logger.info("Falling back to standard hybrid processing")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.hybrid_processor.process_intelligent_query,
            query,
            llm_model,
            kwargs.get('use_web_search', True),
            kwargs.get('max_web_results', 5),
            max_results
        )
        
        # Add GPU status to result
        result['gpu_acceleration'] = {
            'enabled': False,
            'tasks_completed': 0,
            'performance_level': 'fallback',
            'infrastructure_status': self.gpu_manager.get_infrastructure_status()
        }
        
        return result
    
    def get_acceleration_status(self) -> Dict[str, Any]:
        """Get current GPU acceleration status"""
        return {
            'gpu_infrastructure': self.gpu_manager.get_infrastructure_status(),
            'performance_targets': {
                'target_response_time': self.target_response_time,
                'gpu_threshold_time': self.gpu_threshold_time
            },
            'current_load': {
                'thread_pool_size': self.executor._max_workers,
                'active_threads': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0
            }
        }


# Global instance
_hybrid_gpu_processor = None

def get_hybrid_gpu_processor(hybrid_processor: HybridRAGProcessor) -> HybridGPUProcessor:
    """Get global hybrid GPU processor instance"""
    global _hybrid_gpu_processor
    if _hybrid_gpu_processor is None:
        _hybrid_gpu_processor = HybridGPUProcessor(hybrid_processor)
    return _hybrid_gpu_processor