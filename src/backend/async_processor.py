"""
Asynchronous processing system for file uploads and document ingestion
Enables non-blocking operations with progress tracking
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingTask:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AsyncProcessor:
    """
    Manages asynchronous processing of long-running tasks
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, ProcessingTask] = {}
        self.task_callbacks: Dict[str, List[Callable]] = {}
        self.running = True
        
        # Start background cleanup task
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_tasks, daemon=True)
        self._cleanup_thread.start()
    
    def submit_file_processing(self, 
                             files: List[Dict[str, Any]], 
                             processing_func: Callable,
                             **kwargs) -> str:
        """Submit file processing task"""
        task_id = str(uuid.uuid4())
        
        task = ProcessingTask(
            task_id=task_id,
            task_type="file_processing",
            metadata={
                'file_count': len(files),
                'file_names': [f.get('name', 'unknown') for f in files],
                'kwargs': kwargs
            }
        )
        
        self.tasks[task_id] = task
        
        # Submit to thread pool
        future = self.executor.submit(
            self._process_files_wrapper,
            task_id, files, processing_func, kwargs
        )
        
        # Store future for potential cancellation
        task.metadata['future'] = future
        
        logger.info(f"Submitted file processing task {task_id} with {len(files)} files")
        return task_id
    
    def submit_wikipedia_ingestion(self, 
                                 strategy: str, 
                                 count: int, 
                                 ingestion_func: Callable,
                                 **kwargs) -> str:
        """Submit Wikipedia ingestion task"""
        task_id = str(uuid.uuid4())
        
        task = ProcessingTask(
            task_id=task_id,
            task_type="wikipedia_ingestion",
            metadata={
                'strategy': strategy,
                'target_count': count,
                'kwargs': kwargs
            }
        )
        
        self.tasks[task_id] = task
        
        # Submit to thread pool
        future = self.executor.submit(
            self._process_wikipedia_wrapper,
            task_id, strategy, count, ingestion_func, kwargs
        )
        
        task.metadata['future'] = future
        
        logger.info(f"Submitted Wikipedia ingestion task {task_id} for {count} articles")
        return task_id
    
    def _process_files_wrapper(self, 
                             task_id: str, 
                             files: List[Dict[str, Any]], 
                             processing_func: Callable,
                             kwargs: Dict[str, Any]):
        """Wrapper for file processing with progress tracking"""
        task = self.tasks[task_id]
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.message = "Starting file processing..."
            
            processed_files = []
            total_files = len(files)
            
            for i, file_data in enumerate(files):
                if task.status == TaskStatus.CANCELLED:
                    break
                
                try:
                    # Update progress
                    task.progress = (i / total_files) * 100
                    task.message = f"Processing file {i+1}/{total_files}: {file_data.get('name', 'unknown')}"
                    
                    # Process individual file
                    result = processing_func(file_data, **kwargs)
                    processed_files.append({
                        'file': file_data.get('name', 'unknown'),
                        'result': result,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_data.get('name', 'unknown')}: {str(e)}")
                    processed_files.append({
                        'file': file_data.get('name', 'unknown'),
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Complete task
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                task.message = f"Completed processing {len(processed_files)} files"
                task.result = {
                    'processed_files': processed_files,
                    'successful_count': sum(1 for f in processed_files if f['status'] == 'success'),
                    'failed_count': sum(1 for f in processed_files if f['status'] == 'failed')
                }
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.message = f"Task failed: {str(e)}"
            logger.error(f"File processing task {task_id} failed: {str(e)}")
        
        finally:
            task.completed_at = time.time()
            self._notify_callbacks(task_id)
    
    def _process_wikipedia_wrapper(self, 
                                 task_id: str, 
                                 strategy: str, 
                                 count: int, 
                                 ingestion_func: Callable,
                                 kwargs: Dict[str, Any]):
        """Wrapper for Wikipedia ingestion with progress tracking"""
        task = self.tasks[task_id]
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.message = f"Starting Wikipedia ingestion ({strategy})..."
            
            # Create progress callback
            def progress_callback(current: int, total: int, message: str = ""):
                task.progress = (current / total) * 100 if total > 0 else 0
                task.message = message or f"Processing article {current}/{total}"
            
            # Execute ingestion with progress tracking
            result = ingestion_func(strategy, count, progress_callback=progress_callback, **kwargs)
            
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                task.message = f"Completed Wikipedia ingestion: {result.get('successful', 0)} articles"
                task.result = result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.message = f"Wikipedia ingestion failed: {str(e)}"
            logger.error(f"Wikipedia ingestion task {task_id} failed: {str(e)}")
        
        finally:
            task.completed_at = time.time()
            self._notify_callbacks(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get current status of a task"""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.message = "Task cancelled by user"
        
        # Try to cancel the future if possible
        future = task.metadata.get('future')
        if future:
            future.cancel()
        
        logger.info(f"Cancelled task {task_id}")
        return True
    
    def add_task_callback(self, task_id: str, callback: Callable[[ProcessingTask], None]):
        """Add callback to be notified when task completes"""
        if task_id not in self.task_callbacks:
            self.task_callbacks[task_id] = []
        self.task_callbacks[task_id].append(callback)
    
    def _notify_callbacks(self, task_id: str):
        """Notify all callbacks for a task"""
        if task_id in self.task_callbacks:
            task = self.tasks[task_id]
            for callback in self.task_callbacks[task_id]:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"Error in task callback: {str(e)}")
    
    def get_all_tasks(self, limit: int = 50) -> List[ProcessingTask]:
        """Get all tasks, most recent first"""
        all_tasks = list(self.tasks.values())
        all_tasks.sort(key=lambda t: t.created_at, reverse=True)
        return all_tasks[:limit]
    
    def get_active_tasks(self) -> List[ProcessingTask]:
        """Get currently running tasks"""
        return [task for task in self.tasks.values() 
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]
    
    def _cleanup_old_tasks(self):
        """Background thread to clean up old completed tasks"""
        while self.running:
            try:
                current_time = time.time()
                cleanup_threshold = 24 * 3600  # 24 hours
                
                tasks_to_remove = []
                for task_id, task in self.tasks.items():
                    if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and 
                        task.completed_at and 
                        current_time - task.completed_at > cleanup_threshold):
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                    if task_id in self.task_callbacks:
                        del self.task_callbacks[task_id]
                
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
                
            except Exception as e:
                logger.error(f"Error in task cleanup: {str(e)}")
            
            # Sleep for 1 hour before next cleanup
            time.sleep(3600)
    
    def shutdown(self):
        """Shutdown the processor and cleanup resources"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("AsyncProcessor shutdown complete")

# Global processor instance
_global_processor = None

def get_async_processor() -> AsyncProcessor:
    """Get global async processor instance"""
    global _global_processor
    if _global_processor is None:
        _global_processor = AsyncProcessor(max_workers=4)
    return _global_processor

def shutdown_processor():
    """Shutdown global processor"""
    global _global_processor
    if _global_processor:
        _global_processor.shutdown()
        _global_processor = None