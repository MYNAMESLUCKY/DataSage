from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import time

class DataSourceType(Enum):
    """Types of data sources supported"""
    WEB = "web"
    API = "api"
    FILE = "file"
    DATABASE = "database"

class FileType(Enum):
    """Supported file types for document processing"""
    TEXT = "text"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    DOCX = "docx"

class ProcessingStatus(Enum):
    """Status of data processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DataSource:
    """Data source configuration"""
    url: str
    source_type: str
    name: Optional[str] = None
    file_type: Optional[str] = None
    file_path: Optional[str] = None
    file_content: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.name is None:
            if self.file_path:
                self.name = f"File: {Path(self.file_path).name}"
            else:
                self.name = f"Source from {self.url}"

@dataclass
class ProcessingResult:
    """Result of data processing operation"""
    source_url: str
    status: ProcessingStatus
    documents_count: int = 0
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class QueryResult:
    """Result of a query operation"""
    query: str
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    relevant_docs_count: int = 0
    response_time: float = 0.0
    model_used: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    text: str
    embedding: List[float]
    model_name: str
    dimensions: int
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    content: str
    source_url: str
    chunk_id: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class SystemHealth:
    """System health status"""
    is_healthy: bool
    api_status: str
    vector_store_status: str
    embedding_model_status: str
    llm_status: str
    memory_usage: Dict[str, float] = field(default_factory=dict)
    uptime: float = 0.0
    last_check: float = field(default_factory=time.time)

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    llm_model: str = "gpt-4o"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_results: int = 5
    similarity_threshold: float = 0.7
    temperature: float = 0.3
    max_tokens: int = 1000

@dataclass
class UserSession:
    """User session data"""
    session_id: str
    query_history: List[QueryResult] = field(default_factory=list)
    data_sources: List[DataSource] = field(default_factory=list)
    processing_status: Dict[str, ProcessingStatus] = field(default_factory=dict)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass

class DataIngestionError(Exception):
    """Raised when data ingestion fails"""
    pass

class VectorStoreError(Exception):
    """Raised when vector store operations fail"""
    pass

class ModelError(Exception):
    """Raised when AI model operations fail"""
    pass

class QueryProcessingError(Exception):
    """Raised when query processing fails"""
    pass

def validate_data_source(source: DataSource) -> bool:
    """Validate a data source configuration"""
    if not source.url:
        raise ConfigValidationError("Data source URL is required")
    
    if not source.source_type:
        raise ConfigValidationError("Data source type is required")
    
    # Add more validation rules as needed
    return True

def validate_model_config(config: ModelConfig) -> bool:
    """Validate model configuration"""
    if config.chunk_size <= 0:
        raise ConfigValidationError("Chunk size must be positive")
    
    if config.chunk_overlap < 0:
        raise ConfigValidationError("Chunk overlap cannot be negative")
    
    if config.chunk_overlap >= config.chunk_size:
        raise ConfigValidationError("Chunk overlap must be less than chunk size")
    
    if config.similarity_threshold < 0 or config.similarity_threshold > 1:
        raise ConfigValidationError("Similarity threshold must be between 0 and 1")
    
    return True

def create_default_session(session_id: str) -> UserSession:
    """Create a default user session"""
    return UserSession(
        session_id=session_id,
        model_config=ModelConfig()
    )
