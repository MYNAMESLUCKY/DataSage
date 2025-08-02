import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_system"
    username: str = "user"
    password: str = ""
    pool_size: int = 10
    
@dataclass
class AIModelConfig:
    """AI model configuration settings"""
    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1000
    
    # HuggingFace settings
    huggingface_api_key: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", ""))
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Model fallback settings
    enable_fallback: bool = True
    fallback_models: list = field(default_factory=lambda: ["gpt-3.5-turbo", "huggingface"])

@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_concurrent_processes: int = 4
    timeout_seconds: int = 30
    max_document_size_mb: int = 10
    
    # Text cleaning settings
    remove_html: bool = True
    normalize_whitespace: bool = True
    min_chunk_length: int = 50

@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    index_type: str = "chromadb"
    similarity_metric: str = "cosine"
    index_path: str = "vector_store_index"
    cache_embeddings: bool = True
    batch_size: int = 100
    
    # ChromaDB specific settings
    chroma_collection_name: str = "rag_documents"
    chroma_persist_directory: str = "./chroma_db"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # FAISS specific settings (legacy)
    faiss_index_type: str = "IndexFlatIP"  # Inner Product for cosine similarity
    enable_gpu: bool = False

@dataclass
class CacheConfig:
    """Caching configuration"""
    enable_query_cache: bool = True
    query_cache_ttl: int = 300  # 5 minutes
    enable_embedding_cache: bool = True
    embedding_cache_ttl: int = 3600  # 1 hour
    max_cache_size_mb: int = 100

@dataclass
class SecurityConfig:
    """Security configuration"""
    rate_limit_requests_per_minute: int = 60
    max_query_length: int = 1000
    allowed_domains: list = field(default_factory=list)
    blocked_domains: list = field(default_factory=list)
    enable_content_filtering: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_file: str = "rag_system.log"
    max_log_size_mb: int = 10
    backup_count: int = 5
    enable_query_logging: bool = True
    enable_performance_logging: bool = True

class Settings:
    """
    Main settings class that consolidates all configuration
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        
        # Initialize configuration objects
        self.database = DatabaseConfig()
        self.ai_models = AIModelConfig()
        self.processing = ProcessingConfig()
        self.vector_store = VectorStoreConfig()
        self.cache = CacheConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        
        # Load configuration from file if it exists
        self.load_config()
        
        # Override with environment variables
        self._load_from_environment()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration objects
                self._update_config_from_dict(config_data)
                
                print(f"Configuration loaded from {self.config_file}")
            else:
                print(f"Configuration file {self.config_file} not found. Using defaults.")
        except Exception as e:
            print(f"Error loading configuration: {e}. Using defaults.")
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            config_data = self._config_to_dict()
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration objects from dictionary"""
        if 'database' in config_data:
            self._update_dataclass(self.database, config_data['database'])
        
        if 'ai_models' in config_data:
            self._update_dataclass(self.ai_models, config_data['ai_models'])
        
        if 'processing' in config_data:
            self._update_dataclass(self.processing, config_data['processing'])
        
        if 'vector_store' in config_data:
            self._update_dataclass(self.vector_store, config_data['vector_store'])
        
        if 'cache' in config_data:
            self._update_dataclass(self.cache, config_data['cache'])
        
        if 'security' in config_data:
            self._update_dataclass(self.security, config_data['security'])
        
        if 'logging' in config_data:
            self._update_dataclass(self.logging, config_data['logging'])
    
    def _update_dataclass(self, dataclass_obj, data: Dict[str, Any]):
        """Update dataclass object with dictionary data"""
        for key, value in data.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration objects to dictionary"""
        return {
            'database': self._dataclass_to_dict(self.database),
            'ai_models': self._dataclass_to_dict(self.ai_models),
            'processing': self._dataclass_to_dict(self.processing),
            'vector_store': self._dataclass_to_dict(self.vector_store),
            'cache': self._dataclass_to_dict(self.cache),
            'security': self._dataclass_to_dict(self.security),
            'logging': self._dataclass_to_dict(self.logging)
        }
    
    def _dataclass_to_dict(self, dataclass_obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        result = {}
        for field_name in dataclass_obj.__dataclass_fields__:
            value = getattr(dataclass_obj, field_name)
            result[field_name] = value
        return result
    
    def _load_from_environment(self):
        """Load settings from environment variables"""
        # OpenAI API Key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.ai_models.openai_api_key = openai_key
        
        # HuggingFace API Key
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            self.ai_models.huggingface_api_key = hf_key
        
        # Log level
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            self.logging.log_level = log_level.upper()
        
        # Environment-specific overrides
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            self._apply_production_settings()
        elif env == "development":
            self._apply_development_settings()
    
    def _apply_production_settings(self):
        """Apply production-specific settings"""
        self.logging.log_level = "WARNING"
        self.cache.enable_query_cache = True
        self.security.rate_limit_requests_per_minute = 100
        self.processing.max_concurrent_processes = 8
    
    def _apply_development_settings(self):
        """Apply development-specific settings"""
        self.logging.log_level = "DEBUG"
        self.cache.enable_query_cache = False
        self.security.rate_limit_requests_per_minute = 1000
        self.processing.max_concurrent_processes = 2
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-specific configuration"""
        return {
            'api_key': self.ai_models.openai_api_key,
            'model': self.ai_models.openai_model,
            'temperature': self.ai_models.openai_temperature,
            'max_tokens': self.ai_models.openai_max_tokens
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            'model_name': self.ai_models.embedding_model,
            'cache_embeddings': self.vector_store.cache_embeddings,
            'batch_size': self.vector_store.batch_size
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return {
            'chunk_size': self.processing.chunk_size,
            'chunk_overlap': self.processing.chunk_overlap,
            'max_concurrent_processes': self.processing.max_concurrent_processes,
            'timeout_seconds': self.processing.timeout_seconds,
            'max_document_size_mb': self.processing.max_document_size_mb
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate OpenAI configuration
        if not self.ai_models.openai_api_key:
            validation_results['warnings'].append("OpenAI API key not configured")
        
        # Validate processing settings
        if self.processing.chunk_overlap >= self.processing.chunk_size:
            validation_results['errors'].append("Chunk overlap must be less than chunk size")
            validation_results['valid'] = False
        
        if self.processing.chunk_size < 50:
            validation_results['warnings'].append("Very small chunk size may affect quality")
        
        # Validate vector store settings
        if not os.path.exists(os.path.dirname(self.vector_store.index_path)):
            try:
                os.makedirs(os.path.dirname(self.vector_store.index_path), exist_ok=True)
            except Exception as e:
                validation_results['errors'].append(f"Cannot create index directory: {e}")
                validation_results['valid'] = False
        
        # Validate cache settings
        if self.cache.max_cache_size_mb < 10:
            validation_results['warnings'].append("Very low cache size may affect performance")
        
        return validation_results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'ai_models': {
                'openai_configured': bool(self.ai_models.openai_api_key),
                'openai_model': self.ai_models.openai_model,
                'embedding_model': self.ai_models.embedding_model,
                'fallback_enabled': self.ai_models.enable_fallback
            },
            'processing': {
                'chunk_size': self.processing.chunk_size,
                'chunk_overlap': self.processing.chunk_overlap,
                'max_processes': self.processing.max_concurrent_processes
            },
            'vector_store': {
                'index_type': self.vector_store.index_type,
                'cache_enabled': self.vector_store.cache_embeddings,
                'batch_size': self.vector_store.batch_size
            },
            'cache': {
                'query_cache': self.cache.enable_query_cache,
                'embedding_cache': self.cache.enable_embedding_cache,
                'max_size_mb': self.cache.max_cache_size_mb
            }
        }

# Global settings instance
settings = Settings()
