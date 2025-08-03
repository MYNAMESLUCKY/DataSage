import os
import logging
import time
import functools
import hashlib
import re
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse
import json

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

def cache_result(ttl: int = 300):
    """Simple in-memory cache decorator with TTL"""
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(str((args, kwargs)).encode()).hexdigest()
            
            # Check if cached result exists and is valid
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
                else:
                    del cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

def normalize_url(url: str) -> str:
    """Normalize URL by adding protocol if missing"""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL"""
    try:
        # Normalize URL first
        url = normalize_url(url)
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes = int(size_bytes / 1024)
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard index"""
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    if not text:
        return []
    
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    words = [word for word in words if word not in stop_words]
    
    # Count word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

def create_safe_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a safe dictionary for JSON serialization"""
    safe_data = {}
    
    for key, value in data.items():
        try:
            json.dumps(value)  # Test if value is JSON serializable
            safe_data[key] = value
        except (TypeError, ValueError):
            safe_data[key] = str(value)  # Convert to string if not serializable
    
    return safe_data

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    
            if last_exception:
                raise last_exception
            else:
                raise Exception("Function failed after retries")
        
        return wrapper
    return decorator

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    except ImportError:
        return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0, 'available_mb': 0.0}

def format_timestamp(timestamp: float) -> str:
    """Format timestamp to human-readable string"""
    try:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    except (ValueError, OSError):
        return str(timestamp)

class ConfigManager:
    """Simple configuration manager"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load config file: {e}")
        
        return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save config file: {e}")
