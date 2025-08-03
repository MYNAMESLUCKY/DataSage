"""
Advanced Embeddings Manager for Enhanced RAG Performance
========================================================

This module provides better embedding strategies and model management
to improve document similarity search and retrieval quality.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import pickle
import os
from dataclasses import dataclass

try:
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .utils import setup_logging

logger = setup_logging(__name__)

@dataclass 
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    use_hybrid_search: bool = True

class AdvancedEmbeddingsManager:
    """Manager for advanced embedding techniques"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.embedding_model = None
        self.cache_dir = "embeddings_cache"
        self.embedding_cache = {}
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load embedding cache
        self._load_embedding_cache()
    
    def initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            # Try to use sentence-transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.config.model_name)
                logger.info(f"Initialized SentenceTransformer model: {self.config.model_name}")
                return True
            except ImportError:
                logger.warning("SentenceTransformers not available, using default embeddings")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            return False
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts with caching"""
        if not texts:
            return np.array([])
        
        # Check cache first
        cached_embeddings = []
        texts_to_embed = []
        cache_keys = []
        
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_keys.append(cache_key)
            
            if cache_key in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[cache_key])
            else:
                cached_embeddings.append(None)
                texts_to_embed.append(text)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            if self.embedding_model:
                new_embeddings = self._generate_embeddings(texts_to_embed)
            else:
                new_embeddings = self._generate_fallback_embeddings(texts_to_embed)
            
            # Update cache
            embedding_idx = 0
            for i, cached_emb in enumerate(cached_embeddings):
                if cached_emb is None:
                    embedding = new_embeddings[embedding_idx]
                    cached_embeddings[i] = embedding
                    self.embedding_cache[cache_keys[i]] = embedding
                    embedding_idx += 1
        
        # Save cache periodically
        if len(self.embedding_cache) % 100 == 0:
            self._save_embedding_cache()
        
        return np.array(cached_embeddings)
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the loaded model"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return self._generate_fallback_embeddings(texts)
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate simple fallback embeddings using TF-IDF approach"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using random embeddings")
            return np.random.normal(0, 0.1, (len(texts), 384))
        
        try:
            # Use TF-IDF + SVD for dimensionality reduction
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Reduce dimensions to typical embedding size
            svd = TruncatedSVD(n_components=384, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix)
            
            # Normalize
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Fallback embedding generation failed: {str(e)}")
            # Return random embeddings as last resort
            return np.random.normal(0, 0.1, (len(texts), 384))
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a single query with query-specific optimization"""
        # Apply query preprocessing
        processed_query = self._preprocess_query(query)
        
        # Get base embedding
        embedding = self.get_embeddings([processed_query])[0]
        
        # Apply query-specific transformations if needed
        return self._optimize_query_embedding(embedding, query)
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better embedding"""
        # Expand common abbreviations
        expansions = {
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "dl": "deep learning",
            "nlp": "natural language processing"
        }
        
        query_lower = query.lower()
        for abbrev, full_form in expansions.items():
            if abbrev in query_lower:
                query = query.replace(abbrev, full_form)
        
        return query
    
    def _optimize_query_embedding(self, embedding: np.ndarray, original_query: str) -> np.ndarray:
        """Apply query-specific optimizations to embedding"""
        # For basic questions, emphasize certain dimensions
        if any(pattern in original_query.lower() for pattern in ['what is', 'define', 'explain']):
            # Boost dimensions that typically capture definitional information
            # This is a simplified approach; in practice, you'd train this
            definition_boost = np.ones_like(embedding)
            definition_boost[:100] *= 1.1  # Boost first 100 dimensions
            embedding = embedding * definition_boost
        
        # Renormalize
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def compute_similarity_scores(
        self, 
        query_embedding: np.ndarray, 
        doc_embeddings: np.ndarray,
        method: str = "cosine"
    ) -> np.ndarray:
        """Compute similarity scores between query and document embeddings"""
        if len(doc_embeddings) == 0:
            return np.array([])
        
        if method == "cosine":
            # Cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding)
            return similarities
        elif method == "euclidean":
            # Negative euclidean distance (higher is better)
            distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
            return -distances
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use first 200 characters + hash for key
        text_snippet = text[:200]
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{text_hash}_{len(text)}"
    
    def _load_embedding_cache(self):
        """Load embedding cache from disk"""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {str(e)}")
            self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug("Saved embedding cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache = {}
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logger.info("Cleared embedding cache")

# Global instance
_embeddings_manager = None

def get_embeddings_manager() -> AdvancedEmbeddingsManager:
    """Get global embeddings manager instance"""
    global _embeddings_manager
    if _embeddings_manager is None:
        _embeddings_manager = AdvancedEmbeddingsManager()
        _embeddings_manager.initialize_embeddings()
    return _embeddings_manager