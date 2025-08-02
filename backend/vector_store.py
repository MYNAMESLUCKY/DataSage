import os
import logging
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Vector store and embeddings
import faiss
import sklearn.feature_extraction.text as sklearn_text
from sklearn.metrics.pairwise import cosine_similarity

# LangChain
from langchain.schema import Document

from .utils import setup_logging, performance_monitor

logger = setup_logging(__name__)

class VectorStoreManager:
    """
    Manager for vector storage and similarity search using FAISS
    """
    
    def __init__(self):
        self.vector_store = None
        self.vectorizer = None
        self.document_vectors = None
        self.current_embedding_model_name = None
        self.documents = []
        self.is_initialized = False
        self.index_path = "vector_store_index"
        
    def initialize(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store with specified embedding model"""
        try:
            logger.info(f"Initializing vector store with model: {embedding_model_name}")
            
            # Initialize embedding model
            self._load_embedding_model(embedding_model_name)
            
            # Try to load existing index
            if os.path.exists(f"{self.index_path}.faiss"):
                self._load_existing_index()
            
            self.is_initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            self.is_initialized = False
    
    def _load_embedding_model(self, model_name: str):
        """Load the specified embedding model"""
        try:
            # Use TF-IDF vectorizer as a simple embedding alternative
            self.vectorizer = sklearn_text.TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.current_embedding_model_name = model_name
            logger.info(f"Loaded TF-IDF vectorizer as embedding model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            raise
    
    def _load_existing_index(self):
        """Load existing index if available"""
        try:
            if os.path.exists(f"{self.index_path}_docs.pkl"):
                # Load documents metadata
                with open(f"{self.index_path}_docs.pkl", "rb") as f:
                    self.documents = pickle.load(f)
                
                # Load vectorizer if exists
                if os.path.exists(f"{self.index_path}_vectorizer.pkl"):
                    with open(f"{self.index_path}_vectorizer.pkl", "rb") as f:
                        self.vectorizer = pickle.load(f)
                
                # Load document vectors if exists
                if os.path.exists(f"{self.index_path}_vectors.pkl"):
                    with open(f"{self.index_path}_vectors.pkl", "rb") as f:
                        self.document_vectors = pickle.load(f)
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
                
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
            self.documents = []
            self.vectorizer = None
            self.document_vectors = None
    
    @performance_monitor
    def add_documents(
        self, 
        documents: List[Document], 
        embedding_model: str = None
    ):
        """Add documents to the vector store"""
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Change embedding model if specified
            if embedding_model and embedding_model != self.current_embedding_model_name:
                self._load_embedding_model(embedding_model)
                # If model changed, recreate vector store
                self.document_vectors = None
                self.documents = []
            
            # Ensure we have valid documents
            valid_documents = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No valid documents to add")
                return
            
            # Create or update vector store
            if self.document_vectors is None:
                # Create new vectors
                document_texts = [doc.page_content for doc in valid_documents]
                self.document_vectors = self.vectorizer.fit_transform(document_texts)
                self.documents = valid_documents.copy()
            else:
                # Add to existing vectors
                document_texts = [doc.page_content for doc in valid_documents]
                new_vectors = self.vectorizer.transform(document_texts)
                
                # Combine with existing vectors
                import scipy.sparse
                self.document_vectors = scipy.sparse.vstack([self.document_vectors, new_vectors])
                self.documents.extend(valid_documents)
            
            # Save the updated index
            self._save_index()
            
            logger.info(f"Successfully added {len(valid_documents)} documents. Total: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def _save_index(self):
        """Save the current vector store index"""
        try:
            if self.document_vectors is not None:
                # Save documents metadata
                with open(f"{self.index_path}_docs.pkl", "wb") as f:
                    pickle.dump(self.documents, f)
                
                # Save vectorizer
                with open(f"{self.index_path}_vectorizer.pkl", "wb") as f:
                    pickle.dump(self.vectorizer, f)
                
                # Save document vectors
                with open(f"{self.index_path}_vectors.pkl", "wb") as f:
                    pickle.dump(self.document_vectors, f)
                
                logger.info("Vector store index saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    @performance_monitor
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.7
    ) -> List[Document]:
        """
        Perform similarity search for the given query
        """
        try:
            if self.document_vectors is None or self.vectorizer is None:
                logger.warning("No vector store available for similarity search")
                return []
            
            logger.info(f"Performing similarity search for query: {query[:50]}...")
            
            # Transform query to vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top k similar documents above threshold
            similar_indices = np.argsort(similarities)[::-1]  # Sort descending
            
            filtered_docs = []
            for idx in similar_indices:
                similarity = similarities[idx]
                
                if similarity >= threshold and len(filtered_docs) < k:
                    doc = self.documents[idx]
                    # Create a copy to avoid modifying original
                    doc_copy = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                    doc_copy.metadata['similarity_score'] = float(similarity)
                    filtered_docs.append(doc_copy)
            
            logger.info(f"Found {len(filtered_docs)} relevant documents above threshold {threshold}")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def get_similar_documents(
        self, 
        document: Document, 
        k: int = 3
    ) -> List[Document]:
        """Find documents similar to the given document"""
        try:
            if self.document_vectors is None:
                return []
            
            query = document.page_content[:500]  # Use first 500 chars as query
            return self.similarity_search(query, k=k)
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID"""
        try:
            for doc in self.documents:
                if doc.metadata.get('chunk_id') == doc_id:
                    return doc
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def get_all_sources(self) -> List[str]:
        """Get all unique sources in the vector store"""
        try:
            sources = set()
            for doc in self.documents:
                source = doc.metadata.get('source')
                if source:
                    sources.add(source)
            return list(sources)
            
        except Exception as e:
            logger.error(f"Error getting sources: {str(e)}")
            return []
    
    def get_documents_by_source(self, source: str) -> List[Document]:
        """Get all documents from a specific source"""
        try:
            return [doc for doc in self.documents if doc.metadata.get('source') == source]
            
        except Exception as e:
            logger.error(f"Error getting documents by source: {str(e)}")
            return []
    
    def get_size(self) -> int:
        """Get the number of documents in the vector store"""
        return len(self.documents)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            stats = {
                'total_documents': len(self.documents),
                'embedding_model': self.current_embedding_model_name,
                'unique_sources': len(self.get_all_sources()),
                'index_size_mb': 0
            }
            
            # Calculate index size
            if os.path.exists(f"{self.index_path}.faiss"):
                stats['index_size_mb'] = os.path.getsize(f"{self.index_path}.faiss") / (1024 * 1024)
            
            # Document length statistics
            if self.documents:
                lengths = [len(doc.page_content) for doc in self.documents]
                stats.update({
                    'avg_document_length': np.mean(lengths),
                    'min_document_length': np.min(lengths),
                    'max_document_length': np.max(lengths)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def reset(self):
        """Reset the vector store, removing all data"""
        try:
            logger.info("Resetting vector store...")
            
            # Clear in-memory data
            self.document_vectors = None
            self.vectorizer = None
            self.documents = []
            
            # Remove index files
            index_files = [
                f"{self.index_path}_docs.pkl",
                f"{self.index_path}_vectorizer.pkl",
                f"{self.index_path}_vectors.pkl"
            ]
            
            for file_path in index_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")
            
            logger.info("Vector store reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
    
    def optimize_index(self):
        """Optimize the FAISS index for better performance"""
        try:
            if self.document_vectors is None:
                return
            
            logger.info("Optimizing vector store index...")
            
            # For now, just save the index
            self._save_index()
            
            logger.info("Index optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
