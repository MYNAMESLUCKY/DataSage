import os
import logging
import time
from typing import List, Dict, Any, Optional
import uuid

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

# LangChain
from langchain.schema import Document

from backend.utils import setup_logging, performance_monitor

logger = setup_logging(__name__)

class ChromaVectorStoreManager:
    """
    ChromaDB-based vector store manager for document storage and similarity search
    """
    
    def __init__(self, config=None):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.documents = []
        self.is_initialized = False
        
        # Configuration
        self.collection_name = config.chroma_collection_name if config else "rag_documents"
        self.persist_directory = config.chroma_persist_directory if config else "./chroma_db"
        self.embedding_model_name = config.embedding_model_name if config else "all-MiniLM-L6-v2"
        self.batch_size = config.batch_size if config else 100
        
    def initialize(self, embedding_model_name: Optional[str] = None):
        """Initialize ChromaDB client and collection"""
        try:
            model_name = embedding_model_name or self.embedding_model_name
            logger.info(f"Initializing ChromaDB vector store with model: {model_name}")
            
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding function - use default embeddings for now
            try:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            except Exception as e:
                logger.warning(f"SentenceTransformer not available, using default embeddings: {e}")
                # Use default embedding function as fallback
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection '{self.collection_name}'")
                
                # Load existing documents metadata
                self._load_existing_documents()
                
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            
            self.is_initialized = True
            logger.info("ChromaDB vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector store: {str(e)}")
            self.is_initialized = False
            raise
    
    def _load_existing_documents(self):
        """Load existing documents from ChromaDB collection"""
        try:
            if self.collection is None:
                self.documents = []
                return
                
            # Get all documents from collection
            result = self.collection.get(include=['documents', 'metadatas'])
            
            if result.get('documents'):
                self.documents = []
                metadatas = result.get('metadatas', [])
                for i, doc_text in enumerate(result['documents']):
                    metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                    document = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    self.documents.append(document)
                
                logger.info(f"Loaded {len(self.documents)} existing documents from ChromaDB")
            else:
                self.documents = []
            
        except Exception as e:
            logger.error(f"Failed to load existing documents: {str(e)}")
            self.documents = []
    
    @performance_monitor
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the ChromaDB collection"""
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            logger.info(f"Adding {len(documents)} documents to ChromaDB")
            
            # Prepare data for ChromaDB
            doc_texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                doc_texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                # Generate unique ID for each document
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
            
            # Add documents to collection in batches
            for i in range(0, len(documents), self.batch_size):
                batch_end = min(i + self.batch_size, len(documents))
                
                self.collection.add(
                    documents=doc_texts[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
            
            # Update local documents list
            self.documents.extend(documents)
            
            logger.info(f"Successfully added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            return False
    
    @performance_monitor
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.0,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search using ChromaDB
        """
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return []
        
        try:
            logger.info(f"Performing similarity search for query: {query}...")
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to Document objects
            documents = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # ChromaDB returns distance (lower is better)
                    # Convert to similarity score (higher is better)
                    similarity_score = 1.0 - distance
                    
                    # Apply threshold filter
                    if similarity_score >= threshold:
                        # Add similarity score to metadata
                        if metadata is None:
                            metadata = {}
                        if isinstance(metadata, dict):
                            metadata['similarity_score'] = similarity_score
                        
                        document = Document(
                            page_content=doc_text,
                            metadata=metadata
                        )
                        documents.append(document)
            
            logger.info(f"Found {len(documents)} relevant documents above threshold {threshold}")
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def delete_documents(self, filter_metadata: Dict) -> bool:
        """Delete documents based on metadata filter"""
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Get documents to delete
            results = self.collection.get(
                where=filter_metadata,
                include=['documents']  # Changed from 'ids' to 'documents'
            )
            
            if results['ids']:
                # Delete documents
                self.collection.delete(ids=results['ids'])
                
                # Update local documents list
                self.documents = [
                    doc for doc in self.documents 
                    if not all(
                        doc.metadata.get(key) == value 
                        for key, value in filter_metadata.items()
                    )
                ]
                
                logger.info(f"Deleted {len(results['ids'])} documents")
                return True
            else:
                logger.info("No documents found matching filter criteria")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update a specific document"""
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[document.page_content],
                metadatas=[document.metadata]
            )
            
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if not self.is_initialized:
            return {"error": "Vector store not initialized"}
        
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory,
                "is_initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all documents)"""
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Clear local documents
            self.documents = []
            
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """Create a backup of the collection"""
        # ChromaDB handles persistence automatically
        # This method could implement additional backup logic if needed
        logger.info(f"ChromaDB data is automatically persisted to {self.persist_directory}")
        return True
    
    def close(self):
        """Close the ChromaDB client"""
        if self.client:
            # ChromaDB automatically persists data
            logger.info("ChromaDB client closed")
            self.client = None
            self.collection = None
            self.is_initialized = False