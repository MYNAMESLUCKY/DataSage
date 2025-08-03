# Overview

This is an Enterprise RAG (Retrieval-Augmented Generation) System built with Streamlit that enables users to ingest data from various online sources, Wikipedia articles, and local documents. The system provides comprehensive knowledge access through intelligent querying with a professional web interface.

The application allows users to:
- Add and configure data sources (web URLs, APIs, files)
- Upload and process local documents (text, PDF, Excel, CSV, Word)
- **NEW: Ingest Wikipedia articles at scale** with smart sampling strategies
- Process and chunk documents for vector storage with ChromaDB
- Query the processed data using OpenRouter's Kimi model or other AI providers
- Monitor processing status and view query history
- **Enhanced: Wikipedia integration** with balanced, category-focused, and random sampling modes

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **UI Pattern**: Multi-step workflow with session state management
- **Components**: Modular UI components for reusability (status badges, metric cards, data source management)
- **Styling**: Custom CSS for professional appearance with gradients and responsive design
- **Visualization**: Plotly integration for metrics and analytics

## Backend Architecture
- **Pattern**: Service-oriented architecture with clear separation of concerns
- **Core Services**:
  - `RAGEngine`: Orchestrates AI model interactions and query processing
  - `DataIngestionService`: Handles web scraping and document processing
  - `VectorStoreManager`: Manages ChromaDB vector storage and similarity search
  - `WikipediaIngestionService`: **NEW** - Specialized service for Wikipedia article ingestion with rate limiting
  - `RAGSystemAPI`: Main API layer that coordinates all services
- **Concurrency**: ThreadPoolExecutor for parallel processing with Wikipedia rate limiting
- **Error Handling**: Comprehensive logging and graceful failure handling
- **Wikipedia Features**: Smart article selection, category-based filtering, random sampling, and content chunking

## Data Processing Pipeline
- **Text Extraction**: Uses `trafilatura` and `BeautifulSoup` for web content extraction
- **Document Chunking**: LangChain's `RecursiveCharacterTextSplitter` for intelligent text segmentation
- **Embedding Generation**: Sentence Transformers for creating vector representations
- **Storage**: FAISS for efficient similarity search and retrieval

## AI Model Integration
- **Primary**: OpenRouter API with Kimi model (moonshotai/kimi-k2:free) for cost-effective processing
- **Secondary**: DeepSeek API for fast text generation (deepseek-chat, deepseek-coder) as fallback
- **Tertiary**: Direct OpenAI API support as final backup option
- **Embeddings**: ChromaDB's built-in embedding models for vector representation
- **Prompt Engineering**: ChatPromptTemplate for structured AI interactions
- **Model Selection**: Configurable models including moonshotai/kimi-k2:free, deepseek-chat, deepseek-coder, openai/gpt-4o, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-8b-instruct
- **Wikipedia Integration**: Specialized Wikipedia ingestion service with rate limiting and smart content filtering

## Configuration Management
- **Settings**: Centralized configuration with dataclasses for type safety
- **Environment Variables**: API keys and sensitive data loaded from environment
- **Model Selection**: Configurable AI models with fallback mechanisms

# External Dependencies

## AI/ML Services
- **OpenRouter API**: Primary AI service supporting multiple model providers (requires API key)
- **OpenAI API**: Direct API support as fallback option (optional API key)
- **ChromaDB**: Vector database for embeddings and similarity search

## Document Processing Libraries
- **PyPDF2**: PDF text extraction and processing
- **pandas**: CSV and Excel data manipulation and analysis
- **openpyxl**: Modern Excel (.xlsx) file processing
- **xlrd**: Legacy Excel (.xls) file support
- **python-docx**: Word document (.docx) text extraction

## Vector Storage
- **ChromaDB**: Advanced vector database with built-in embeddings and persistent storage
- **Default Embeddings**: ChromaDB's built-in embedding models for text vectorization
- **Cosine Similarity**: High-performance similarity search with ChromaDB's HNSW indexing
- **Persistent Storage**: Automatic data persistence with ChromaDB's built-in storage layer

## Web Scraping & Content Processing
- **Trafilatura**: Web content extraction and cleaning
- **BeautifulSoup**: HTML parsing and content extraction
- **Requests**: HTTP client for web data retrieval
- **LangChain**: Document processing, text splitting, and RAG orchestration

## UI/Visualization
- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations for metrics

## Development & Utilities
- **Logging**: Python standard logging for system monitoring
- **Concurrent Processing**: ThreadPoolExecutor for parallel operations
- **Data Models**: Dataclasses and Enums for type-safe data structures