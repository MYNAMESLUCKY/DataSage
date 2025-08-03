# Overview

This is an Enterprise RAG (Retrieval-Augmented Generation) System built with Streamlit that enables users to ingest data from various online sources, Wikipedia articles, and local documents. The system provides comprehensive knowledge access through intelligent querying with a professional web interface.

The application allows users to:
- Add and configure data sources (web URLs, APIs, files)
- Upload and process local documents (text, PDF, Excel, CSV, Word)
- **NEW: Ingest Wikipedia articles at scale** with smart sampling strategies
- **NEW: Real-time web search integration** using Tavily API for live data retrieval
- Process and chunk documents for vector storage with ChromaDB
- Query the processed data using SARVAM API or other AI providers with web-enhanced context
- Monitor processing status and view query history
- **Enhanced: Wikipedia integration** with balanced, category-focused, and random sampling modes
- **Enhanced: Hybrid RAG system** combining local knowledge base with real-time web search

# User Preferences

Preferred communication style: Simple, everyday language.

# Enhanced Web Search & Database Caching (Updated 2025-08-03)

User has requested comprehensive real-time web search integration with database caching for any question:

## Intelligent Hybrid RAG Features:
- **Smart Knowledge Comparison**: Checks existing documents first, then compares with live web data
- **Automatic Knowledge Base Updates**: Adds new web information to vector store when gaps are detected
- **Intelligent Processing Strategies**: 
  - Hybrid comparison when both KB and web data exist
  - Knowledge base update when KB lacks information
  - Fallback to KB-only when web search fails
- **PostgreSQL database caching** for search results and content optimization
- **Real-time decision making** for optimal answer generation

## 8-Point RAG Improvement Plan - FULLY IMPLEMENTED:
1. ✅ **Document Chunking Strategy** - Intelligent semantic chunking with content type detection
2. ✅ **Aggressive Source Filtering** - Quality-based document filtering and relevance scoring  
3. ✅ **Better Embeddings** - Advanced embedding manager with caching and optimization
4. ✅ **Enhanced Retrieval Logic** - Multi-strategy search with query expansion
5. ✅ **Result Reranking** - Intelligent document ranking and relevance scoring
6. ✅ **Metadata Filtering** - Content type and authority-based filtering
7. ✅ **Retrieval Pipeline Audit** - Comprehensive monitoring and analytics
8. ✅ **Search API Fallback** - Wikipedia/DuckDuckGo fallback for missing knowledge

## Training System Features:
- Continuous performance monitoring and improvement recommendations
- User feedback integration for quality enhancement
- Query analysis and optimization suggestions
- Comprehensive training insights and metrics export

# System Architecture (Updated 2025-08-03)

## Clean Project Structure
```
├── app.py                    # Main Streamlit application entry point
├── src/                      # Organized source code directory
│   ├── backend/             # Core RAG system logic
│   │   ├── api.py          # Main API orchestrator
│   │   ├── rag_engine.py   # AI model interactions
│   │   ├── vector_store_chroma.py # ChromaDB integration
│   │   ├── data_ingestion.py # Document processing
│   │   ├── rag_improvements.py # Enhanced retrieval logic
│   │   └── ...
│   ├── components/          # UI components and interfaces
│   │   ├── ui_components.py # Reusable UI elements
│   │   ├── enterprise_ui.py # Analytics dashboard
│   │   └── ...
│   ├── config/             # Configuration management
│   │   └── settings.py     # System settings
│   └── utils/              # Shared utilities
│       └── utils.py        # Logging and performance monitoring
├── chroma_db/              # Vector database storage
└── README.md               # Project documentation
```

## Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **UI Pattern**: Multi-step workflow with session state management
- **Components**: Modular UI components for reusability (status badges, metric cards, data source management)
- **Styling**: Custom CSS for professional appearance with gradients and responsive design
- **Visualization**: Plotly integration for metrics and analytics
- **Copy Functionality**: Simple text area implementation for easy answer copying

## Backend Architecture
- **Pattern**: Service-oriented architecture with clear separation of concerns
- **Core Services**:
  - `RAGEngine`: Orchestrates AI model interactions and query processing
  - `DataIngestionService`: Handles web scraping and document processing
  - `VectorStoreManager`: Manages ChromaDB vector storage and similarity search
  - `WikipediaIngestionService`: Specialized service for Wikipedia article ingestion with rate limiting
  - `TavilyIntegrationService`: Real-time web search integration with content cleaning and processing
  - `HybridRAGProcessor`: Intelligent processor that compares KB data with web data and updates knowledge base
  - `WebRAGProcessor`: Legacy processor for web search integration
  - `WebCacheDatabase`: PostgreSQL-based caching system for web search results and content
  - `WebCacheUI`: Dashboard for monitoring and managing web search cache
  - `RAGSystemAPI`: Main API layer that coordinates all services
  - `EnhancedRetrieval`: Multi-strategy retrieval with intelligent ranking
  - `AdvancedEmbeddingsManager`: Better embeddings with caching and optimization
  - `RAGTrainingSystem`: Continuous improvement and performance monitoring
  - `SearchFallbackService`: External search API integration for missing knowledge
- **Concurrency**: ThreadPoolExecutor for parallel processing with Wikipedia rate limiting
- **Error Handling**: Comprehensive logging and graceful failure handling
- **Clean Architecture**: Organized src/ structure with proper import management

## Data Processing Pipeline
- **Text Extraction**: Uses `trafilatura` and `BeautifulSoup` for web content extraction
- **Document Chunking**: LangChain's `RecursiveCharacterTextSplitter` for intelligent text segmentation
- **Embedding Generation**: Sentence Transformers for creating vector representations
- **Storage**: FAISS for efficient similarity search and retrieval

## AI Model Integration
- **Primary**: SARVAM API (sarvam-m) for reliable processing with automatic fallback
- **Secondary**: DeepSeek API for fast text generation (deepseek-chat, deepseek-coder) as fallback
- **Tertiary**: OpenRouter API with Kimi model (moonshotai/kimi-k2:free) as backup
- **Final**: Direct OpenAI API support as final backup option
- **Embeddings**: ChromaDB's built-in embedding models for vector representation
- **Prompt Engineering**: ChatPromptTemplate for structured AI interactions
- **Model Selection**: Configurable models including sarvam-m, deepseek-chat, deepseek-coder, moonshotai/kimi-k2:free, openai/gpt-4o, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-8b-instruct
- **Wikipedia Integration**: Specialized Wikipedia ingestion service with rate limiting and smart content filtering

## Configuration Management
- **Settings**: Centralized configuration with dataclasses for type safety
- **Environment Variables**: API keys and sensitive data loaded from environment
- **Model Selection**: Configurable AI models with fallback mechanisms

# External Dependencies

## AI/ML Services
- **SARVAM API**: Primary AI service with automatic fallback capabilities (requires API key)
- **DeepSeek API**: Fast text generation with competitive performance (requires API key)
- **OpenRouter API**: Multi-provider AI service supporting various models (optional API key)
- **OpenAI API**: Direct API support as final fallback option (optional API key)
- **Tavily API**: Real-time web search and content retrieval service (requires API key)
- **PostgreSQL**: Database for caching web search results and processed content
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