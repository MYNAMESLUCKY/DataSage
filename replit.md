# Overview
This Enterprise RAG (Retrieval-Augmented Generation) System, built with Streamlit, provides comprehensive knowledge access through intelligent querying via a professional web interface, plus a separate AI-powered coding assistant. It enables users to ingest data from various online sources, Wikipedia articles, and local documents, combining a local knowledge base with real-time web search. Key capabilities include:
- Adding and configuring diverse data sources (web URLs, APIs, files).
- Uploading and processing local documents (text, PDF, Excel, CSV, Word).
- Ingesting Wikipedia articles at scale with smart sampling strategies.
- Integrating real-time web search (Tavily API) for live data retrieval.
- Processing and chunking documents for vector storage.
- Querying processed data using SARVAM API or other AI providers with web-enhanced context.
- Monitoring processing status and viewing query history.
- Intelligent hybrid RAG features like smart knowledge comparison, automatic knowledge base updates, and enhanced answer quality with specific, detailed responses.
- **Separate AI Coding Assistant** - "Coding Ground" with Cursor/Lovable-like features using DeepSeek R1 and Qwen3 Coder models, user-selectable AI models with detailed descriptions, documentation access, real-time code execution, error fixing capabilities, and intelligent fallback system for API rate limits.

The project aims to deliver a robust, secure, and performant RAG solution for enterprise knowledge management, featuring ultra-fast response times, advanced mathematical and physics-enhanced processing for similarity analysis, GPU-based distributed computing for complex queries, and comprehensive coding assistance capabilities.

# User Preferences
Preferred communication style: Simple, everyday language.

# System Architecture

## Project Structure
The project follows a clean, organized structure with `enterprise_app.py` as the main entry point. Core logic resides in `src/backend/`, UI components in `src/components/`, configurations in `src/config/`, and utilities in `src/utils/`.

## GPU Computing Architecture
- **Complexity Classification**: Advanced algorithm analyzes query complexity (0-1 scale) using pattern matching, vocabulary analysis, and structural indicators.
- **GPU Service Manager**: Manages distributed processing across multiple free GPU platforms with intelligent provider selection.
- **Processing Pipeline**: Automatic routing of complex queries to GPU infrastructure for enhanced computational capabilities.
- **Fallback System**: Graceful degradation from GPU → Standard API → Knowledge Base fallback ensures reliable responses.
- **Service Integration**: Supports Google Colab, Kaggle Kernels, HuggingFace Spaces, Paperspace, Lightning AI, and Saturn Cloud.

## Frontend Architecture
- **Framework**: Streamlit for rapid web application development.
- **UI Pattern**: Multi-step workflow with session state management.
- **Components**: Modular UI components for reusability, including status badges, metric cards, and data source management.
- **Styling**: Custom CSS for a professional appearance with a clean, minimal design (no gradients).
- **Visualization**: Plotly integration for metrics and analytics.
- **Security**: Streamlit debug elements hidden, prevention of text selection, right-click blocking, and console protection for production hardening.

## Backend Architecture
- **Pattern**: Service-oriented architecture ensuring clear separation of concerns.
- **Core Services**:
    - `RAGEngine`: Orchestrates AI model interactions and query processing.
    - `DataIngestionService`: Handles web scraping and document processing.
    - `VectorStoreManager`: Manages ChromaDB vector storage and similarity search.
    - `WikipediaIngestionService`: Specialized service for Wikipedia ingestion with rate limiting.
    - `TavilyIntegrationService`: Real-time web search integration with content cleaning.
    - `HybridRAGProcessor`: Compares knowledge base data with web data and updates the knowledge base.
    - `WebCacheDatabase`: PostgreSQL-based caching for web search results.
    - `RAGSystemAPI`: Main API layer coordinating all services.
    - `EnhancedRetrieval`: Multi-strategy retrieval with intelligent ranking.
    - `AdvancedEmbeddingsManager`: Manages embeddings with caching and optimization.
    - **Agentic RAG System**: Multi-agent architecture (Researcher, Analyzer, Validator, Synthesizer) for autonomous query processing and response synthesis with real-time monitoring and configuration.
- **API Gateway**: FastAPI-based REST API on port 8000 with JWT authentication, rate limiting, health monitoring, data ingestion endpoints, CORS middleware, and interactive documentation.
    - **API Key Management**: Secure API key generation with cryptographic security (SHA-256 hashing), complete key lifecycle management (create, update, revoke, regenerate, delete), multiple access scopes, usage analytics, custom expiration settings, and SQLite database for secure key storage.
- **Concurrency**: ThreadPoolExecutor for parallel processing.
- **Error Handling**: Comprehensive logging and graceful failure handling.
- **Authentication**: JWT-based authentication with configurable expiry, role-based access control (Admin, User, Viewer), multi-level rate limiting, secure session management, PBKDF2 password hashing, brute-force protection, and Firebase-only Google Authentication synced with a local database.
- **Coding Ground System**: Separate AI-powered coding assistant running on independent infrastructure (API on port 8001, Frontend on port 5002) with user-selectable AI models (DeepSeek R1 for advanced reasoning, Qwen3 Coder 7B/14B for efficient coding), detailed model descriptions and quick switching, documentation search integration, real-time Python execution, error fixing, code explanation, Cursor-like features, and intelligent fallback responses when API rate limits are reached.
- **Performance-Based Rate Limiting**: Advanced rate limiter uses actual API processing time and token consumption for query complexity.
- **Dynamic Source Retrieval**: System respects user's "Max Sources" setting (1-20), scaling vector search and reranking accordingly.
- **Ultra-Fast Response System**: Sub-second responses using ingested documents, smart content routing for pre-written responses for basic definitions, and intelligent classification for optimal processing path.
- **Advanced Mathematical & Physics-Enhanced System**: Incorporates quantum-inspired processing, electromagnetic field theory, gravitational ranking, harmonic analysis, thermodynamic information theory, fractal dimension analysis, topological data analysis, special relativity processing, wave interference patterns, holographic principle, supersymmetric matching, chaos theory analysis, and golden ratio optimization for advanced similarity analysis and ranking.

## Data Processing Pipeline
- **Text Extraction**: Utilizes `trafilatura` and `BeautifulSoup` for web content.
- **Document Chunking**: LangChain's `RecursiveCharacterTextSplitter` for intelligent segmentation.
- **Embedding Generation**: Sentence Transformers for vector representations.
- **Storage**: FAISS for efficient similarity search and retrieval during processing.

## AI Model Integration

### RAG System Models
- **Dual-Model System**: SARVAM API (sarvam-m) and LLaMA 3.3 70B (meta-llama/llama-3.3-70b-instruct:free) with user selection capability.
- **SARVAM**: Optimized for speed and efficiency, ideal for quick queries and real-time responses.
- **LLaMA 3.3 70B**: Advanced reasoning capabilities, perfect for complex analysis and detailed explanations.
- **Embeddings**: ChromaDB's built-in embedding models.
- **Prompt Engineering**: Unified plain-text prompt system for consistent responses across both models.
- **Client Management**: Intelligent client selection based on chosen model with proper fallback handling.

### Coding Ground Models
- **DeepSeek R1** (`deepseek-reasoner`): Advanced reasoning for complex coding problems and sophisticated debugging.
- **Qwen3 Coder 7B** (`qwen/qwen-2.5-coder-7b-instruct:free`): Fast, efficient coding assistance for quick development tasks.
- **Qwen3 Coder 14B** (`qwen/qwen-2.5-coder-14b-instruct:free`): Enhanced coding capabilities for complex programming challenges.
- **Documentation Integration**: Real-time access to programming documentation, Stack Overflow, and open-source resources.

## Configuration Management
Centralized configuration using dataclasses, with API keys and sensitive data loaded from environment variables.

# External Dependencies

## AI/ML Services
- **SARVAM API**: Primary AI service for RAG system.
- **DeepSeek API**: Advanced reasoning model for Coding Ground.
- **OpenRouter API**: Multi-provider AI service for Qwen3 Coder models.
- **LLaMA API**: Advanced reasoning capabilities for RAG system.
- **Tavily API**: Real-time web search and content retrieval for both systems.

## Databases
- **PostgreSQL**: For caching web search results and processed content.
- **ChromaDB**: Vector database for embeddings and similarity search, with built-in embeddings and persistent storage.
- **SQLite**: For user and session management in the authentication system.

## Document Processing Libraries
- **PyPDF2**: PDF text extraction.
- **pandas**: CSV and Excel data manipulation.
- **openpyxl**: Modern Excel (.xlsx) processing.
- **xlrd**: Legacy Excel (.xls) support.
- **python-docx**: Word document (.docx) text extraction.

## Web Scraping & Content Processing
- **Trafilatura**: Web content extraction and cleaning.
- **BeautifulSoup**: HTML parsing.
- **Requests**: HTTP client for web data retrieval.
- **LangChain**: Document processing, text splitting, and RAG orchestration.

## UI/Visualization
- **Streamlit**: Web application framework.
- **Plotly**: Interactive charts and visualizations.

## Authentication
- **Firebase Admin SDK**: Server-side Firebase authentication.
- **Firebase Web SDK**: Client-side Firebase authentication.