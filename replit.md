# Overview

This Enterprise RAG (Retrieval-Augmented Generation) System, built with Streamlit, provides comprehensive knowledge access through intelligent querying via a professional web interface. It enables users to ingest data from various online sources, Wikipedia articles, and local documents, combining a local knowledge base with real-time web search. Key capabilities include:

-   Adding and configuring diverse data sources (web URLs, APIs, files).
-   Uploading and processing local documents (text, PDF, Excel, CSV, Word).
-   Ingesting Wikipedia articles at scale with smart sampling strategies (balanced, category-focused, random).
-   Integrating real-time web search (Tavily API) for live data retrieval.
-   Processing and chunking documents for vector storage.
-   Querying processed data using SARVAM API or other AI providers with web-enhanced context.
-   Monitoring processing status and viewing query history.
-   Intelligent hybrid RAG features like smart knowledge comparison, automatic knowledge base updates, and enhanced answer quality with specific, detailed responses.

The project aims to deliver a robust, secure, and performant RAG solution for enterprise knowledge management.

# User Preferences

Preferred communication style: Simple, everyday language.

# Recent Changes

## Enhanced UI and Feature Updates (August 3, 2025)
- ✅ Clean, minimal UI design without gradients per user preference
- ✅ API key delete functionality with double-confirmation safety
- ✅ One-time display security for copied API keys (keys hidden after copying)
- ✅ Real-time analytics dashboard with auto-refresh every 30 seconds
- ✅ Fixed Agentic RAG summary to provide direct answers instead of performance metrics
- ✅ Enhanced user interface with professional styling and clean aesthetics
- ✅ Improved API key management with complete lifecycle controls
- ✅ Auto-updating system metrics and real-time data visualization
- ✅ Fixed AI response disappearing issue by removing unwanted auto-refresh functionality
- ✅ Implemented robust rate limit fallback system to always provide answers when knowledge base has relevant data
- ✅ Enhanced knowledge base fallback with intelligent text processing and relevance scoring

## API Gateway Implementation (August 3, 2025)
- ✅ Built comprehensive FastAPI-based REST API Gateway on port 8000
- ✅ JWT authentication with role-based access control
- ✅ Rate limiting (100 requests/hour per IP)
- ✅ Health monitoring and system statistics endpoints
- ✅ Query processing with fallback when RAG system unavailable
- ✅ Data ingestion endpoints for URLs and text content
- ✅ CORS middleware and comprehensive error handling
- ✅ Background task processing for async operations
- ✅ Interactive API documentation at /docs and /redoc
- ✅ Python client example and integration guide created

## One-Click API Key Management (August 3, 2025)
- ✅ Secure API key generation with cryptographic security (SHA-256 hashing)
- ✅ Complete key lifecycle management (create, update, revoke, regenerate, delete)
- ✅ Multiple access scopes (read-only, query-only, ingest-only, full-access, admin)
- ✅ Usage analytics and monitoring with detailed statistics
- ✅ Rate limiting configuration (1-10,000 requests/hour per key)
- ✅ Custom expiration settings (1-365 days or never expires)
- ✅ SQLite database for secure key storage and audit logging
- ✅ Streamlit web interface integration with user-friendly management
- ✅ Complete REST API endpoints for programmatic key management
- ✅ Enterprise security features and comprehensive documentation
- ✅ One-time key display with permanent hiding after copying for enhanced security

## Agentic RAG System (August 3, 2025)
- ✅ Multi-agent architecture with four specialized AI agents (Researcher, Analyzer, Validator, Synthesizer)
- ✅ Intelligent query complexity classification (Simple, Complex, Research, Analytical)
- ✅ Autonomous agent orchestration with phase-based processing workflow
- ✅ Real-time agent monitoring and status visualization in Streamlit interface
- ✅ Advanced configuration panel with processing modes and research depth controls
- ✅ Cross-agent validation and fact-checking capabilities
- ✅ Comprehensive response synthesis with enterprise-grade quality assessment
- ✅ API endpoint integration for programmatic agentic processing
- ✅ Pre-built complex query examples and debug mode for detailed insights
- ✅ Performance optimization with parallel processing and intelligent caching
- ✅ Fixed conclusion generation to provide direct answers to user questions instead of performance metrics

# System Architecture

## Project Structure
The project follows a clean, organized structure with `app.py` as the main entry point. Core logic resides in `src/backend/`, UI components in `src/components/`, configurations in `src/config/`, and utilities in `src/utils/`.

## Frontend Architecture
-   **Framework**: Streamlit for rapid web application development.
-   **UI Pattern**: Multi-step workflow with session state management.
-   **Components**: Modular UI components for reusability, including status badges, metric cards, and data source management.
-   **Styling**: Custom CSS for a professional appearance with gradients and responsive design.
-   **Visualization**: Plotly integration for metrics and analytics.
-   **Security**: Streamlit debug elements hidden, prevention of text selection, right-click blocking, and console protection for production hardening.

## Backend Architecture
-   **Pattern**: Service-oriented architecture ensuring clear separation of concerns.
-   **Core Services**:
    -   `RAGEngine`: Orchestrates AI model interactions and query processing.
    -   `DataIngestionService`: Handles web scraping and document processing.
    -   `VectorStoreManager`: Manages ChromaDB vector storage and similarity search.
    -   `WikipediaIngestionService`: Specialized service for Wikipedia ingestion with rate limiting.
    -   `TavilyIntegrationService`: Real-time web search integration with content cleaning.
    -   `HybridRAGProcessor`: Compares knowledge base data with web data and updates the knowledge base.
    -   `WebCacheDatabase`: PostgreSQL-based caching for web search results.
    -   `RAGSystemAPI`: Main API layer coordinating all services.
    -   `EnhancedRetrieval`: Multi-strategy retrieval with intelligent ranking.
    -   `AdvancedEmbeddingsManager`: Manages embeddings with caching and optimization.
-   **API Gateway**: FastAPI-based REST API with JWT authentication, rate limiting, and enterprise integrations.
    -   `Enterprise API Gateway`: Full REST API access on port 8000 with comprehensive endpoints.
    -   `Webhook Integration`: Slack and Teams webhook support for notifications.
    -   `Enterprise Integrations`: Salesforce, Office 365, Google Workspace, and Zendesk connectors.
    -   `RAGTrainingSystem`: Continuous improvement and performance monitoring.
    -   `SearchFallbackService`: External search API integration.
-   **Concurrency**: ThreadPoolExecutor for parallel processing.
-   **Error Handling**: Comprehensive logging and graceful failure handling.
-   **Authentication**: JWT-based authentication with configurable expiry, role-based access control (Admin, User, Viewer), multi-level rate limiting, secure session management, PBKDF2 password hashing, and brute-force protection. Firebase-only Google Authentication is integrated, syncing users with a local database.
-   **Performance-Based Rate Limiting**: Advanced rate limiter uses actual API processing time and token consumption to determine query complexity. Conservative limits prevent 429 errors: simple (8/min), complex (4/min), quantum_physics (2/min). Features aggressive exponential backoff, failure tracking, and graceful degradation.
-   **Dynamic Source Retrieval**: Removed hardcoded source limitations. The system now properly respects user's "Max Sources" setting (1-20), scaling vector search and reranking accordingly for comprehensive knowledge retrieval.

## Data Processing Pipeline
-   **Text Extraction**: Utilizes `trafilatura` and `BeautifulSoup` for web content.
-   **Document Chunking**: LangChain's `RecursiveCharacterTextSplitter` for intelligent segmentation.
-   **Embedding Generation**: Sentence Transformers for vector representations.
-   **Storage**: FAISS for efficient similarity search and retrieval during processing.

## AI Model Integration
-   **Primary**: SARVAM API (sarvam-m) with automatic fallback.
-   **Secondary**: DeepSeek API (deepseek-chat, deepseek-coder) for fast generation.
-   **Tertiary**: OpenRouter API with Kimi model (moonshotai/kimi-k2:free).
-   **Final Backup**: Direct OpenAI API support.
-   **Embeddings**: ChromaDB's built-in embedding models.
-   **Prompt Engineering**: ChatPromptTemplate for structured AI interactions.
-   **Model Selection**: Configurable models including sarvam-m, deepseek-chat, deepseek-coder, moonshotai/kimi-k2:free, openai/gpt-4o, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-8b-instruct.

## Configuration Management
Centralized configuration using dataclasses, with API keys and sensitive data loaded from environment variables.

# External Dependencies

## AI/ML Services
-   **SARVAM API**: Primary AI service.
-   **DeepSeek API**: Text generation.
-   **OpenRouter API**: Multi-provider AI service.
-   **OpenAI API**: Final fallback AI option.
-   **Tavily API**: Real-time web search and content retrieval.

## Databases
-   **PostgreSQL**: For caching web search results and processed content.
-   **ChromaDB**: Vector database for embeddings and similarity search, with built-in embeddings and persistent storage.
-   **SQLite**: For user and session management in the authentication system.

## Document Processing Libraries
-   **PyPDF2**: PDF text extraction.
-   **pandas**: CSV and Excel data manipulation.
-   **openpyxl**: Modern Excel (.xlsx) processing.
-   **xlrd**: Legacy Excel (.xls) support.
-   **python-docx**: Word document (.docx) text extraction.

## Web Scraping & Content Processing
-   **Trafilatura**: Web content extraction and cleaning.
-   **BeautifulSoup**: HTML parsing.
-   **Requests**: HTTP client for web data retrieval.
-   **LangChain**: Document processing, text splitting, and RAG orchestration.

## UI/Visualization
-   **Streamlit**: Web application framework.
-   **Plotly**: Interactive charts and visualizations.

## Authentication
-   **Firebase Admin SDK**: Server-side Firebase authentication.
-   **Firebase Web SDK**: Client-side Firebase authentication.