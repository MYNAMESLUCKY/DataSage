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
    -   `RAGTrainingSystem`: Continuous improvement and performance monitoring.
    -   `SearchFallbackService`: External search API integration.
-   **Concurrency**: ThreadPoolExecutor for parallel processing.
-   **Error Handling**: Comprehensive logging and graceful failure handling.
-   **Authentication**: JWT-based authentication with configurable expiry, role-based access control (Admin, User, Viewer), multi-level rate limiting, secure session management, PBKDF2 password hashing, and brute-force protection. Firebase-only Google Authentication is integrated, syncing users with a local database.
-   **Performance-Based Rate Limiting**: Advanced rate limiter now uses actual API processing time and token consumption to determine query complexity, replacing keyword-based classification. Complexity levels: simple (<8s, <800 tokens), complex (8-15s, 800-1200 tokens), quantum_physics (>15s, >1200 tokens).
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