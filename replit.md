# Overview

This Enterprise RAG (Retrieval-Augmented Generation) System, built with Streamlit, provides comprehensive knowledge access through intelligent querying via a professional web interface. It enables users to ingest data from various online sources, Wikipedia articles, and local documents, combining a local knowledge base with real-time web search. Key capabilities include adding and configuring diverse data sources, uploading and processing local documents, ingesting Wikipedia articles with smart sampling, integrating real-time web search, processing and chunking documents for vector storage, and querying processed data using various AI providers with web-enhanced context. The system also supports intelligent hybrid RAG features like smart knowledge comparison, automatic knowledge base updates, and enhanced answer quality with specific, detailed responses. The project aims to deliver a robust, secure, and performant RAG solution for enterprise knowledge management with a business vision to provide scalable AI-powered knowledge management.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Project Structure
The project follows a clean, organized structure with `app.py` as the main entry point. Core logic resides in `src/backend/`, UI components in `src/components/`, configurations in `src/config/`, and utilities in `src/utils/`.

## Frontend Architecture
-   **Framework**: Streamlit for rapid web application development.
-   **UI Pattern**: Multi-step workflow with session state management.
-   **Components**: Modular UI components for reusability, including status badges, metric cards, and data source management.
-   **Styling**: Custom CSS for a professional appearance. Clean, minimal UI design.
-   **Visualization**: Plotly integration for metrics and analytics.
-   **Security**: Streamlit debug elements hidden, prevention of text selection, right-click blocking, and console protection for production hardening.

## Backend Architecture
-   **Pattern**: Service-oriented architecture ensuring clear separation of concerns.
-   **Core Services**: `RAGEngine`, `DataIngestionService`, `VectorStoreManager`, `WikipediaIngestionService`, `TavilyIntegrationService`, `HybridRAGProcessor`, `WebCacheDatabase`, `RAGSystemAPI`, `EnhancedRetrieval`, `AdvancedEmbeddingsManager`.
-   **API Gateway**: FastAPI-based REST API with JWT authentication, rate limiting, health monitoring, and data ingestion endpoints. Includes webhook support (Slack, Teams) and enterprise integrations (Salesforce, Office 365, Google Workspace, Zendesk).
-   **Concurrency**: ThreadPoolExecutor for parallel processing.
-   **Error Handling**: Comprehensive logging and graceful failure handling.
-   **Authentication**: JWT-based authentication with configurable expiry, role-based access control (Admin, User, Viewer), multi-level rate limiting, secure session management, PBKDF2 password hashing, and brute-force protection. Firebase-only Google Authentication is integrated, syncing users with a local database.
-   **Performance-Based Rate Limiting**: Advanced rate limiter uses actual API processing time and token consumption to determine query complexity, featuring aggressive exponential backoff and graceful degradation.
-   **Dynamic Source Retrieval**: System respects user's "Max Sources" setting (1-20), scaling vector search and reranking accordingly.
-   **Agentic RAG System**: Multi-agent architecture with four specialized AI agents (Researcher, Analyzer, Validator, Synthesizer). Intelligent query complexity classification, autonomous agent orchestration, real-time agent monitoring, cross-agent validation, and comprehensive response synthesis.

## Data Processing Pipeline
-   **Text Extraction**: Utilizes `trafilatura` and `BeautifulSoup` for web content.
-   **Document Chunking**: LangChain's `RecursiveCharacterTextSplitter` for intelligent segmentation.
-   **Embedding Generation**: Sentence Transformers for vector representations.
-   **Storage**: FAISS for efficient similarity search and retrieval during processing.

## AI Model Integration
-   **Primary**: SARVAM API (sarvam-m) with automatic fallback.
-   **Secondary**: DeepSeek API (deepseek-chat, deepseek-coder).
-   **Tertiary**: OpenRouter API with Kimi model (moonshotai/kimi-k2:free).
-   **Final Backup**: Direct OpenAI API support.
-   **Embeddings**: ChromaDB's built-in embedding models.
-   **Prompt Engineering**: ChatPromptTemplate for structured AI interactions.
-   **Model Selection**: Configurable models including sarvam-m, deepseek-chat, deepseek-coder, moonshotai/kimi-k2:free, openai/gpt-4o, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-8b-instruct. Includes integration of free LLM models (Llama 3.2, Mistral 7B, Gemma, Qwen, DeepSeek) via Hugging Face Inference API.
-   **Intelligent Query Classification**: GPU only for complex queries, SARVAM/Tavily for simple ones.
-   **Cost Optimization**: Intelligent model selection prioritizing free models.

## Configuration Management
Centralized configuration using dataclasses, with API keys and sensitive data loaded from environment variables. Secure API key generation with cryptographic security, complete key lifecycle management, multiple access scopes, usage analytics, rate limiting configuration, and custom expiration settings.

# External Dependencies

## AI/ML Services
-   **SARVAM API**: Primary AI service.
-   **DeepSeek API**: Text generation.
-   **OpenRouter API**: Multi-provider AI service.
-   **OpenAI API**: Final fallback AI option.
-   **Tavily API**: Real-time web search and content retrieval.
-   **Serper Dev API**: For fast, cost-effective web search.
-   **Hugging Face Inference API**: For free LLM models.

## Databases
-   **PostgreSQL**: For caching web search results and processed content.
-   **ChromaDB**: Vector database for embeddings and similarity search, with built-in embeddings and persistent storage.
-   **SQLite**: For user and session management in the authentication system and usage-based billing.

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

## Payments
-   **Stripe**: For subscription system integration.