# Overview

This is an Enterprise RAG (Retrieval-Augmented Generation) System built with Streamlit that enables users to ingest data from various online sources and query that data using AI models. The system provides a multi-step workflow for data ingestion, processing, and intelligent querying with a professional web interface.

The application allows users to:
- Add and configure data sources (web URLs, APIs, files)
- Process and chunk documents for vector storage
- Query the processed data using OpenAI or HuggingFace models
- Monitor processing status and view query history
- Manage vector embeddings with FAISS

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
  - `VectorStoreManager`: Manages FAISS vector storage and similarity search
  - `RAGSystemAPI`: Main API layer that coordinates all services
- **Concurrency**: ThreadPoolExecutor for parallel processing
- **Error Handling**: Comprehensive logging and graceful failure handling

## Data Processing Pipeline
- **Text Extraction**: Uses `trafilatura` and `BeautifulSoup` for web content extraction
- **Document Chunking**: LangChain's `RecursiveCharacterTextSplitter` for intelligent text segmentation
- **Embedding Generation**: Sentence Transformers for creating vector representations
- **Storage**: FAISS for efficient similarity search and retrieval

## AI Model Integration
- **Primary**: OpenRouter API supporting multiple models (GPT-4, Claude, Llama) for text generation
- **Fallback**: Direct OpenAI API support as backup option
- **Embeddings**: TF-IDF vectorization with scikit-learn for efficient similarity search
- **Prompt Engineering**: ChatPromptTemplate for structured AI interactions
- **Model Selection**: Configurable models including openai/gpt-4o, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-8b-instruct

## Configuration Management
- **Settings**: Centralized configuration with dataclasses for type safety
- **Environment Variables**: API keys and sensitive data loaded from environment
- **Model Selection**: Configurable AI models with fallback mechanisms

# External Dependencies

## AI/ML Services
- **OpenRouter API**: Primary AI service supporting multiple model providers (requires API key)
- **OpenAI API**: Direct API support as fallback option (optional API key)
- **TF-IDF Vectorization**: Local document embedding using scikit-learn

## Vector Storage
- **TF-IDF Vectorization**: Scikit-learn based text vectorization for similarity search
- **Persistent Storage**: Local file system for vector index and document storage
- **Cosine Similarity**: Efficient similarity calculation for document retrieval

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