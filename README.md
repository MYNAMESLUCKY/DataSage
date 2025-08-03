# Enterprise RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with Streamlit, featuring intelligent document processing, vector storage with ChromaDB, and multi-provider AI support.

## Features

- **Multi-source Data Ingestion**: Web URLs, Wikipedia articles, local documents (PDF, Word, Excel, CSV)
- **Advanced Vector Storage**: ChromaDB with semantic embeddings and similarity search
- **AI Model Flexibility**: Support for OpenRouter, DeepSeek, and OpenAI models
- **Enterprise Features**: Analytics dashboard, query optimization, caching
- **Professional UI**: Clean Streamlit interface with copy functionality

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export DEEPSEEK_API="your-key-here"
   export OPENROUTER_API="your-key-here"
   ```

3. **Run Application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Architecture

```
src/
├── backend/           # Core RAG system logic
│   ├── api.py        # Main API orchestrator
│   ├── rag_engine.py # AI model interactions
│   ├── vector_store_chroma.py # ChromaDB integration
│   └── ...
├── components/       # UI components
│   ├── ui_components.py
│   ├── enterprise_ui.py
│   └── ...
└── config/          # Configuration management
    └── settings.py
```

## Usage

1. **Data Ingestion**: Add URLs, upload files, or ingest Wikipedia articles
2. **Processing**: Documents are chunked and embedded into vector storage
3. **Querying**: Ask questions and get AI-powered answers with source attribution
4. **Analytics**: Monitor system performance and query effectiveness

## Configuration

- Model selection via UI (Kimi, DeepSeek, GPT-4o, Claude)
- Adjustable similarity thresholds and result counts
- Caching and performance optimization settings

## Enterprise Features

- User authentication ready
- Performance monitoring
- Query history and analytics
- Scalable architecture for multi-user deployment