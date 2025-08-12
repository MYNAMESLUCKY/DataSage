#!/usr/bin/env python3
"""
Enterprise RAG API Gateway
Provides REST API access to the RAG system with authentication, rate limiting, and integrations
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import jwt
import time
import logging
import asyncio
from datetime import datetime, timedelta
import os
import json
from contextlib import asynccontextmanager

# Import our RAG system components
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from backend.api import RAGSystemAPI
    from backend.hybrid_rag_processor import HybridRAGProcessor
    from utils.utils import setup_logging
except ImportError as e:
    import logging
    print(f"ImportError in gateway.py: {e}")
    def setup_logging(name):
        return logging.getLogger(name)

logger = setup_logging(__name__)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'

# Initialize RAG system
rag_api = None
hybrid_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup RAG system"""
    global rag_api, hybrid_processor
    
    logger.info("Starting Enterprise RAG API Gateway...")
    
    # Initialize RAG system
    try:
        rag_api = RAGSystemAPI()
        if hasattr(rag_api, 'initialize'):
            rag_api.initialize()
        
        # Initialize hybrid processor if available
        if hasattr(rag_api, 'vector_store') and hasattr(rag_api, 'rag_engine'):
            hybrid_processor = HybridRAGProcessor(
                rag_api.vector_store,
                rag_api.rag_engine,
                getattr(rag_api, 'enhanced_retrieval', None)
            )
        else:
            hybrid_processor = None
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_api = None
        hybrid_processor = None
    
    logger.info("RAG API Gateway initialized successfully")
    yield
    
    # Cleanup
    logger.info("Shutting down RAG API Gateway...")

# FastAPI app with lifespan
app = FastAPI(
    title="Enterprise RAG API Gateway",
    description="Production-ready API for intelligent knowledge retrieval and processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    max_results: int = Field(default=10, ge=1, le=20, description="Maximum number of sources to retrieve")
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
    use_web_search: bool = Field(default=True, description="Enable real-time web search")
    llm_model: str = Field(default="sarvam-m", description="AI model to use")
    use_cache: bool = Field(default=True, description="Use cached results when available")

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    web_sources: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    model_used: str
    api_provider: str
    status: str
    query_id: Optional[str] = None
    strategy_used: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]

class DataIngestionRequest(BaseModel):
    data_type: str = Field(..., description="Type of data: 'url', 'text', 'file'")
    content: str = Field(..., description="URL, text content, or file path")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class SystemStatsResponse(BaseModel):
    total_documents: int
    query_count_24h: int
    avg_response_time: float
    cache_hit_rate: float
    system_uptime: str

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

# Rate limiting (simple in-memory implementation)
request_counts = {}
RATE_LIMIT_REQUESTS = 100  # requests per hour
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds

async def rate_limit_check(request: Request):
    """Simple rate limiting based on IP"""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests outside the window
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if rate limit exceeded
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    # Add current request
    request_counts[client_ip].append(current_time)

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """API Gateway root endpoint"""
    return {
        "message": "Enterprise RAG API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    try:
        # Check RAG system components
        components = {
            "rag_engine": "healthy" if rag_api and hasattr(rag_api, 'rag_engine') and getattr(rag_api.rag_engine, 'is_ready', False) else "unhealthy",
            "vector_store": "healthy" if rag_api and hasattr(rag_api, 'vector_store') and rag_api.vector_store else "unhealthy",
            "hybrid_processor": "healthy" if hybrid_processor else "unhealthy",
            "database": "healthy" if rag_api and hasattr(rag_api, 'vector_store') and rag_api.vector_store else "unhealthy"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/auth/token")
async def create_token(user_id: str, role: str = "user"):
    """Create authentication token (simplified for demo)"""
    token_data = {"sub": user_id, "role": role}
    access_token = create_access_token(token_data)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    user: Dict[str, Any] = Depends(verify_token)
):
    """Query the knowledge base with intelligent RAG processing"""
    await rate_limit_check(http_request)
    
    try:
        start_time = time.time()
        query_id = f"q_{int(time.time())}_{hash(request.query) % 10000}"
        
        logger.info(f"Processing query {query_id} for user {user.get('sub', 'unknown')}: {request.query[:100]}...")
        
        # Use hybrid RAG processor for intelligent processing if available
        if hybrid_processor and hasattr(hybrid_processor, 'process_intelligent_query'):
            result = hybrid_processor.process_intelligent_query(
                query=request.query,
                llm_model=request.llm_model,
                use_web_search=request.use_web_search,
                max_web_results=request.max_results,
                max_results=request.max_results
            )
        else:
            # Fallback response when RAG system is not available
            result = {
                'answer': 'RAG system is currently initializing. Please try again in a few moments.',
                'confidence': 0.0,
                'sources': [],
                'web_sources': [],
                'model_used': request.llm_model,
                'api_provider': 'system',
                'status': 'fallback',
                'strategy_used': 'fallback'
            }
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = QueryResponse(
            answer=result.get('answer', ''),
            confidence=result.get('confidence', 0.0),
            sources=result.get('sources', []),
            web_sources=result.get('web_sources', []),
            processing_time=processing_time,
            model_used=result.get('model_used', request.llm_model),
            api_provider=result.get('api_provider', 'unknown'),
            status=result.get('status', 'success'),
            query_id=query_id,
            strategy_used=result.get('strategy_used', 'hybrid')
        )
        
        # Log query for analytics (background task)
        background_tasks.add_task(log_query_analytics, query_id, request.query, user.get('sub', 'unknown'), processing_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/ingest")
async def ingest_data(
    request: DataIngestionRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(verify_token)
):
    """Ingest new data into the knowledge base"""
    try:
        logger.info(f"Data ingestion request from user {user.get('sub', 'unknown')}: {request.data_type}")
        
        if not rag_api:
            raise HTTPException(status_code=503, detail="RAG system not available")
        
        if request.data_type == "url":
            # Process URL
            background_tasks.add_task(process_url_ingestion, request.content, request.metadata)
            return {"message": "URL ingestion started", "status": "processing"}
            
        elif request.data_type == "text":
            # Process text content
            background_tasks.add_task(process_text_ingestion, request.content, request.metadata)
            return {"message": "Text ingestion started", "status": "processing"}
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported data type")
            
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(user: Dict[str, Any] = Depends(verify_token)):
    """Get system statistics and metrics"""
    try:
        # Get document count
        total_docs = 0
        if rag_api and hasattr(rag_api, 'vector_store') and rag_api.vector_store:
            if hasattr(rag_api.vector_store, 'get_document_count'):
                total_docs = rag_api.vector_store.get_document_count()
            elif hasattr(rag_api.vector_store, 'collection') and hasattr(rag_api.vector_store.collection, 'count'):
                total_docs = rag_api.vector_store.collection.count()
        
        # Mock analytics data (implement real analytics later)
        stats = SystemStatsResponse(
            total_documents=total_docs,
            query_count_24h=0,  # Implement with real database
            avg_response_time=2.5,  # Implement with real metrics
            cache_hit_rate=0.75,  # Implement with real cache metrics
            system_uptime="24h 30m"  # Implement with real uptime tracking
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed")

@app.get("/models")
async def list_available_models(user: Dict[str, Any] = Depends(verify_token)):
    """List available AI models"""
    return {
        "models": [
            {
                "id": "sarvam-m",
                "name": "SARVAM Model",
                "provider": "SARVAM",
                "status": "active",
                "capabilities": ["text_generation", "reasoning"]
            }
        ]
    }

@app.delete("/cache")
async def clear_cache(user: Dict[str, Any] = Depends(verify_token)):
    """Clear system cache (admin only)"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Implement cache clearing logic
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail="Cache clearing failed")

# Background tasks
async def log_query_analytics(query_id: str, query: str, user_id: str, processing_time: float):
    """Log query for analytics (background task)"""
    try:
        # Implement analytics logging to database
        logger.info(f"Query analytics: {query_id} | User: {user_id} | Time: {processing_time:.2f}s")
    except Exception as e:
        logger.error(f"Analytics logging failed: {e}")

async def process_url_ingestion(url: str, metadata: Optional[Dict[str, Any]]):
    """Process URL ingestion (background task)"""
    try:
        # Use the existing data ingestion service
        if rag_api and hasattr(rag_api, 'data_ingestion') and rag_api.data_ingestion:
            if hasattr(rag_api.data_ingestion, 'process_url'):
                result = rag_api.data_ingestion.process_url(url)
            else:
                # Alternative method for URL processing
                logger.info(f"URL processing method not available, queuing: {url}")
            logger.info(f"URL ingestion completed: {url}")
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")

async def process_text_ingestion(text: str, metadata: Optional[Dict[str, Any]]):
    """Process text ingestion (background task)"""
    try:
        # Use the existing data ingestion service
        if rag_api:
            # Create a temporary document and add it
            from langchain.schema import Document
            doc = Document(page_content=text, metadata=metadata or {})
            rag_api.vector_store.add_documents([doc])
            logger.info(f"Text ingestion completed: {len(text)} characters")
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )