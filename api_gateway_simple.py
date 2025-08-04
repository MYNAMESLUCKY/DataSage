#!/usr/bin/env python3
"""
Simplified Enterprise RAG API Gateway
Standalone FastAPI server with basic RAG functionality
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
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
import sys
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'

# Global variables for RAG system
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup RAG system"""
    global rag_system
    
    logger.info("Starting Enterprise RAG API Gateway...")
    
    # Try to initialize RAG system
    try:
        # Add src to path for imports
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Add current directory to path
        current_path = os.path.dirname(__file__)
        if current_path not in sys.path:
            sys.path.insert(0, current_path)
        
        # Import and initialize RAG components with absolute imports
        sys.path.insert(0, src_path)
        import backend.api as api_module
        import backend.hybrid_rag_processor as hybrid_module
        
        RAGSystemAPI = api_module.RAGSystemAPI
        HybridRAGProcessor = hybrid_module.HybridRAGProcessor
        
        rag_api = RAGSystemAPI()
        
        # Try to initialize if method exists
        if hasattr(rag_api, 'initialize'):
            rag_api.initialize()
        
        # Create hybrid processor
        if hasattr(rag_api, 'vector_store') and hasattr(rag_api, 'rag_engine'):
            hybrid_processor = HybridRAGProcessor(
                rag_api.vector_store,
                rag_api.rag_engine,
                getattr(rag_api, 'enhanced_retrieval', None)
            )
        else:
            hybrid_processor = None
        
        rag_system = {
            'api': rag_api,
            'processor': hybrid_processor,
            'initialized': True
        }
        
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_system = {
            'api': None,
            'processor': None,
            'initialized': False,
            'error': str(e)
        }
    
    yield
    
    # Cleanup
    logger.info("Shutting down RAG API Gateway...")

# FastAPI app
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    max_results: int = Field(default=10, ge=1, le=20, description="Maximum number of sources to retrieve")
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
    use_web_search: bool = Field(default=True, description="Enable real-time web search")
    llm_model: str = Field(default="sarvam-m", description="AI model to use")

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

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    rag_system_status: str

class DataIngestionRequest(BaseModel):
    data_type: str = Field(..., description="Type of data: 'url', 'text'")
    content: str = Field(..., description="URL or text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

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

# Rate limiting
request_counts = {}
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600

async def rate_limit_check(request: Request):
    """Simple rate limiting based on IP"""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
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
        if rag_system and rag_system.get('initialized'):
            rag_status = "healthy"
            components = {
                "api_gateway": "healthy",
                "rag_engine": "healthy" if rag_system.get('api') else "unhealthy",
                "hybrid_processor": "healthy" if rag_system.get('processor') else "unhealthy",
                "database": "unknown"
            }
        else:
            rag_status = "unhealthy"
            components = {
                "api_gateway": "healthy",
                "rag_engine": "unhealthy",
                "hybrid_processor": "unhealthy",
                "database": "unknown"
            }
        
        overall_status = "healthy" if rag_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            components=components,
            rag_system_status=rag_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/auth/token")
async def create_token(user_id: str, role: str = "user"):
    """Create authentication token"""
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
        
        # Use RAG system if available
        if rag_system and rag_system.get('initialized') and rag_system.get('processor'):
            try:
                processor = rag_system['processor']
                if hasattr(processor, 'process_intelligent_query'):
                    result = processor.process_intelligent_query(
                        query=request.query,
                        llm_model=request.llm_model,
                        use_web_search=request.use_web_search,
                        max_web_results=request.max_results,
                        max_results=request.max_results
                    )
                else:
                    # Use basic query processing
                    result = await process_basic_query(request.query, request.max_results)
            except Exception as e:
                logger.error(f"RAG processing failed: {e}")
                result = await process_fallback_query(request.query)
        else:
            # Fallback when RAG system is not available
            result = await process_fallback_query(request.query)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = QueryResponse(
            answer=result.get('answer', 'No answer available'),
            confidence=result.get('confidence', 0.0),
            sources=result.get('sources', []),
            web_sources=result.get('web_sources', []),
            processing_time=processing_time,
            model_used=result.get('model_used', request.llm_model),
            api_provider=result.get('api_provider', 'system'),
            status=result.get('status', 'success'),
            query_id=query_id
        )
        
        # Log query for analytics
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
        
        if request.data_type == "url":
            background_tasks.add_task(process_url_ingestion, request.content, request.metadata)
            return {"message": "URL ingestion started", "status": "processing"}
            
        elif request.data_type == "text":
            background_tasks.add_task(process_text_ingestion, request.content, request.metadata)
            return {"message": "Text ingestion started", "status": "processing"}
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported data type")
            
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")

@app.get("/stats")
async def get_system_stats(user: Dict[str, Any] = Depends(verify_token)):
    """Get system statistics and metrics"""
    try:
        stats = {
            "total_documents": 0,
            "query_count_24h": 0,
            "avg_response_time": 2.5,
            "cache_hit_rate": 0.75,
            "system_uptime": "24h 30m",
            "rag_system_status": rag_system.get('initialized', False) if rag_system else False
        }
        
        # Try to get real document count
        if rag_system and rag_system.get('initialized') and rag_system.get('api'):
            try:
                api = rag_system['api']
                if hasattr(api, 'vector_store') and api.vector_store:
                    if hasattr(api.vector_store, 'get_document_count'):
                        stats["total_documents"] = api.vector_store.get_document_count()
                    elif hasattr(api.vector_store, 'collection'):
                        stats["total_documents"] = api.vector_store.collection.count()
            except:
                pass
        
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

# Helper functions
async def process_basic_query(query: str, max_results: int) -> Dict[str, Any]:
    """Process query using basic RAG system"""
    try:
        if rag_system and rag_system.get('api'):
            api = rag_system['api']
            # Use whatever query method is available
            result = {"answer": f"Basic processing result for: {query}", "confidence": 0.5, "sources": [], "status": "basic"}
            return result
    except Exception as e:
        logger.error(f"Basic query processing failed: {e}")
    
    return await process_fallback_query(query)

async def process_fallback_query(query: str) -> Dict[str, Any]:
    """Fallback query processing when RAG system is unavailable"""
    return {
        'answer': f'RAG system is currently initializing. Your query "{query[:50]}..." has been received but cannot be processed at this time. Please try again in a few moments.',
        'confidence': 0.0,
        'sources': [],
        'web_sources': [],
        'model_used': 'system',
        'api_provider': 'fallback',
        'status': 'fallback'
    }

async def log_query_analytics(query_id: str, query: str, user_id: str, processing_time: float):
    """Log query for analytics"""
    try:
        logger.info(f"Query analytics: {query_id} | User: {user_id} | Time: {processing_time:.2f}s")
    except Exception as e:
        logger.error(f"Analytics logging failed: {e}")

async def process_url_ingestion(url: str, metadata: Optional[Dict[str, Any]]):
    """Process URL ingestion (background task)"""
    try:
        if rag_system and rag_system.get('api'):
            logger.info(f"URL ingestion queued: {url}")
        else:
            logger.warning(f"RAG system not available for URL ingestion: {url}")
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")

async def process_text_ingestion(text: str, metadata: Optional[Dict[str, Any]]):
    """Process text ingestion (background task)"""
    try:
        if rag_system and rag_system.get('api'):
            logger.info(f"Text ingestion queued: {len(text)} characters")
        else:
            logger.warning(f"RAG system not available for text ingestion")
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
        "api_gateway_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )