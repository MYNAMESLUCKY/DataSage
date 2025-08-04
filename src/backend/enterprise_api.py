"""
Enterprise API Integration for Free GPU Models and Business Logic
Integrates subscription system, free LLMs, Serper search, and business logic protection
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.backend.free_llm_models import get_free_llm_manager
from src.backend.serper_search import get_serper_service
from src.backend.subscription_system import get_subscription_manager
from src.utils.utils import setup_logging

logger = setup_logging(__name__)

app = FastAPI(
    title="Enterprise RAG API",
    description="Scalable RAG system with free GPU models and business logic protection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    user_id: str
    subscription_tier: str = "free"
    enable_gpu_acceleration: bool = True
    max_sources: int = 10
    model_preference: Optional[str] = None
    search_web: bool = True
    response_format: str = "detailed"

class QueryResponse(BaseModel):
    status: str
    answer: str
    sources: List[str] = []
    web_sources: List[str] = []
    processing_time: float
    model_used: str
    gpu_accelerated: bool
    confidence: float
    cost_saved: float
    tokens_used: int
    subscription_info: Dict[str, Any]

class SubscriptionInfo(BaseModel):
    tier: str
    name: str
    status: str
    limits: Dict[str, Any]
    usage_today: Dict[str, Any]
    features: List[str]

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    quality_score: float
    features: List[str]
    context_length: int
    cost_per_token: float

class SystemStatus(BaseModel):
    gpu_providers_available: int
    api_gateway_healthy: bool
    avg_response_time: float
    success_rate: float
    total_models_available: int
    free_tier_usage: float

# Initialize services
free_llm_manager = get_free_llm_manager()
serper_service = get_serper_service()
subscription_manager = get_subscription_manager()

# Initialize simple fallback system for demo
class SimpleFallbackRAG:
    def enhanced_similarity_search(self, vector_store, query, max_sources):
        # Simple fallback that returns mock documents for demo
        return [
            type('Document', (), {
                'page_content': f"Sample knowledge about: {query}",
                'metadata': {'source': f'Knowledge Base {i+1}'}
            })()
            for i in range(min(max_sources, 3))
        ]

simple_rag = SimpleFallbackRAG()

@app.post("/api/v1/query/process", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process user query with business logic protection and GPU acceleration
    """
    start_time = time.time()
    
    try:
        # Get user subscription
        subscription = subscription_manager.get_user_subscription(request.user_id)
        
        # Check access permissions
        access_check = subscription_manager.check_access_permission(
            user_id=request.user_id,
            operation="query",
            resource_amount=1
        )
        
        if not access_check['allowed']:
            raise HTTPException(
                status_code=402 if 'limit exceeded' in access_check['reason'].lower() else 403,
                detail={
                    'error': access_check['reason'],
                    'suggestion': access_check.get('suggestion', ''),
                    'remaining_quota': access_check.get('remaining_quota', {})
                }
            )
        
        # Process with free models and simplified RAG
        result = await _process_with_free_models(request)
        
        # Record usage for billing
        background_tasks.add_task(
            subscription_manager.record_usage,
            request.user_id,
            "query",
            1,
            result.get('cost', 0.0)
        )
        
        # Prepare response
        processing_time = time.time() - start_time
        
        return QueryResponse(
            status=result.get('status', 'success'),
            answer=result.get('answer', ''),
            sources=result.get('sources', []),
            web_sources=result.get('web_sources', []),
            processing_time=result.get('processing_time', processing_time),
            model_used=result.get('model_used', 'free-llm'),
            gpu_accelerated=result.get('gpu_accelerated', False),
            confidence=result.get('confidence', 0.85),
            cost_saved=result.get('cost_saved', 0.0),
            tokens_used=result.get('tokens_used', 0),
            subscription_info={
                'tier': subscription.subscription_tier.value,
                'remaining_quota': access_check.get('remaining_quota', {}),
                'upgrade_available': subscription.subscription_tier.value != 'enterprise'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

async def _process_with_free_models(request: QueryRequest) -> Dict[str, Any]:
    """Process query using free LLM models"""
    
    # Get context from simple knowledge base
    kb_docs = simple_rag.enhanced_similarity_search(
        None,
        request.query,
        request.max_sources
    )
    
    # Get web search results if requested
    web_results = []
    if request.search_web and serper_service.is_available():
        web_results = await serper_service.search(
            query=request.query,
            user_id=request.user_id,
            subscription_tier=request.subscription_tier,
            max_results=5
        )
    
    # Combine context
    context_parts = []
    for doc in kb_docs[:5]:
        content = getattr(doc, 'page_content', str(doc))
        context_parts.append(content[:300])
    
    for result in web_results[:3]:
        context_parts.append(f"{result.title}: {result.snippet}")
    
    context = '\n\n'.join(context_parts)
    
    # Generate response using free LLM
    llm_response = await free_llm_manager.generate_response(
        prompt=f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:",
        user_id=request.user_id,
        subscription_tier=request.subscription_tier,
        model_preference=request.model_preference
    )
    
    if llm_response['status'] != 'success':
        return {
            'status': 'error',
            'answer': llm_response.get('message', 'Failed to generate response'),
            'model_used': 'error'
        }
    
    return {
        'status': 'success',
        'answer': llm_response['text'],
        'sources': [getattr(doc, 'metadata', {}).get('source', f'Document {i+1}') for i, doc in enumerate(kb_docs)],
        'web_sources': [r.url for r in web_results],
        'processing_time': llm_response['processing_time'],
        'model_used': llm_response['model_used'],
        'gpu_accelerated': False,
        'confidence': 0.8,
        'cost_saved': llm_response.get('cost_saved', 0.0),
        'tokens_used': llm_response.get('tokens_used', 0)
    }

@app.get("/api/v1/models/available", response_model=List[ModelInfo])
async def get_available_models(subscription_tier: str = "free"):
    """Get available LLM models for subscription tier"""
    
    models = free_llm_manager.get_available_models(subscription_tier)
    
    return [
        ModelInfo(
            id=model['name'].lower().replace(' ', '_'),
            name=model['name'],
            provider=model['provider'],
            quality_score=model['quality_score'],
            features=model['features'],
            context_length=model['context_length'],
            cost_per_token=0.0  # All models are free
        )
        for model in models
    ]

@app.get("/api/v1/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health metrics"""
    
    # Get GPU infrastructure status
    gpu_status = gpu_processor.get_acceleration_status()
    
    # Get service availability
    models_available = len(free_llm_manager.models)
    serper_available = serper_service.is_available()
    
    return SystemStatus(
        gpu_providers_available=gpu_status['gpu_infrastructure']['providers_configured'],
        api_gateway_healthy=True,
        avg_response_time=2.5,  # Would be calculated from metrics
        success_rate=95.0,      # Would be calculated from metrics
        total_models_available=models_available,
        free_tier_usage=0.65    # Would be calculated from usage
    )

@app.get("/api/v1/subscription/current", response_model=SubscriptionInfo)
async def get_current_subscription(user_id: str):
    """Get user's current subscription details"""
    
    subscription = subscription_manager.get_user_subscription(user_id)
    plan = subscription_manager.plans.get(subscription.plan_id)
    usage = subscription_manager.usage_monitor.get_daily_usage(user_id)
    
    if not plan:
        raise HTTPException(status_code=404, detail="Subscription plan not found")
    
    return SubscriptionInfo(
        tier=subscription.subscription_tier.value,
        name=plan.name,
        status=subscription.status,
        limits=plan.limits,
        usage_today=usage,
        features=plan.features
    )

@app.get("/api/v1/subscription/plans")
async def get_subscription_plans():
    """Get all available subscription plans"""
    return subscription_manager.get_subscription_plans()

@app.post("/api/v1/subscription/upgrade")
async def upgrade_subscription(
    user_id: str,
    plan_id: str,
    payment_method_id: Optional[str] = None
):
    """Upgrade user subscription"""
    
    result = subscription_manager.upgrade_subscription(
        user_id=user_id,
        new_plan_id=plan_id,
        payment_method_id=payment_method_id
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result

@app.get("/api/v1/subscription/usage")
async def get_usage_stats(user_id: str):
    """Get user's current usage statistics"""
    return subscription_manager.usage_monitor.get_daily_usage(user_id)

@app.get("/api/v1/admin/dashboard")
async def get_admin_dashboard():
    """Get admin dashboard analytics"""
    return subscription_manager.get_admin_dashboard_data()

@app.post("/api/v1/search")
async def search_web(
    query: str,
    user_id: str,
    subscription_tier: str = "free",
    max_results: int = 10,
    search_type: str = "search"
):
    """Perform web search using Serper API"""
    
    results = await serper_service.search(
        query=query,
        user_id=user_id,
        subscription_tier=subscription_tier,
        max_results=max_results,
        search_type=search_type
    )
    
    return {
        'results': [
            {
                'title': r.title,
                'url': r.url,
                'snippet': r.snippet,
                'position': r.position,
                'date': r.date,
                'source': r.source
            }
            for r in results
        ],
        'total': len(results)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Enterprise RAG API...")
    logger.info(f"Free LLM models available: {len(free_llm_manager.models)}")
    logger.info(f"Serper search available: {serper_service.is_available()}")
    logger.info(f"GPU infrastructure configured: {gpu_processor.get_acceleration_status()['gpu_infrastructure']['providers_configured']} providers")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enterprise RAG API...")

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.enterprise_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )