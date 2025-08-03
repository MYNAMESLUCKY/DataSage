#!/usr/bin/env python3
"""
API Key Management Endpoints
FastAPI endpoints for API key generation, management, and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .key_management import APIKeyManager, KeyScope, KeyStatus, APIKey

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
key_manager = APIKeyManager()

# Router
router = APIRouter(prefix="/api-keys", tags=["API Key Management"])

# Pydantic Models
class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name for the key")
    description: str = Field(default="", max_length=500, description="Optional description")
    scope: str = Field(default="query_only", description="Access scope for the key")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365, description="Expiry in days (optional)")
    rate_limit: int = Field(default=100, ge=1, le=10000, description="Requests per hour")

class APIKeyResponse(BaseModel):
    key_id: str
    name: str
    description: str
    scope: str
    status: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    rate_limit: int
    rate_window: int

class APIKeyCreateResponse(BaseModel):
    api_key: str
    key_info: APIKeyResponse
    warning: str

class APIKeyUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class APIKeyUsageStats(BaseModel):
    total_requests: int
    successful_requests: int
    success_rate: float
    top_endpoints: List[Dict[str, Any]]
    daily_usage: List[Dict[str, Any]]

# Helper function to verify user token (reuse from main API)
async def verify_user_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    import jwt
    import os
    
    try:
        token = credentials.credentials
        JWT_SECRET = os.getenv('JWT_SECRET', 'enterprise-rag-secret-key-change-in-production')
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
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

def api_key_to_response(api_key: APIKey) -> APIKeyResponse:
    """Convert APIKey object to response model"""
    return APIKeyResponse(
        key_id=api_key.key_id,
        name=api_key.name,
        description=api_key.description,
        scope=api_key.scope.value,
        status=api_key.status.value,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        last_used=api_key.last_used,
        usage_count=api_key.usage_count,
        rate_limit=api_key.rate_limit,
        rate_window=api_key.rate_window
    )

# API Endpoints

@router.post("/generate", response_model=APIKeyCreateResponse)
async def generate_api_key(
    request: APIKeyCreateRequest,
    user: Dict[str, Any] = Depends(verify_user_token)
):
    """Generate a new API key for the authenticated user"""
    try:
        user_id = user.get('sub')
        
        # Validate scope
        try:
            scope = KeyScope(request.scope)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid scope. Valid scopes: {[s.value for s in KeyScope]}"
            )
        
        # Check if user already has too many keys
        existing_keys = key_manager.list_user_keys(user_id)
        active_keys = [k for k in existing_keys if k.status == KeyStatus.ACTIVE]
        
        if len(active_keys) >= 10:  # Limit per user
            raise HTTPException(
                status_code=400,
                detail="Maximum number of active API keys reached (10). Please revoke unused keys."
            )
        
        # Generate the key
        api_key, key_info = key_manager.generate_api_key(
            name=request.name,
            user_id=user_id,
            scope=scope,
            description=request.description,
            expires_in_days=request.expires_in_days,
            rate_limit=request.rate_limit
        )
        
        logger.info(f"Generated API key '{request.name}' for user {user_id}")
        
        return APIKeyCreateResponse(
            api_key=api_key,
            key_info=api_key_to_response(key_info),
            warning="Store this API key securely. It will not be shown again."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate API key")

@router.get("/list", response_model=List[APIKeyResponse])
async def list_api_keys(user: Dict[str, Any] = Depends(verify_user_token)):
    """List all API keys for the authenticated user"""
    try:
        user_id = user.get('sub')
        keys = key_manager.list_user_keys(user_id)
        
        return [api_key_to_response(key) for key in keys]
        
    except Exception as e:
        logger.error(f"Failed to list API keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve API keys")

@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    user: Dict[str, Any] = Depends(verify_user_token)
):
    """Get information about a specific API key"""
    try:
        user_id = user.get('sub')
        key_info = key_manager.get_key_info(key_id)
        
        if not key_info:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Check ownership
        if key_info.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return api_key_to_response(key_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve API key information")

@router.put("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    request: APIKeyUpdateRequest,
    user: Dict[str, Any] = Depends(verify_user_token)
):
    """Update API key metadata"""
    try:
        user_id = user.get('sub')
        
        # Verify ownership
        key_info = key_manager.get_key_info(key_id)
        if not key_info or key_info.user_id != user_id:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Update the key
        success = key_manager.update_key_metadata(
            key_id=key_id,
            user_id=user_id,
            name=request.name,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update API key")
        
        # Return updated key info
        updated_key = key_manager.get_key_info(key_id)
        return api_key_to_response(updated_key)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")

@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: Dict[str, Any] = Depends(verify_user_token)
):
    """Revoke an API key"""
    try:
        user_id = user.get('sub')
        
        success = key_manager.revoke_key(key_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found or already revoked")
        
        return {"message": "API key revoked successfully", "key_id": key_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke API key")

@router.get("/{key_id}/usage", response_model=APIKeyUsageStats)
async def get_key_usage_stats(
    key_id: str,
    days: int = 30,
    user: Dict[str, Any] = Depends(verify_user_token)
):
    """Get usage statistics for an API key"""
    try:
        user_id = user.get('sub')
        
        # Verify ownership
        key_info = key_manager.get_key_info(key_id)
        if not key_info or key_info.user_id != user_id:
            raise HTTPException(status_code=404, detail="API key not found")
        
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        stats = key_manager.get_key_usage_stats(key_id, days)
        
        return APIKeyUsageStats(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get key usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve usage statistics")

@router.post("/{key_id}/regenerate", response_model=APIKeyCreateResponse)
async def regenerate_api_key(
    key_id: str,
    user: Dict[str, Any] = Depends(verify_user_token)
):
    """Regenerate an API key (creates new key, revokes old one)"""
    try:
        user_id = user.get('sub')
        
        # Get existing key info
        old_key = key_manager.get_key_info(key_id)
        if not old_key or old_key.user_id != user_id:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Generate new key with same settings
        new_api_key, new_key_info = key_manager.generate_api_key(
            name=old_key.name,
            user_id=user_id,
            scope=old_key.scope,
            description=old_key.description,
            expires_in_days=None if not old_key.expires_at else 
                            (old_key.expires_at - datetime.utcnow()).days,
            rate_limit=old_key.rate_limit,
            metadata=old_key.metadata
        )
        
        # Revoke old key
        key_manager.revoke_key(key_id, user_id)
        
        logger.info(f"Regenerated API key '{old_key.name}' for user {user_id}")
        
        return APIKeyCreateResponse(
            api_key=new_api_key,
            key_info=api_key_to_response(new_key_info),
            warning="Store this new API key securely. The old key has been revoked."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to regenerate API key")

@router.get("/scopes/available")
async def get_available_scopes():
    """Get list of available API key scopes"""
    return {
        "scopes": [
            {
                "value": scope.value,
                "name": scope.value.replace("_", " ").title(),
                "description": _get_scope_description(scope)
            }
            for scope in KeyScope
        ]
    }

def _get_scope_description(scope: KeyScope) -> str:
    """Get human-readable description for scope"""
    descriptions = {
        KeyScope.READ_ONLY: "Read access to system information and health endpoints",
        KeyScope.QUERY_ONLY: "Query the knowledge base and retrieve answers",
        KeyScope.INGEST_ONLY: "Add new documents and data to the knowledge base",
        KeyScope.FULL_ACCESS: "Full access to query and ingest operations",
        KeyScope.ADMIN: "Administrative access to all system operations"
    }
    return descriptions.get(scope, "Unknown scope")