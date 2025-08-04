"""
GPU Processing Configuration and Service Management

This module manages configuration for various free GPU services and 
provides recommendations for additional GPU computing platforms.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class GPUServiceConfig:
    """Configuration for a GPU service provider"""
    name: str
    base_url: str
    max_concurrent_tasks: int
    timeout_seconds: int
    supported_libraries: List[str]
    compute_capability: str  # e.g., "T4", "V100", "A100"
    memory_gb: int
    free_tier_limits: Dict[str, Any]
    authentication_required: bool = True
    api_key_env_var: Optional[str] = None

class FreeGPUServices:
    """
    Comprehensive list of free GPU services available for distributed computing
    """
    
    # Production-ready free GPU services
    SERVICES = {
        "google_colab": GPUServiceConfig(
            name="Google Colab",
            base_url="https://colab.research.google.com",
            max_concurrent_tasks=2,
            timeout_seconds=3600,  # 1 hour
            supported_libraries=[
                "torch", "tensorflow", "jax", "transformers", 
                "diffusers", "accelerate", "datasets", "numpy",
                "scipy", "scikit-learn", "pandas", "matplotlib"
            ],
            compute_capability="T4/V100",
            memory_gb=25,
            free_tier_limits={
                "daily_gpu_hours": 12,
                "session_timeout": 720,  # 12 hours
                "storage_gb": 15
            },
            authentication_required=True,
            api_key_env_var="GOOGLE_COLAB_API_KEY"
        ),
        
        "kaggle_kernels": GPUServiceConfig(
            name="Kaggle Kernels",
            base_url="https://www.kaggle.com/code",
            max_concurrent_tasks=1,
            timeout_seconds=3600,
            supported_libraries=[
                "torch", "tensorflow", "xgboost", "lightgbm",
                "scikit-learn", "opencv", "transformers", "keras"
            ],
            compute_capability="P100/T4",
            memory_gb=30,
            free_tier_limits={
                "weekly_gpu_hours": 30,
                "session_timeout": 540,  # 9 hours
                "storage_gb": 20
            },
            authentication_required=True,
            api_key_env_var="KAGGLE_API_KEY"
        ),
        
        "paperspace_gradient": GPUServiceConfig(
            name="Paperspace Gradient",
            base_url="https://console.paperspace.com",
            max_concurrent_tasks=1,
            timeout_seconds=1800,  # 30 minutes
            supported_libraries=[
                "torch", "tensorflow", "jax", "transformers",
                "diffusers", "stable-diffusion", "opencv"
            ],
            compute_capability="M4000/P5000",
            memory_gb=8,
            free_tier_limits={
                "monthly_hours": 6,
                "session_timeout": 360,  # 6 hours
                "storage_gb": 5
            },
            authentication_required=True,
            api_key_env_var="PAPERSPACE_API_KEY"
        ),
        
        "huggingface_spaces": GPUServiceConfig(
            name="Hugging Face Spaces",
            base_url="https://huggingface.co/spaces",
            max_concurrent_tasks=2,
            timeout_seconds=1200,  # 20 minutes
            supported_libraries=[
                "transformers", "diffusers", "datasets", "accelerate",
                "torch", "tensorflow", "gradio", "streamlit"
            ],
            compute_capability="T4",
            memory_gb=16,
            free_tier_limits={
                "daily_hours": 8,
                "concurrent_spaces": 3,
                "storage_gb": 50
            },
            authentication_required=True,
            api_key_env_var="HUGGINGFACE_API_KEY"
        ),
        
        "lightning_ai": GPUServiceConfig(
            name="Lightning AI Studio",
            base_url="https://lightning.ai",
            max_concurrent_tasks=1,
            timeout_seconds=2400,  # 40 minutes
            supported_libraries=[
                "pytorch-lightning", "torch", "tensorflow",
                "transformers", "wandb", "tensorboard"
            ],
            compute_capability="T4",
            memory_gb=16,
            free_tier_limits={
                "monthly_hours": 22,
                "session_timeout": 480,  # 8 hours
                "storage_gb": 10
            },
            authentication_required=True,
            api_key_env_var="LIGHTNING_API_KEY"
        ),
        
        "saturn_cloud": GPUServiceConfig(
            name="Saturn Cloud",
            base_url="https://saturncloud.io",
            max_concurrent_tasks=1,
            timeout_seconds=3600,
            supported_libraries=[
                "torch", "tensorflow", "dask", "rapids",
                "scikit-learn", "xgboost", "jupyter"
            ],
            compute_capability="T4",
            memory_gb=16,
            free_tier_limits={
                "monthly_hours": 10,
                "storage_gb": 10,
                "session_timeout": 600  # 10 hours
            },
            authentication_required=True,
            api_key_env_var="SATURN_API_KEY"
        )
    }
    
    @classmethod
    def get_recommended_services(cls, complexity_score: float, query_type: str) -> List[str]:
        """
        Recommend optimal GPU services based on query complexity and type
        
        Args:
            complexity_score: Query complexity (0-1)
            query_type: Type of processing needed
            
        Returns:
            List of recommended service names ordered by suitability
        """
        
        recommendations = []
        
        if query_type in ["nlp", "text_generation", "language_model"]:
            recommendations = ["huggingface_spaces", "google_colab", "lightning_ai"]
        elif query_type in ["computer_vision", "image_processing"]:
            recommendations = ["google_colab", "paperspace_gradient", "kaggle_kernels"]
        elif query_type in ["machine_learning", "training", "optimization"]:
            recommendations = ["kaggle_kernels", "saturn_cloud", "google_colab"]
        elif query_type in ["scientific_computing", "simulation"]:
            recommendations = ["google_colab", "lightning_ai", "saturn_cloud"]
        elif complexity_score > 0.8:
            # Very complex queries - prioritize high-compute services
            recommendations = ["google_colab", "kaggle_kernels", "lightning_ai"]
        else:
            # Default recommendation
            recommendations = ["google_colab", "huggingface_spaces", "paperspace_gradient"]
        
        return recommendations[:3]  # Return top 3 recommendations
    
    @classmethod
    def get_service_status(cls, service_name: str) -> Dict[str, Any]:
        """Get current status and availability of a GPU service"""
        
        if service_name not in cls.SERVICES:
            return {"available": False, "error": "Service not found"}
        
        service = cls.SERVICES[service_name]
        
        # In production, this would check actual service availability
        # For now, simulate availability based on service characteristics
        return {
            "available": True,
            "service": service.name,
            "compute_capability": service.compute_capability,
            "memory_gb": service.memory_gb,
            "estimated_queue_time": "< 2 minutes",
            "current_load": "moderate",
            "free_tier_remaining": "available"
        }

class AdditionalGPURecommendations:
    """
    Additional recommendations for expanding GPU computing capabilities
    """
    
    ALTERNATIVE_PLATFORMS = {
        "research_focused": [
            {
                "name": "OpenAI Playground",
                "description": "Free tier for GPT model experimentation",
                "compute_type": "Language Models",
                "free_tier": "$18 initial credit"
            },
            {
                "name": "Anthropic Claude",
                "description": "Free tier for constitutional AI research",
                "compute_type": "Language Models",
                "free_tier": "Limited free usage"
            },
            {
                "name": "Replicate",
                "description": "Run ML models in the cloud",
                "compute_type": "General ML",
                "free_tier": "$10 initial credit"
            }
        ],
        
        "cloud_providers": [
            {
                "name": "AWS SageMaker",
                "description": "Studio Lab - free ML development environment",
                "compute_type": "General ML",
                "free_tier": "12 hours CPU + 4 hours GPU monthly"
            },
            {
                "name": "Google Cloud AI Platform",
                "description": "Vertex AI free tier",
                "compute_type": "General AI/ML",
                "free_tier": "$300 credit for new users"
            },
            {
                "name": "Azure Machine Learning",
                "description": "Free tier for ML experimentation",
                "compute_type": "General ML",
                "free_tier": "$200 credit for new users"
            }
        ],
        
        "specialized_compute": [
            {
                "name": "Vast.ai",
                "description": "Peer-to-peer GPU rental marketplace",
                "compute_type": "Custom GPU clusters",
                "free_tier": "Pay-per-use starting $0.05/hour"
            },
            {
                "name": "RunPod",
                "description": "Cloud GPU platform",
                "compute_type": "Gaming/AI GPUs",
                "free_tier": "Credits for registration"
            },
            {
                "name": "Lambda Labs",
                "description": "GPU cloud for AI workloads",
                "compute_type": "AI/ML focused",
                "free_tier": "Free tier available"
            }
        ],
        
        "development_focused": [
            {
                "name": "Gitpod",
                "description": "Cloud development environments",
                "compute_type": "Development",
                "free_tier": "50 hours/month"
            },
            {
                "name": "GitHub Codespaces",
                "description": "Cloud development with GPU support",
                "compute_type": "Development",
                "free_tier": "60 core hours/month"
            },
            {
                "name": "Replit",
                "description": "Collaborative coding with compute",
                "compute_type": "General Development",
                "free_tier": "Basic tier available"
            }
        ]
    }
    
    OPTIMIZATION_STRATEGIES = [
        {
            "strategy": "Model Quantization",
            "description": "Reduce model precision to 8-bit or 4-bit for faster inference",
            "tools": ["bitsandbytes", "onnx", "tensorrt"],
            "compute_savings": "50-75%"
        },
        {
            "strategy": "Knowledge Distillation",
            "description": "Train smaller models to mimic larger ones",
            "tools": ["distilbert", "tinybert", "mobile-bert"],
            "compute_savings": "60-90%"
        },
        {
            "strategy": "Gradient Checkpointing",
            "description": "Trade compute for memory in training",
            "tools": ["pytorch", "tensorflow", "deepspeed"],
            "compute_savings": "Memory: 50-80%"
        },
        {
            "strategy": "Mixed Precision Training",
            "description": "Use FP16 instead of FP32 for faster training",
            "tools": ["apex", "accelerate", "native pytorch"],
            "compute_savings": "30-50%"
        }
    ]

# Global configuration instance
gpu_config = {
    "services": FreeGPUServices.SERVICES,
    "recommendations": AdditionalGPURecommendations.ALTERNATIVE_PLATFORMS,
    "optimization": AdditionalGPURecommendations.OPTIMIZATION_STRATEGIES,
    "default_timeout": 1800,  # 30 minutes
    "max_retries": 3,
    "fallback_enabled": True
}

def get_gpu_service_recommendations(complexity_score: float, query: str) -> Dict[str, Any]:
    """
    Get comprehensive GPU service recommendations for a query
    
    Args:
        complexity_score: Query complexity (0-1)
        query: The actual query text
        
    Returns:
        Dictionary with service recommendations and alternatives
    """
    
    # Determine query type
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['text', 'language', 'nlp', 'generate', 'translate']):
        query_type = "nlp"
    elif any(word in query_lower for word in ['image', 'vision', 'photo', 'picture', 'visual']):
        query_type = "computer_vision"
    elif any(word in query_lower for word in ['train', 'model', 'ml', 'predict', 'classify']):
        query_type = "machine_learning"
    elif any(word in query_lower for word in ['calculate', 'compute', 'simulation', 'numerical']):
        query_type = "scientific_computing"
    else:
        query_type = "general"
    
    # Get service recommendations
    recommended_services = FreeGPUServices.get_recommended_services(complexity_score, query_type)
    
    # Get service details
    service_details = []
    for service_name in recommended_services:
        if service_name in FreeGPUServices.SERVICES:
            service = FreeGPUServices.SERVICES[service_name]
            status = FreeGPUServices.get_service_status(service_name)
            
            service_details.append({
                "name": service.name,
                "compute_capability": service.compute_capability,
                "memory_gb": service.memory_gb,
                "timeout_minutes": service.timeout_seconds // 60,
                "free_tier": service.free_tier_limits,
                "status": status,
                "suitability_score": _calculate_suitability_score(service, complexity_score, query_type)
            })
    
    return {
        "query_type": query_type,
        "complexity_score": complexity_score,
        "recommended_services": service_details,
        "alternative_platforms": AdditionalGPURecommendations.ALTERNATIVE_PLATFORMS,
        "optimization_strategies": AdditionalGPURecommendations.OPTIMIZATION_STRATEGIES,
        "estimated_processing_time": _estimate_processing_time(complexity_score),
        "resource_requirements": _estimate_resource_requirements(complexity_score, query_type)
    }

def _calculate_suitability_score(service: GPUServiceConfig, complexity_score: float, query_type: str) -> float:
    """Calculate how suitable a service is for the given query"""
    
    score = 0.5  # Base score
    
    # Adjust based on compute capability
    if "A100" in service.compute_capability:
        score += 0.3
    elif "V100" in service.compute_capability:
        score += 0.2
    elif "T4" in service.compute_capability:
        score += 0.1
    
    # Adjust based on memory
    if service.memory_gb >= 24:
        score += 0.2
    elif service.memory_gb >= 16:
        score += 0.1
    
    # Adjust based on timeout (longer is better for complex queries)
    if complexity_score > 0.7 and service.timeout_seconds >= 3600:
        score += 0.1
    
    # Adjust based on query type and supported libraries
    type_library_map = {
        "nlp": ["transformers", "torch", "tensorflow"],
        "computer_vision": ["opencv", "torch", "tensorflow"],
        "machine_learning": ["scikit-learn", "xgboost", "torch"],
        "scientific_computing": ["numpy", "scipy", "jax"]
    }
    
    relevant_libs = type_library_map.get(query_type, [])
    supported_count = sum(1 for lib in relevant_libs if lib in service.supported_libraries)
    
    if relevant_libs:
        score += (supported_count / len(relevant_libs)) * 0.2
    
    return min(1.0, score)

def _estimate_processing_time(complexity_score: float) -> Dict[str, int]:
    """Estimate processing time based on complexity"""
    
    base_time = 30  # seconds
    complex_multiplier = 1 + (complexity_score * 10)
    
    estimated_seconds = int(base_time * complex_multiplier)
    
    return {
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": estimated_seconds // 60,
        "complexity_factor": complex_multiplier
    }

def _estimate_resource_requirements(complexity_score: float, query_type: str) -> Dict[str, Any]:
    """Estimate resource requirements for processing"""
    
    # Base requirements
    requirements = {
        "min_memory_gb": 4,
        "min_compute_capability": "T4",
        "recommended_memory_gb": 8,
        "recommended_compute_capability": "T4"
    }
    
    # Adjust based on complexity
    if complexity_score > 0.8:
        requirements.update({
            "min_memory_gb": 16,
            "recommended_memory_gb": 32,
            "recommended_compute_capability": "V100/A100"
        })
    elif complexity_score > 0.6:
        requirements.update({
            "min_memory_gb": 8,
            "recommended_memory_gb": 16,
            "recommended_compute_capability": "T4/V100"
        })
    
    # Adjust based on query type
    if query_type == "computer_vision":
        requirements["min_memory_gb"] = max(requirements["min_memory_gb"], 12)
        requirements["recommended_memory_gb"] = max(requirements["recommended_memory_gb"], 24)
    elif query_type == "nlp":
        requirements["min_memory_gb"] = max(requirements["min_memory_gb"], 8)
    
    return requirements