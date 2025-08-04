"""
Free LLM Models Integration for Enterprise RAG System
Integrates Hugging Face Inference API, Llama, Mistral and other free models
"""

import os
import time
import logging
import requests
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

@dataclass
class FreeLLMModel:
    """Configuration for free LLM models"""
    name: str
    model_id: str
    provider: str
    max_tokens: int
    context_length: int
    cost_per_1k_tokens: float
    rate_limit_rpm: int
    features: List[str]
    quality_score: float

class FreeLLMManager:
    """
    Manages free LLM models from various providers
    Includes business logic to prevent resource abuse
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.usage_tracker = LLMUsageTracker()
        self.abuse_prevention = AbusePreventionSystem()
        self.model_selector = ModelSelector()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def _initialize_models(self) -> List[FreeLLMModel]:
        """Initialize available free LLM models"""
        models = [
            # Hugging Face Free Models
            FreeLLMModel(
                name="Llama 3.2 7B",
                model_id="meta-llama/Llama-3.2-7B-Instruct",
                provider="huggingface",
                max_tokens=4096,
                context_length=8192,
                cost_per_1k_tokens=0.0,  # Free tier
                rate_limit_rpm=20,
                features=["instruction_following", "conversation", "code"],
                quality_score=8.5
            ),
            FreeLLMModel(
                name="Mistral 7B Instruct",
                model_id="mistralai/Mistral-7B-Instruct-v0.3",
                provider="huggingface",
                max_tokens=4096,
                context_length=8192,
                cost_per_1k_tokens=0.0,
                rate_limit_rpm=25,
                features=["instruction_following", "multilingual", "fast"],
                quality_score=8.7
            ),
            FreeLLMModel(
                name="Gemma 2 7B",
                model_id="google/gemma-2-7b-it",
                provider="huggingface",
                max_tokens=4096,
                context_length=8192,
                cost_per_1k_tokens=0.0,
                rate_limit_rpm=15,
                features=["safety_focused", "factual", "google_trained"],
                quality_score=8.3
            ),
            FreeLLMModel(
                name="Qwen 2.5 7B",
                model_id="Qwen/Qwen2.5-7B-Instruct",
                provider="huggingface",
                max_tokens=4096,
                context_length=32768,  # Longer context
                cost_per_1k_tokens=0.0,
                rate_limit_rpm=20,
                features=["long_context", "multilingual", "reasoning"],
                quality_score=8.6
            ),
            FreeLLMModel(
                name="DeepSeek Coder 7B",
                model_id="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                provider="huggingface",
                max_tokens=4096,
                context_length=16384,
                cost_per_1k_tokens=0.0,
                rate_limit_rpm=15,
                features=["code_generation", "debugging", "technical"],
                quality_score=8.8
            ),
            # Together AI Free Models
            FreeLLMModel(
                name="Llama 3.1 8B",
                model_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                provider="together",
                max_tokens=4096,
                context_length=8192,
                cost_per_1k_tokens=0.18,  # Very cheap
                rate_limit_rpm=60,
                features=["fast_inference", "latest_model", "balanced"],
                quality_score=9.0
            ),
            FreeLLMModel(
                name="Mixtral 8x7B",
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                provider="together",
                max_tokens=4096,
                context_length=32768,
                cost_per_1k_tokens=0.6,
                rate_limit_rpm=30,
                features=["mixture_of_experts", "high_quality", "long_context"],
                quality_score=9.2
            ),
            # SARVAM API Models
            FreeLLMModel(
                name="SARVAM-M",
                model_id="sarvam-m",
                provider="sarvam",
                max_tokens=4096,
                context_length=8192,
                cost_per_1k_tokens=0.001,  # Very low cost
                rate_limit_rpm=60,
                features=["fast_inference", "multilingual", "indian_languages"],
                quality_score=8.6
            )
        ]
        
        # Filter available models based on API keys
        available_models = []
        for model in models:
            if self._check_provider_availability(model.provider):
                available_models.append(model)
                logger.info(f"Free LLM '{model.name}' available via {model.provider}")
            else:
                logger.debug(f"Provider '{model.provider}' not configured for model {model.name}")
        
        if not available_models:
            logger.warning("No free LLM providers configured - add API keys to enable models")
        
        return available_models
    
    def _check_provider_availability(self, provider: str) -> bool:
        """Check if provider API key is available"""
        provider_keys = {
            'huggingface': 'HUGGINGFACE_API_KEY',
            'together': 'TOGETHER_API_KEY',
            'openrouter': 'OPENROUTER_API',
            'groq': 'GROQ_API_KEY',
            'sarvam': 'SARVAM_API'
        }
        
        key_name = provider_keys.get(provider)
        return key_name and os.getenv(key_name) is not None
    
    async def generate_response(
        self, 
        prompt: str, 
        user_id: str,
        subscription_tier: str = "free",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response with business logic protection
        """
        start_time = time.time()
        
        try:
            # Business logic checks
            abuse_check = await self.abuse_prevention.check_request(
                user_id=user_id,
                prompt=prompt,
                subscription_tier=subscription_tier
            )
            
            if not abuse_check['allowed']:
                return {
                    'status': 'blocked',
                    'message': abuse_check['reason'],
                    'suggestions': abuse_check.get('suggestions', []),
                    'retry_after': abuse_check.get('retry_after')
                }
            
            # Select best model based on request and availability
            selected_model = await self.model_selector.select_optimal_model(
                models=self.models,
                prompt=prompt,
                subscription_tier=subscription_tier,
                preference=model_preference
            )
            
            if not selected_model:
                return {
                    'status': 'error',
                    'message': 'No suitable models available',
                    'suggestion': 'Try again later or upgrade subscription'
                }
            
            # Generate response using selected model
            response = await self._call_model(
                model=selected_model,
                prompt=prompt,
                max_tokens=max_tokens or selected_model.max_tokens,
                temperature=temperature
            )
            
            # Track usage for billing and abuse prevention
            processing_time = time.time() - start_time
            await self.usage_tracker.record_usage(
                user_id=user_id,
                model=selected_model,
                input_tokens=len(prompt.split()) * 1.3,  # Rough estimate
                output_tokens=len(response.get('text', '').split()) * 1.3,
                processing_time=processing_time,
                subscription_tier=subscription_tier
            )
            
            return {
                'status': 'success',
                'text': response.get('text', ''),
                'model_used': selected_model.name,
                'provider': selected_model.provider,
                'tokens_used': response.get('tokens_used', 0),
                'processing_time': processing_time,
                'quality_score': selected_model.quality_score,
                'cost_saved': self._calculate_cost_savings(selected_model, response.get('tokens_used', 0))
            }
            
        except Exception as e:
            logger.error(f"Free LLM generation failed: {e}")
            return {
                'status': 'error',
                'message': f'Generation failed: {str(e)}',
                'model_attempted': selected_model.name if 'selected_model' in locals() else 'unknown'
            }
    
    async def _call_model(
        self, 
        model: FreeLLMModel, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Call specific model based on provider"""
        
        if model.provider == "huggingface":
            return await self._call_huggingface(model, prompt, max_tokens, temperature)
        elif model.provider == "together":
            return await self._call_together(model, prompt, max_tokens, temperature)
        elif model.provider == "openrouter":
            return await self._call_openrouter(model, prompt, max_tokens, temperature)
        elif model.provider == "sarvam":
            return await self._call_sarvam(model, prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")
    
    async def _call_huggingface(
        self, 
        model: FreeLLMModel, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Call Hugging Face Inference API"""
        
        api_token = os.getenv("HUGGINGFACE_API_KEY")
        if not api_token:
            raise ValueError("HUGGINGFACE_API_KEY not found")
        
        url = f"https://api-inference.huggingface.co/models/{model.model_id}"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Use asyncio with requests in executor for non-blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(url, headers=headers, json=payload, timeout=30)
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
                return {
                    'text': text,
                    'tokens_used': len(text.split()) * 1.3  # Rough estimate
                }
            else:
                return {'text': str(result), 'tokens_used': 50}
        else:
            raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
    
    async def _call_together(
        self, 
        model: FreeLLMModel, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Call Together AI API"""
        
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found")
        
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(url, headers=headers, json=payload, timeout=30)
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['choices'][0]['message']['content']
            return {
                'text': text,
                'tokens_used': result.get('usage', {}).get('total_tokens', len(text.split()) * 1.3)
            }
        else:
            raise Exception(f"Together AI API error: {response.status_code} - {response.text}")
    
    async def _call_openrouter(
        self, 
        model: FreeLLMModel, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Call OpenRouter API for free models"""
        
        api_key = os.getenv("OPENROUTER_API")
        if not api_key:
            raise ValueError("OPENROUTER_API not found")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Enterprise RAG System"
        }
        
        payload = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(url, headers=headers, json=payload, timeout=30)
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['choices'][0]['message']['content']
            return {
                'text': text,
                'tokens_used': result.get('usage', {}).get('total_tokens', len(text.split()) * 1.3)
            }
        else:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
            
    async def _call_sarvam(
        self, 
        model: FreeLLMModel, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Call SARVAM API for text generation"""
        
        api_key = os.getenv("SARVAM_API")
        if not api_key:
            raise ValueError("SARVAM_API not found")
        
        url = "https://api.sarvam.ai/v1/chat/completions"
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(url, headers=headers, json=payload, timeout=30)
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            return {
                'text': text,
                'tokens_used': result.get('usage', {}).get('total_tokens', len(text.split()) * 1.3)
            }
        else:
            raise Exception(f"SARVAM API error: {response.status_code} - {response.text}")
    
    def _calculate_cost_savings(self, model: FreeLLMModel, tokens_used: int) -> float:
        """Calculate cost savings compared to premium models"""
        # Assume premium models cost ~$0.03 per 1K tokens
        premium_cost = (tokens_used / 1000) * 0.03
        actual_cost = (tokens_used / 1000) * model.cost_per_1k_tokens
        return premium_cost - actual_cost
    
    def get_available_models(self, subscription_tier: str = "free") -> List[Dict[str, Any]]:
        """Get list of available models for subscription tier"""
        available = []
        for model in self.models:
            model_info = {
                'name': model.name,
                'provider': model.provider,
                'quality_score': model.quality_score,
                'features': model.features,
                'context_length': model.context_length,
                'rate_limit': model.rate_limit_rpm
            }
            
            # Filter based on subscription tier
            if subscription_tier == "free" and model.cost_per_1k_tokens == 0:
                available.append(model_info)
            elif subscription_tier in ["pro", "enterprise"]:
                available.append(model_info)
        
        return sorted(available, key=lambda x: x['quality_score'], reverse=True)


class ModelSelector:
    """Intelligently selects the best model for each request"""
    
    async def select_optimal_model(
        self, 
        models: List[FreeLLMModel], 
        prompt: str, 
        subscription_tier: str,
        preference: Optional[str] = None
    ) -> Optional[FreeLLMModel]:
        """Select optimal model based on request characteristics"""
        
        if not models:
            return None
        
        # Filter by subscription tier
        eligible_models = [
            m for m in models 
            if subscription_tier == "free" and m.cost_per_1k_tokens == 0 
            or subscription_tier in ["pro", "enterprise"]
        ]
        
        if not eligible_models:
            return None
        
        # Handle explicit preference
        if preference:
            for model in eligible_models:
                if preference.lower() in model.name.lower():
                    return model
        
        # Score models based on prompt characteristics
        scored_models = []
        for model in eligible_models:
            score = self._calculate_model_score(model, prompt)
            scored_models.append((model, score))
        
        # Return highest scoring model
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0]
    
    def _calculate_model_score(self, model: FreeLLMModel, prompt: str) -> float:
        """Calculate model fitness score for given prompt"""
        score = model.quality_score  # Base score
        
        # Prompt length considerations
        prompt_length = len(prompt)
        if prompt_length > 2000 and model.context_length >= 16384:
            score += 1.0  # Bonus for long context models
        elif prompt_length > 4000 and model.context_length >= 32768:
            score += 2.0  # Bigger bonus for very long context
        
        # Content type analysis
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['code', 'programming', 'debug', 'function']):
            if 'code_generation' in model.features:
                score += 1.5
        
        if any(word in prompt_lower for word in ['reasoning', 'analysis', 'logic', 'math']):
            if 'reasoning' in model.features:
                score += 1.0
        
        if len(prompt_lower.encode('utf-8')) != len(prompt_lower):  # Non-ASCII characters
            if 'multilingual' in model.features:
                score += 0.5
        
        # Provider reliability bonus
        if model.provider == "huggingface":
            score += 0.3  # Stable free tier
        elif model.provider == "together":
            score += 0.5  # Fast inference
        
        return score


class LLMUsageTracker:
    """Tracks LLM usage for billing and analytics"""
    
    def __init__(self):
        self.usage_data = {}
    
    async def record_usage(
        self, 
        user_id: str, 
        model: FreeLLMModel, 
        input_tokens: float,
        output_tokens: float, 
        processing_time: float,
        subscription_tier: str
    ):
        """Record usage metrics"""
        
        if user_id not in self.usage_data:
            self.usage_data[user_id] = {
                'total_requests': 0,
                'total_tokens': 0,
                'models_used': {},
                'subscription_tier': subscription_tier,
                'monthly_usage': {},
                'cost_saved': 0.0
            }
        
        user_data = self.usage_data[user_id]
        user_data['total_requests'] += 1
        user_data['total_tokens'] += (input_tokens + output_tokens)
        
        if model.name not in user_data['models_used']:
            user_data['models_used'][model.name] = 0
        user_data['models_used'][model.name] += 1
        
        # Calculate cost savings
        tokens_used = input_tokens + output_tokens
        premium_cost = (tokens_used / 1000) * 0.03
        actual_cost = (tokens_used / 1000) * model.cost_per_1k_tokens
        user_data['cost_saved'] += (premium_cost - actual_cost)
        
        logger.info(f"Usage recorded for {user_id}: {model.name}, {tokens_used:.0f} tokens")
    
    def get_user_usage(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        return self.usage_data.get(user_id, {})


class AbusePreventionSystem:
    """Prevents resource abuse and implements business logic"""
    
    def __init__(self):
        self.rate_limits = {
            'free': {'requests_per_hour': 50, 'tokens_per_day': 50000},
            'pro': {'requests_per_hour': 500, 'tokens_per_day': 500000},
            'enterprise': {'requests_per_hour': 5000, 'tokens_per_day': 5000000}
        }
        self.user_tracking = {}
    
    async def check_request(
        self, 
        user_id: str, 
        prompt: str, 
        subscription_tier: str
    ) -> Dict[str, Any]:
        """Check if request is allowed based on business logic"""
        
        current_time = time.time()
        
        # Initialize user tracking
        if user_id not in self.user_tracking:
            self.user_tracking[user_id] = {
                'requests_today': 0,
                'tokens_today': 0,
                'last_request': 0,
                'flagged_content_count': 0,
                'subscription_tier': subscription_tier
            }
        
        user_data = self.user_tracking[user_id]
        
        # Rate limiting check
        limits = self.rate_limits.get(subscription_tier, self.rate_limits['free'])
        
        # Reset daily counters if needed
        if current_time - user_data['last_request'] > 86400:  # 24 hours
            user_data['requests_today'] = 0
            user_data['tokens_today'] = 0
        
        # Check hourly request limit
        if user_data['requests_today'] >= limits['requests_per_hour']:
            return {
                'allowed': False,
                'reason': f'Rate limit exceeded: {limits["requests_per_hour"]} requests/hour for {subscription_tier} tier',
                'retry_after': 3600,  # 1 hour
                'suggestions': ['Upgrade subscription', 'Wait and try again']
            }
        
        # Check daily token limit
        estimated_tokens = len(prompt.split()) * 2  # Rough estimate including response
        if user_data['tokens_today'] + estimated_tokens > limits['tokens_per_day']:
            return {
                'allowed': False,
                'reason': f'Daily token limit exceeded: {limits["tokens_per_day"]} tokens/day for {subscription_tier} tier',
                'retry_after': 86400,  # 24 hours
                'suggestions': ['Upgrade subscription', 'Use shorter prompts']
            }
        
        # Content safety check
        if self._is_harmful_content(prompt):
            user_data['flagged_content_count'] += 1
            if user_data['flagged_content_count'] > 3:
                return {
                    'allowed': False,
                    'reason': 'Account flagged for repeated policy violations',
                    'suggestions': ['Review terms of service', 'Contact support']
                }
            return {
                'allowed': False,
                'reason': 'Content violates usage policy',
                'suggestions': ['Modify your request', 'Review content guidelines']
            }
        
        # Update tracking
        user_data['requests_today'] += 1
        user_data['tokens_today'] += estimated_tokens
        user_data['last_request'] = current_time
        
        return {
            'allowed': True,
            'remaining_requests': limits['requests_per_hour'] - user_data['requests_today'],
            'remaining_tokens': limits['tokens_per_day'] - user_data['tokens_today']
        }
    
    def _is_harmful_content(self, prompt: str) -> bool:
        """Basic content safety check"""
        harmful_keywords = [
            'illegal', 'hack', 'crack', 'piracy', 'bomb', 'weapon',
            'drug', 'violence', 'hate', 'discrimination', 'suicide'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in harmful_keywords)


# Global instance
_free_llm_manager = None

def get_free_llm_manager() -> FreeLLMManager:
    """Get global free LLM manager instance"""
    global _free_llm_manager
    if _free_llm_manager is None:
        _free_llm_manager = FreeLLMManager()
    return _free_llm_manager