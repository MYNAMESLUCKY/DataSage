"""
Serper Dev API Integration for Web Search
High-performance Google Search API with rate limiting and business logic
"""

import os
import time
import logging
import requests
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

@dataclass
class SearchResult:
    """Structured search result from Serper API"""
    title: str
    url: str
    snippet: str
    position: int
    date: Optional[str] = None
    source: Optional[str] = None

class SerperSearchService:
    """
    Serper Dev API service with business logic protection
    Provides fast, cost-effective web search with abuse prevention
    """
    
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev"
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Rate limiting and usage tracking
        self.rate_limiter = SerperRateLimiter()
        self.usage_tracker = SerperUsageTracker()
        self.abuse_prevention = SearchAbusePreventionSystem()
        
        # Pricing configuration (based on 2025 rates)
        self.pricing_tiers = {
            'free': {'queries_per_day': 2500, 'cost_per_query': 0.0},
            'paid': {'queries_per_1k': 1000, 'cost_per_query': 0.001}  # $1 per 1000 queries
        }
        
        if not self.api_key:
            logger.warning("SERPER_API_KEY not found - web search will be unavailable")
        else:
            logger.info("Serper search service initialized successfully")
    
    def is_available(self) -> bool:
        """Check if Serper API is available"""
        return self.api_key is not None
    
    async def search(
        self, 
        query: str, 
        user_id: str,
        subscription_tier: str = "free",
        max_results: int = 10,
        search_type: str = "search",
        location: Optional[str] = None,
        time_range: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform web search with business logic protection
        
        Args:
            query: Search query
            user_id: User identifier for tracking
            subscription_tier: User's subscription level
            max_results: Maximum results to return
            search_type: Type of search (search, news, images, videos)
            location: Geographic location for search
            time_range: Time range filter (day, week, month, year)
        """
        start_time = time.time()
        
        try:
            # Business logic checks
            abuse_check = await self.abuse_prevention.check_search_request(
                user_id=user_id,
                query=query,
                subscription_tier=subscription_tier
            )
            
            if not abuse_check['allowed']:
                logger.warning(f"Search blocked for user {user_id}: {abuse_check['reason']}")
                return []
            
            # Rate limiting check
            rate_limit_check = await self.rate_limiter.check_rate_limit(user_id, subscription_tier)
            if not rate_limit_check['allowed']:
                logger.warning(f"Rate limit exceeded for user {user_id}")
                return []
            
            # Perform the actual search
            results = await self._perform_search(
                query=query,
                max_results=max_results,
                search_type=search_type,
                location=location,
                time_range=time_range
            )
            
            # Track usage
            processing_time = time.time() - start_time
            await self.usage_tracker.record_usage(
                user_id=user_id,
                query=query,
                results_count=len(results),
                processing_time=processing_time,
                subscription_tier=subscription_tier,
                search_type=search_type
            )
            
            logger.info(f"Search completed for user {user_id}: {len(results)} results in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for user {user_id}: {e}")
            return []
    
    async def _perform_search(
        self, 
        query: str, 
        max_results: int,
        search_type: str,
        location: Optional[str],
        time_range: Optional[str]
    ) -> List[SearchResult]:
        """Perform the actual search API call"""
        
        if not self.api_key:
            raise ValueError("Serper API key not configured")
        
        # Build request payload
        payload = {
            "q": query,
            "num": min(max_results, 20),  # Serper max is 20
            "type": search_type
        }
        
        if location:
            payload["location"] = location
        
        if time_range:
            payload["tbs"] = f"qdr:{time_range[0]}"  # day=d, week=w, month=m, year=y
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Use async execution
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            lambda: requests.post(
                f"{self.base_url}/{search_type}",
                headers=headers,
                json=payload,
                timeout=10
            )
        )
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_search_results(data, search_type)
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded - too many requests")
        elif response.status_code == 403:
            raise Exception("API key invalid or quota exceeded")
        else:
            raise Exception(f"Search API error: {response.status_code} - {response.text}")
    
    def _parse_search_results(self, data: Dict[str, Any], search_type: str) -> List[SearchResult]:
        """Parse Serper API response into structured results"""
        results = []
        
        if search_type == "search":
            organic_results = data.get("organic", [])
            for i, result in enumerate(organic_results):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    position=i + 1,
                    date=result.get("date"),
                    source=result.get("source")
                )
                results.append(search_result)
        
        elif search_type == "news":
            news_results = data.get("news", [])
            for i, result in enumerate(news_results):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    position=i + 1,
                    date=result.get("date"),
                    source=result.get("source")
                )
                results.append(search_result)
        
        elif search_type == "images":
            image_results = data.get("images", [])
            for i, result in enumerate(image_results):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("imageUrl", ""),
                    snippet=f"Image from {result.get('source', 'unknown')}",
                    position=i + 1,
                    source=result.get("source")
                )
                results.append(search_result)
        
        return results
    
    async def get_search_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Get search suggestions for query completion"""
        try:
            payload = {"q": query}
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.post(
                    f"{self.base_url}/autocomplete",
                    headers=headers,
                    json=payload,
                    timeout=5
                )
            )
            
            if response.status_code == 200:
                data = response.json()
                suggestions = data.get("suggestions", [])
                return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.warning(f"Search suggestions failed: {e}")
        
        return []
    
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user's search usage statistics"""
        return self.usage_tracker.get_user_stats(user_id)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and health metrics"""
        return {
            'available': self.is_available(),
            'api_configured': self.api_key is not None,
            'pricing_tiers': self.pricing_tiers,
            'rate_limiter_status': self.rate_limiter.get_status(),
            'total_users': len(self.usage_tracker.user_stats)
        }


class SerperRateLimiter:
    """Rate limiting for Serper API to prevent abuse"""
    
    def __init__(self):
        self.rate_limits = {
            'free': {'requests_per_minute': 5, 'requests_per_hour': 50, 'requests_per_day': 200},
            'pro': {'requests_per_minute': 20, 'requests_per_hour': 500, 'requests_per_day': 2000},
            'enterprise': {'requests_per_minute': 100, 'requests_per_hour': 2000, 'requests_per_day': 10000}
        }
        self.user_requests = {}
    
    async def check_rate_limit(self, user_id: str, subscription_tier: str) -> Dict[str, Any]:
        """Check if user can make a search request"""
        current_time = time.time()
        limits = self.rate_limits.get(subscription_tier, self.rate_limits['free'])
        
        if user_id not in self.user_requests:
            self.user_requests[user_id] = {
                'requests_this_minute': [],
                'requests_this_hour': [],
                'requests_this_day': []
            }
        
        user_data = self.user_requests[user_id]
        
        # Clean old requests
        user_data['requests_this_minute'] = [
            t for t in user_data['requests_this_minute'] 
            if current_time - t < 60
        ]
        user_data['requests_this_hour'] = [
            t for t in user_data['requests_this_hour'] 
            if current_time - t < 3600
        ]
        user_data['requests_this_day'] = [
            t for t in user_data['requests_this_day'] 
            if current_time - t < 86400
        ]
        
        # Check limits
        if len(user_data['requests_this_minute']) >= limits['requests_per_minute']:
            return {
                'allowed': False,
                'reason': f'Minute limit exceeded: {limits["requests_per_minute"]} searches/minute',
                'retry_after': 60
            }
        
        if len(user_data['requests_this_hour']) >= limits['requests_per_hour']:
            return {
                'allowed': False,
                'reason': f'Hour limit exceeded: {limits["requests_per_hour"]} searches/hour',
                'retry_after': 3600
            }
        
        if len(user_data['requests_this_day']) >= limits['requests_per_day']:
            return {
                'allowed': False,
                'reason': f'Daily limit exceeded: {limits["requests_per_day"]} searches/day',
                'retry_after': 86400
            }
        
        # Record this request
        user_data['requests_this_minute'].append(current_time)
        user_data['requests_this_hour'].append(current_time)
        user_data['requests_this_day'].append(current_time)
        
        return {
            'allowed': True,
            'remaining_minute': limits['requests_per_minute'] - len(user_data['requests_this_minute']),
            'remaining_hour': limits['requests_per_hour'] - len(user_data['requests_this_hour']),
            'remaining_day': limits['requests_per_day'] - len(user_data['requests_this_day'])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        return {
            'total_users_tracked': len(self.user_requests),
            'rate_limits': self.rate_limits
        }


class SerperUsageTracker:
    """Track search usage for billing and analytics"""
    
    def __init__(self):
        self.user_stats = {}
    
    async def record_usage(
        self, 
        user_id: str, 
        query: str, 
        results_count: int,
        processing_time: float, 
        subscription_tier: str,
        search_type: str
    ):
        """Record search usage"""
        
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                'total_searches': 0,
                'total_results': 0,
                'avg_processing_time': 0,
                'search_types': {},
                'subscription_tier': subscription_tier,
                'cost_incurred': 0.0,
                'searches_today': 0,
                'last_search': 0
            }
        
        user_data = self.user_stats[user_id]
        
        # Reset daily counter if needed
        current_time = time.time()
        if current_time - user_data['last_search'] > 86400:  # 24 hours
            user_data['searches_today'] = 0
        
        # Update statistics
        user_data['total_searches'] += 1
        user_data['searches_today'] += 1
        user_data['total_results'] += results_count
        user_data['last_search'] = current_time
        
        # Update average processing time
        total_time = user_data['avg_processing_time'] * (user_data['total_searches'] - 1)
        user_data['avg_processing_time'] = (total_time + processing_time) / user_data['total_searches']
        
        # Track search types
        if search_type not in user_data['search_types']:
            user_data['search_types'][search_type] = 0
        user_data['search_types'][search_type] += 1
        
        # Calculate cost (if applicable)
        if subscription_tier != 'free':
            user_data['cost_incurred'] += 0.001  # $0.001 per search
        
        logger.debug(f"Usage recorded for {user_id}: {query[:50]}...")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        return self.user_stats.get(user_id, {})


class SearchAbusePreventionSystem:
    """Prevent search abuse and implement business logic"""
    
    def __init__(self):
        self.blocked_patterns = [
            # Spam patterns
            r'(.)\1{10,}',  # Repeated characters
            r'\b(test|spam|hello)\b.*\1',  # Repeated words
            # Harmful patterns
            r'\b(hack|crack|illegal|piracy)\b',
            # Resource waste patterns
            r'^.{200,}$',  # Very long queries
        ]
        self.user_violations = {}
    
    async def check_search_request(
        self, 
        user_id: str, 
        query: str, 
        subscription_tier: str
    ) -> Dict[str, Any]:
        """Check if search request should be allowed"""
        
        # Initialize user tracking
        if user_id not in self.user_violations:
            self.user_violations[user_id] = {
                'violation_count': 0,
                'last_violation': 0,
                'blocked_until': 0
            }
        
        user_data = self.user_violations[user_id]
        current_time = time.time()
        
        # Check if user is currently blocked
        if current_time < user_data['blocked_until']:
            return {
                'allowed': False,
                'reason': 'Account temporarily blocked due to policy violations',
                'blocked_until': user_data['blocked_until']
            }
        
        # Basic query validation
        if len(query.strip()) < 2:
            return {
                'allowed': False,
                'reason': 'Query too short - minimum 2 characters required'
            }
        
        if len(query) > 500:
            return {
                'allowed': False,
                'reason': 'Query too long - maximum 500 characters allowed'
            }
        
        # Pattern-based abuse detection
        import re
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                user_data['violation_count'] += 1
                user_data['last_violation'] = current_time
                
                # Progressive penalties
                if user_data['violation_count'] >= 3:
                    user_data['blocked_until'] = current_time + 3600  # 1 hour block
                
                return {
                    'allowed': False,
                    'reason': 'Query violates usage policy',
                    'violation_count': user_data['violation_count']
                }
        
        # Subscription-specific limits
        if subscription_tier == 'free':
            # Free tier gets basic queries only
            if len(query.split()) > 20:
                return {
                    'allowed': False,
                    'reason': 'Free tier limited to 20 words per query',
                    'suggestion': 'Upgrade to Pro for longer queries'
                }
        
        return {'allowed': True}


# Global instance
_serper_service = None

def get_serper_service() -> SerperSearchService:
    """Get global Serper search service instance"""
    global _serper_service
    if _serper_service is None:
        _serper_service = SerperSearchService()
    return _serper_service