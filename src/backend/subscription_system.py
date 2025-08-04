"""
Subscription System for Enterprise RAG Platform
Implements tiered pricing, billing, and business logic protection
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import hashlib
import secrets
from enum import Enum

from ..utils.utils import setup_logging

logger = setup_logging(__name__)

class SubscriptionTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class SubscriptionPlan:
    """Subscription plan configuration"""
    tier: SubscriptionTier
    name: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    limits: Dict[str, int]
    priority_support: bool
    api_access: bool

@dataclass
class UserSubscription:
    """User subscription data"""
    user_id: str
    subscription_tier: SubscriptionTier
    plan_id: str
    status: str  # active, cancelled, expired, suspended
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    payment_method: Optional[str] = None
    stripe_subscription_id: Optional[str] = None

class SubscriptionManager:
    """
    Manages subscription plans, billing, and access control
    Implements business logic to prevent resource abuse
    """
    
    def __init__(self):
        self.db_path = "subscriptions.db"
        self.plans = self._initialize_plans()
        self.usage_monitor = UsageMonitor()
        self.billing_system = BillingSystem()
        self._initialize_database()
        
    def _initialize_plans(self) -> Dict[str, SubscriptionPlan]:
        """Initialize subscription plans with 2025 pricing"""
        plans = {
            "free": SubscriptionPlan(
                tier=SubscriptionTier.FREE,
                name="Free Tier",
                price_monthly=0.0,
                price_yearly=0.0,
                features=[
                    "50 queries per day",
                    "Basic RAG functionality", 
                    "Free LLM models only",
                    "Standard search results",
                    "Community support"
                ],
                limits={
                    "queries_per_day": 50,
                    "tokens_per_day": 50000,
                    "search_requests_per_day": 20,
                    "max_document_upload_mb": 10,
                    "max_sources_per_query": 5,
                    "concurrent_queries": 1,
                    "rate_limit_rpm": 10
                },
                priority_support=False,
                api_access=False
            ),
            "pro": SubscriptionPlan(
                tier=SubscriptionTier.PRO,
                name="Pro Plan",
                price_monthly=29.99,
                price_yearly=299.99,  # 2 months free
                features=[
                    "2,000 queries per day",
                    "Advanced RAG with GPU acceleration",
                    "Premium LLM models access",
                    "Enhanced search capabilities",
                    "Priority email support",
                    "API access included",
                    "Advanced analytics",
                    "Custom model selection"
                ],
                limits={
                    "queries_per_day": 2000,
                    "tokens_per_day": 2000000,
                    "search_requests_per_day": 500,
                    "max_document_upload_mb": 100,
                    "max_sources_per_query": 15,
                    "concurrent_queries": 5,
                    "rate_limit_rpm": 60
                },
                priority_support=True,
                api_access=True
            ),
            "enterprise": SubscriptionPlan(
                tier=SubscriptionTier.ENTERPRISE,
                name="Enterprise Plan",
                price_monthly=99.99,
                price_yearly=999.99,  # 2 months free
                features=[
                    "Unlimited queries",
                    "Full GPU infrastructure access",
                    "All LLM models available",
                    "White-label deployment",
                    "Dedicated support channel", 
                    "Custom integrations",
                    "Advanced security features",
                    "SLA guarantees",
                    "On-premises deployment option",
                    "Custom rate limits"
                ],
                limits={
                    "queries_per_day": -1,  # Unlimited
                    "tokens_per_day": -1,   # Unlimited
                    "search_requests_per_day": -1,  # Unlimited
                    "max_document_upload_mb": 1000,
                    "max_sources_per_query": 50,
                    "concurrent_queries": 20,
                    "rate_limit_rpm": 300
                },
                priority_support=True,
                api_access=True
            )
        }
        
        logger.info(f"Initialized {len(plans)} subscription plans")
        return plans
    
    def _initialize_database(self):
        """Initialize SQLite database for subscription management"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create subscriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    subscription_tier TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_period_start TEXT NOT NULL,
                    current_period_end TEXT NOT NULL,
                    cancel_at_period_end BOOLEAN DEFAULT FALSE,
                    payment_method TEXT,
                    stripe_subscription_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create usage tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    queries_count INTEGER DEFAULT 0,
                    tokens_consumed INTEGER DEFAULT 0,
                    search_requests INTEGER DEFAULT 0,
                    api_calls INTEGER DEFAULT 0,
                    cost_incurred REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, date)
                )
            """)
            
            # Create billing events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    stripe_payment_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Subscription database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize subscription database: {e}")
    
    def get_user_subscription(self, user_id: str) -> UserSubscription:
        """Get user's current subscription"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id, subscription_tier, plan_id, status, 
                       current_period_start, current_period_end, cancel_at_period_end,
                       payment_method, stripe_subscription_id
                FROM subscriptions 
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return UserSubscription(
                    user_id=row[0],
                    subscription_tier=SubscriptionTier(row[1]),
                    plan_id=row[2],
                    status=row[3],
                    current_period_start=datetime.fromisoformat(row[4]),
                    current_period_end=datetime.fromisoformat(row[5]),
                    cancel_at_period_end=bool(row[6]),
                    payment_method=row[7],
                    stripe_subscription_id=row[8]
                )
            else:
                # Return default free subscription
                return self._create_free_subscription(user_id)
                
        except Exception as e:
            logger.error(f"Failed to get user subscription for {user_id}: {e}")
            return self._create_free_subscription(user_id)
    
    def _create_free_subscription(self, user_id: str) -> UserSubscription:
        """Create default free subscription for new user"""
        now = datetime.now()
        subscription = UserSubscription(
            user_id=user_id,
            subscription_tier=SubscriptionTier.FREE,
            plan_id="free",
            status="active",
            current_period_start=now,
            current_period_end=now + timedelta(days=365),  # Free never expires
            cancel_at_period_end=False
        )
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, subscription_tier, plan_id, status, current_period_start, 
                 current_period_end, cancel_at_period_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, subscription.subscription_tier.value, subscription.plan_id,
                subscription.status, subscription.current_period_start.isoformat(),
                subscription.current_period_end.isoformat(), subscription.cancel_at_period_end
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Created free subscription for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save free subscription for {user_id}: {e}")
        
        return subscription
    
    def check_access_permission(
        self, 
        user_id: str, 
        operation: str, 
        resource_amount: int = 1
    ) -> Dict[str, Any]:
        """
        Check if user has permission for operation based on subscription
        
        Args:
            user_id: User identifier
            operation: Operation type (query, search, api_call, upload)
            resource_amount: Amount of resource requested
        """
        subscription = self.get_user_subscription(user_id)
        plan = self.plans.get(subscription.plan_id)
        
        if not plan:
            return {
                'allowed': False,
                'reason': 'Invalid subscription plan',
                'suggestion': 'Contact support'
            }
        
        # Check subscription status
        if subscription.status != "active":
            return {
                'allowed': False,
                'reason': f'Subscription is {subscription.status}',
                'suggestion': 'Renew your subscription'
            }
        
        # Check if subscription has expired
        if datetime.now() > subscription.current_period_end:
            return {
                'allowed': False,
                'reason': 'Subscription has expired',
                'suggestion': 'Renew your subscription'
            }
        
        # Check daily usage limits
        usage_today = self.usage_monitor.get_daily_usage(user_id)
        
        operation_limits = {
            'query': ('queries_per_day', usage_today.get('queries', 0)),
            'search': ('search_requests_per_day', usage_today.get('searches', 0)),
            'api_call': ('queries_per_day', usage_today.get('api_calls', 0)),  # API calls count as queries
            'upload': ('max_document_upload_mb', resource_amount)  # resource_amount is file size in MB
        }
        
        if operation in operation_limits:
            limit_key, current_usage = operation_limits[operation]
            limit_value = plan.limits.get(limit_key, 0)
            
            # -1 means unlimited (enterprise)
            if limit_value != -1 and current_usage + resource_amount > limit_value:
                return {
                    'allowed': False,
                    'reason': f'Daily {operation} limit exceeded: {limit_value}/{current_usage}',
                    'current_usage': current_usage,
                    'limit': limit_value,
                    'suggestion': f'Upgrade to {self._get_next_tier(subscription.subscription_tier)} for higher limits'
                }
        
        return {
            'allowed': True,
            'remaining_quota': self._calculate_remaining_quota(subscription, plan, usage_today)
        }
    
    def _get_next_tier(self, current_tier: SubscriptionTier) -> str:
        """Get next subscription tier for upgrade suggestions"""
        if current_tier == SubscriptionTier.FREE:
            return "Pro"
        elif current_tier == SubscriptionTier.PRO:
            return "Enterprise"
        else:
            return "Enterprise"
    
    def _calculate_remaining_quota(
        self, 
        subscription: UserSubscription,
        plan: SubscriptionPlan, 
        usage_today: Dict[str, int]
    ) -> Dict[str, Any]:
        """Calculate remaining quota for user"""
        remaining = {}
        
        for limit_key, limit_value in plan.limits.items():
            if limit_value == -1:  # Unlimited
                remaining[limit_key] = "unlimited"
            else:
                usage_key_map = {
                    'queries_per_day': 'queries',
                    'search_requests_per_day': 'searches',
                    'tokens_per_day': 'tokens'
                }
                
                usage_key = usage_key_map.get(limit_key, limit_key)
                current_usage = usage_today.get(usage_key, 0)
                remaining[limit_key] = max(0, limit_value - current_usage)
        
        return remaining
    
    def record_usage(
        self, 
        user_id: str, 
        operation: str, 
        amount: int = 1,
        cost: float = 0.0
    ):
        """Record usage for billing and limit tracking"""
        try:
            self.usage_monitor.record_usage(user_id, operation, amount, cost)
            logger.debug(f"Recorded usage for {user_id}: {operation} x{amount}")
        except Exception as e:
            logger.error(f"Failed to record usage for {user_id}: {e}")
    
    def get_subscription_plans(self) -> List[Dict[str, Any]]:
        """Get all available subscription plans"""
        plans_data = []
        for plan_id, plan in self.plans.items():
            plan_dict = asdict(plan)
            plan_dict['tier'] = plan.tier.value
            plans_data.append(plan_dict)
        
        return sorted(plans_data, key=lambda x: x['price_monthly'])
    
    def upgrade_subscription(
        self, 
        user_id: str, 
        new_plan_id: str,
        payment_method_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upgrade user subscription"""
        try:
            current_subscription = self.get_user_subscription(user_id)
            new_plan = self.plans.get(new_plan_id)
            
            if not new_plan:
                return {
                    'success': False,
                    'error': 'Invalid plan selected'
                }
            
            # Check if it's actually an upgrade
            current_plan = self.plans.get(current_subscription.plan_id)
            if new_plan.price_monthly <= current_plan.price_monthly:
                return {
                    'success': False,
                    'error': 'Cannot downgrade or switch to same tier'
                }
            
            # Process payment if required
            if new_plan.price_monthly > 0:
                payment_result = self.billing_system.process_subscription_payment(
                    user_id, new_plan, payment_method_id
                )
                
                if not payment_result['success']:
                    return {
                        'success': False,
                        'error': f'Payment failed: {payment_result["error"]}'
                    }
            
            # Update subscription
            now = datetime.now()
            updated_subscription = UserSubscription(
                user_id=user_id,
                subscription_tier=new_plan.tier,
                plan_id=new_plan_id,
                status="active",
                current_period_start=now,
                current_period_end=now + timedelta(days=30),  # Monthly billing
                cancel_at_period_end=False,
                payment_method=payment_method_id,
                stripe_subscription_id=payment_result.get('subscription_id') if 'payment_result' in locals() else None
            )
            
            self._save_subscription(updated_subscription)
            
            logger.info(f"Upgraded subscription for {user_id} to {new_plan_id}")
            return {
                'success': True,
                'subscription': updated_subscription,
                'message': f'Successfully upgraded to {new_plan.name}'
            }
            
        except Exception as e:
            logger.error(f"Failed to upgrade subscription for {user_id}: {e}")
            return {
                'success': False,
                'error': f'Upgrade failed: {str(e)}'
            }
    
    def _save_subscription(self, subscription: UserSubscription):
        """Save subscription to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, subscription_tier, plan_id, status, current_period_start, 
                 current_period_end, cancel_at_period_end, payment_method, stripe_subscription_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                subscription.user_id,
                subscription.subscription_tier.value,
                subscription.plan_id,
                subscription.status,
                subscription.current_period_start.isoformat(),
                subscription.current_period_end.isoformat(),
                subscription.cancel_at_period_end,
                subscription.payment_method,
                subscription.stripe_subscription_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save subscription: {e}")
            raise
    
    def get_admin_dashboard_data(self) -> Dict[str, Any]:
        """Get subscription analytics for admin dashboard"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get subscription distribution
            cursor.execute("""
                SELECT subscription_tier, COUNT(*) as count
                FROM subscriptions 
                WHERE status = 'active'
                GROUP BY subscription_tier
            """)
            tier_distribution = dict(cursor.fetchall())
            
            # Get monthly revenue
            cursor.execute("""
                SELECT SUM(amount) as monthly_revenue
                FROM billing_events 
                WHERE status = 'completed'
                AND created_at >= date('now', '-30 days')
            """)
            monthly_revenue = cursor.fetchone()[0] or 0
            
            # Get total active subscriptions
            cursor.execute("""
                SELECT COUNT(*) as total_active
                FROM subscriptions 
                WHERE status = 'active'
            """)
            total_active = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'tier_distribution': tier_distribution,
                'monthly_revenue': monthly_revenue,
                'total_active_subscriptions': total_active,
                'average_revenue_per_user': monthly_revenue / max(total_active, 1),
                'conversion_metrics': self._calculate_conversion_metrics()
            }
            
        except Exception as e:
            logger.error(f"Failed to get admin dashboard data: {e}")
            return {}
    
    def _calculate_conversion_metrics(self) -> Dict[str, float]:
        """Calculate subscription conversion metrics"""
        # This would be implemented with proper analytics
        return {
            'free_to_pro_conversion': 0.12,  # 12%
            'pro_to_enterprise_conversion': 0.08,  # 8%
            'churn_rate': 0.05  # 5%
        }


class UsageMonitor:
    """Monitor and track user usage for billing"""
    
    def __init__(self):
        self.db_path = "subscriptions.db"
    
    def record_usage(self, user_id: str, operation: str, amount: int, cost: float):
        """Record usage event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date().isoformat()
            
            # Update or insert daily usage
            cursor.execute("""
                INSERT OR IGNORE INTO usage_tracking 
                (user_id, date, queries_count, tokens_consumed, search_requests, api_calls, cost_incurred)
                VALUES (?, ?, 0, 0, 0, 0, 0.0)
            """, (user_id, today))
            
            # Update specific usage type
            if operation == 'query':
                cursor.execute("""
                    UPDATE usage_tracking 
                    SET queries_count = queries_count + ?
                    WHERE user_id = ? AND date = ?
                """, (amount, user_id, today))
            elif operation == 'tokens':
                cursor.execute("""
                    UPDATE usage_tracking 
                    SET tokens_consumed = tokens_consumed + ?
                    WHERE user_id = ? AND date = ?
                """, (amount, user_id, today))
            elif operation == 'search':
                cursor.execute("""
                    UPDATE usage_tracking 
                    SET search_requests = search_requests + ?
                    WHERE user_id = ? AND date = ?
                """, (amount, user_id, today))
            elif operation == 'api_call':
                cursor.execute("""
                    UPDATE usage_tracking 
                    SET api_calls = api_calls + ?, cost_incurred = cost_incurred + ?
                    WHERE user_id = ? AND date = ?
                """, (amount, cost, user_id, today))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
    
    def get_daily_usage(self, user_id: str) -> Dict[str, int]:
        """Get today's usage for user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date().isoformat()
            cursor.execute("""
                SELECT queries_count, tokens_consumed, search_requests, api_calls, cost_incurred
                FROM usage_tracking 
                WHERE user_id = ? AND date = ?
            """, (user_id, today))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'queries': row[0],
                    'tokens': row[1], 
                    'searches': row[2],
                    'api_calls': row[3],
                    'cost': row[4]
                }
            else:
                return {'queries': 0, 'tokens': 0, 'searches': 0, 'api_calls': 0, 'cost': 0.0}
                
        except Exception as e:
            logger.error(f"Failed to get daily usage for {user_id}: {e}")
            return {'queries': 0, 'tokens': 0, 'searches': 0, 'api_calls': 0, 'cost': 0.0}


class BillingSystem:
    """Handle subscription billing and payments"""
    
    def __init__(self):
        self.stripe_api_key = os.getenv("STRIPE_SECRET_KEY")
        if not self.stripe_api_key:
            logger.warning("STRIPE_SECRET_KEY not configured - billing will be simulated")
    
    def process_subscription_payment(
        self, 
        user_id: str, 
        plan: SubscriptionPlan,
        payment_method_id: Optional[str]
    ) -> Dict[str, Any]:
        """Process subscription payment"""
        
        if not self.stripe_api_key:
            # Simulate successful payment for development
            logger.info(f"Simulated payment for {user_id}: ${plan.price_monthly}")
            return {
                'success': True,
                'subscription_id': f'sim_sub_{secrets.token_hex(8)}',
                'payment_id': f'sim_pay_{secrets.token_hex(8)}'
            }
        
        # Real Stripe integration would go here
        try:
            # This would use the Stripe Python library
            # import stripe
            # stripe.api_key = self.stripe_api_key
            # 
            # subscription = stripe.Subscription.create(
            #     customer=customer_id,
            #     items=[{'price': plan.stripe_price_id}],
            #     payment_behavior='default_incomplete',
            #     expand=['latest_invoice.payment_intent'],
            # )
            
            logger.info(f"Payment processing not implemented for {user_id}")
            return {
                'success': False,
                'error': 'Payment processing not implemented'
            }
            
        except Exception as e:
            logger.error(f"Payment processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global instance
_subscription_manager = None

def get_subscription_manager() -> SubscriptionManager:
    """Get global subscription manager instance"""
    global _subscription_manager
    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()
    return _subscription_manager