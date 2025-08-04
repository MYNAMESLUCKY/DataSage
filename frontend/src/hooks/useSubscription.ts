import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../utils/apiClient';

interface SubscriptionHook {
  subscription: Subscription | null;
  usage: Usage | null;
  plans: SubscriptionPlan[];
  checkUsageLimit: (operation: string, amount: number) => UsageCheckResult;
  upgradeSubscription: (planId: string, paymentMethodId?: string) => Promise<UpgradeResult>;
  loading: boolean;
  error: string | null;
}

interface Subscription {
  tier: 'free' | 'pro' | 'enterprise';
  name: string;
  status: string;
  currentPeriodStart: string;
  currentPeriodEnd: string;
  limits: {
    queries_per_day: number;
    tokens_per_day: number;
    search_requests_per_day: number;
    max_sources_per_query: number;
    concurrent_queries: number;
    rate_limit_rpm: number;
  };
  features: string[];
  priceMonthly: number;
  priceYearly: number;
  prioritySupport: boolean;
  apiAccess: boolean;
}

interface Usage {
  queries: number;
  tokens: number;
  searches: number;
  apiCalls: number;
  cost: number;
  date: string;
}

interface SubscriptionPlan {
  id: string;
  tier: string;
  name: string;
  priceMonthly: number;
  priceYearly: number;
  features: string[];
  limits: Record<string, number>;
  prioritySupport: boolean;
  apiAccess: boolean;
}

interface UsageCheckResult {
  allowed: boolean;
  reason?: string;
  suggestion?: string;
  remainingQuota?: Record<string, any>;
}

interface UpgradeResult {
  success: boolean;
  error?: string;
  message?: string;
  subscription?: Subscription;
}

export const useSubscription = (): SubscriptionHook => {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [plans, setPlans] = useState<SubscriptionPlan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch user subscription
  const fetchSubscription = useCallback(async () => {
    try {
      const response = await apiClient.get('/subscription/current');
      setSubscription(response.data);
    } catch (err) {
      console.error('Failed to fetch subscription:', err);
      setError('Failed to load subscription data');
    }
  }, []);

  // Fetch usage data
  const fetchUsage = useCallback(async () => {
    try {
      const response = await apiClient.get('/subscription/usage');
      setUsage(response.data);
    } catch (err) {
      console.error('Failed to fetch usage:', err);
      setError('Failed to load usage data');
    }
  }, []);

  // Fetch available plans
  const fetchPlans = useCallback(async () => {
    try {
      const response = await apiClient.get('/subscription/plans');
      setPlans(response.data);
    } catch (err) {
      console.error('Failed to fetch plans:', err);
      setError('Failed to load subscription plans');
    }
  }, []);

  // Check usage limits
  const checkUsageLimit = useCallback((operation: string, amount: number = 1): UsageCheckResult => {
    if (!subscription || !usage) {
      return {
        allowed: false,
        reason: 'Subscription data not loaded',
        suggestion: 'Please refresh the page'
      };
    }

    const limits = subscription.limits;
    
    // Map operations to usage tracking
    const operationLimits: Record<string, { limit: number; current: number; name: string }> = {
      'query': {
        limit: limits.queries_per_day,
        current: usage.queries,
        name: 'daily queries'
      },
      'search': {
        limit: limits.search_requests_per_day,
        current: usage.searches,
        name: 'daily searches'
      },
      'api_call': {
        limit: limits.queries_per_day, // API calls count as queries
        current: usage.apiCalls,
        name: 'daily API calls'
      }
    };

    const opLimit = operationLimits[operation];
    if (!opLimit) {
      return { allowed: true }; // Unknown operation, allow by default
    }

    // -1 means unlimited (enterprise tier)
    if (opLimit.limit === -1) {
      return { allowed: true };
    }

    // Check if adding the requested amount would exceed the limit
    if (opLimit.current + amount > opLimit.limit) {
      const nextTier = subscription.tier === 'free' ? 'Pro' : 'Enterprise';
      return {
        allowed: false,
        reason: `${opLimit.name} limit exceeded: ${opLimit.current}/${opLimit.limit}`,
        suggestion: `Upgrade to ${nextTier} for higher limits`
      };
    }

    return {
      allowed: true,
      remainingQuota: {
        [operation]: opLimit.limit - opLimit.current
      }
    };
  }, [subscription, usage]);

  // Upgrade subscription
  const upgradeSubscription = useCallback(async (planId: string, paymentMethodId?: string): Promise<UpgradeResult> => {
    try {
      const response = await apiClient.post('/subscription/upgrade', {
        plan_id: planId,
        payment_method_id: paymentMethodId
      });

      if (response.data.success) {
        // Refresh subscription data
        await fetchSubscription();
        return {
          success: true,
          message: response.data.message,
          subscription: response.data.subscription
        };
      } else {
        return {
          success: false,
          error: response.data.error
        };
      }
    } catch (err: any) {
      console.error('Subscription upgrade failed:', err);
      return {
        success: false,
        error: err.response?.data?.error || 'Upgrade failed'
      };
    }
  }, [fetchSubscription]);

  // Initialize data
  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      setError(null);

      try {
        await Promise.all([
          fetchSubscription(),
          fetchUsage(),
          fetchPlans()
        ]);
      } catch (err) {
        console.error('Failed to initialize subscription data:', err);
      } finally {
        setLoading(false);
      }
    };

    initializeData();
  }, [fetchSubscription, fetchUsage, fetchPlans]);

  // Periodically refresh usage data
  useEffect(() => {
    const interval = setInterval(() => {
      fetchUsage();
    }, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, [fetchUsage]);

  return {
    subscription,
    usage,
    plans,
    checkUsageLimit,
    upgradeSubscription,
    loading,
    error
  };
};