import { useState, useEffect } from 'react';
import { apiClient } from '../utils/apiClient';

interface Subscription {
  tier: string;
  name: string;
  status: string;
  limits: {
    daily_queries: number;
    max_sources: number;
    gpu_acceleration: boolean;
  };
  usage_today: {
    queries: number;
    sources_used: number;
  };
  features: string[];
}

export function useSubscription(userId: string) {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSubscription();
  }, [userId]);

  const fetchSubscription = async () => {
    try {
      setIsLoading(true);
      // For demo purposes, simulate API call
      setTimeout(() => {
        setSubscription({
          tier: 'free',
          name: 'Free Plan',
          status: 'active',
          limits: {
            daily_queries: 50,
            max_sources: 5,
            gpu_acceleration: false,
          },
          usage_today: {
            queries: 12,
            sources_used: 3,
          },
          features: [
            'Free LLM models',
            'Basic web search',
            'Standard processing',
            'Community support'
          ]
        });
        setIsLoading(false);
      }, 500);
    } catch (err) {
      setError('Failed to fetch subscription');
      setIsLoading(false);
    }
  };

  const upgradeSubscription = async (newPlan: string) => {
    try {
      // Simulate upgrade
      console.log('Upgrading to:', newPlan);
    } catch (err) {
      setError('Failed to upgrade subscription');
    }
  };

  return {
    subscription,
    isLoading,
    error,
    upgradeSubscription,
    refetch: fetchSubscription,
  };
}