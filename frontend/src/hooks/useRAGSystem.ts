import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../utils/apiClient';

interface RAGSystemHook {
  processQuery: (params: QueryParams) => Promise<QueryResponse>;
  availableModels: Model[];
  systemStatus: SystemStatus | null;
  loading: boolean;
  error: string | null;
}

interface QueryParams {
  query: string;
  userId: string;
  subscriptionTier: string;
  enableGPU?: boolean;
  maxSources?: number;
  modelPreference?: string | null;
}

interface QueryResponse {
  status: string;
  answer: string;
  sources?: string[];
  processingTime?: number;
  modelUsed?: string;
  gpuAccelerated?: boolean;
  confidence?: number;
  costSaved?: number;
}

interface Model {
  id: string;
  name: string;
  provider: string;
  qualityScore: number;
  features: string[];
  contextLength: number;
  costPerToken: number;
}

interface SystemStatus {
  gpu_providers_available: number;
  api_gateway_healthy: boolean;
  avg_response_time: number;
  success_rate: number;
  total_models_available: number;
  free_tier_usage: number;
}

export const useRAGSystem = (): RAGSystemHook => {
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch available models
  const fetchAvailableModels = useCallback(async () => {
    try {
      const response = await apiClient.get('/models/available');
      setAvailableModels(response.data);
    } catch (err) {
      console.error('Failed to fetch available models:', err);
      setError('Failed to load available models');
    }
  }, []);

  // Fetch system status
  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await apiClient.get('/system/status');
      setSystemStatus(response.data);
    } catch (err) {
      console.error('Failed to fetch system status:', err);
      setError('Failed to load system status');
    }
  }, []);

  // Process query
  const processQuery = useCallback(async (params: QueryParams): Promise<QueryResponse> => {
    try {
      const response = await apiClient.post('/query/process', {
        query: params.query,
        user_id: params.userId,
        subscription_tier: params.subscriptionTier,
        enable_gpu_acceleration: params.enableGPU,
        max_sources: params.maxSources,
        model_preference: params.modelPreference,
      });

      return response.data;
    } catch (err: any) {
      console.error('Query processing failed:', err);
      
      // Handle different error types
      if (err.response?.status === 429) {
        throw new Error('Rate limit exceeded. Please try again later.');
      } else if (err.response?.status === 402) {
        throw new Error('Usage limit exceeded. Please upgrade your subscription.');
      } else if (err.response?.status === 403) {
        throw new Error('Access denied. Please check your subscription.');
      } else {
        throw new Error(err.response?.data?.message || 'Query processing failed');
      }
    }
  }, []);

  // Initialize data
  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      setError(null);

      try {
        await Promise.all([
          fetchAvailableModels(),
          fetchSystemStatus()
        ]);
      } catch (err) {
        console.error('Failed to initialize RAG system data:', err);
      } finally {
        setLoading(false);
      }
    };

    initializeData();
  }, [fetchAvailableModels, fetchSystemStatus]);

  // Periodically refresh system status
  useEffect(() => {
    const interval = setInterval(() => {
      fetchSystemStatus();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [fetchSystemStatus]);

  return {
    processQuery,
    availableModels,
    systemStatus,
    loading,
    error
  };
};