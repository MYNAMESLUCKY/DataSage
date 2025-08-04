import { useState } from 'react';
import { apiClient } from '../utils/apiClient';

interface QueryRequest {
  query: string;
  user_id: string;
  subscription_tier: string;
  enable_gpu_acceleration: boolean;
  max_sources: number;
  search_web: boolean;
}

interface QueryResponse {
  status: string;
  answer: string;
  sources: string[];
  web_sources: string[];
  processing_time: number;
  model_used: string;
  gpu_accelerated: boolean;
  confidence: number;
  cost_saved: number;
}

export function useRAGSystem() {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const processQuery = async (request: QueryRequest) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Simulate API call for demo
      setTimeout(() => {
        setResponse({
          status: 'success',
          answer: `Based on the query "${request.query}", here's a comprehensive response using our free LLM models and web search capabilities. This demonstrates the enterprise RAG system with intelligent document processing and real-time information retrieval.`,
          sources: [
            'Knowledge Base Document 1',
            'Knowledge Base Document 2', 
            'Knowledge Base Document 3'
          ],
          web_sources: [
            'https://example.com/article1',
            'https://example.com/article2'
          ],
          processing_time: 2.3,
          model_used: 'Llama 3.2 7B (Free)',
          gpu_accelerated: request.enable_gpu_acceleration,
          confidence: 0.92,
          cost_saved: 0.05
        });
        setIsLoading(false);
      }, 2000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query processing failed');
      setIsLoading(false);
    }
  };

  const clearResponse = () => {
    setResponse(null);
    setError(null);
  };

  return {
    processQuery,
    clearResponse,
    isLoading,
    response,
    error,
  };
}