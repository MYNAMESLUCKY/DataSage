import React, { useState, useRef, useEffect } from 'react';
import { 
  PaperAirplaneIcon, 
  CpuChipIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { useRAGSystem } from '../hooks/useRAGSystem';
import { useSubscription } from '../hooks/useSubscription';
import LoadingSpinner from './LoadingSpinner';
import ReactMarkdown from 'react-markdown';
// Simple code highlighting alternative
const CodeBlock = ({ children, className }: any) => (
  <pre style={{
    backgroundColor: '#2d3748',
    color: '#e2e8f0',
    padding: '12px',
    borderRadius: '6px',
    overflow: 'auto',
    fontSize: '14px'
  }}>
    <code className={className}>{children}</code>
  </pre>
);

interface QueryInterfaceProps {
  user: any;
  className?: string;
}

interface QueryResult {
  id: string;
  query: string;
  answer: string;
  sources: string[];
  processingTime: number;
  modelUsed: string;
  gpuAccelerated: boolean;
  timestamp: Date;
  status: 'success' | 'error' | 'limited';
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ user, className = '' }) => {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<QueryResult[]>([]);
  const [selectedModel, setSelectedModel] = useState('auto');
  const [enableGPU, setEnableGPU] = useState(true);
  const [maxSources, setMaxSources] = useState(10);
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { processQuery, isLoading, response, error } = useRAGSystem();
  const { subscription } = useSubscription(user?.id);
  
  // Default available models for free tier
  const availableModels = [
    { id: 'auto', name: 'Auto-Select (Recommended)', free: true },
    { id: 'llama-3.2-7b', name: 'Llama 3.2 7B', free: true },
    { id: 'mistral-7b', name: 'Mistral 7B', free: true },
    { id: 'gemma-7b', name: 'Gemma 7B', free: true },
    { id: 'qwen-7b', name: 'Qwen 7B', free: true },
    { id: 'deepseek-coder', name: 'DeepSeek Coder', free: true }
  ];
  
  const systemStatus = {
    gpu_available: true,
    models_loaded: availableModels.length,
    search_enabled: true
  };

  // Example queries for different subscription tiers
  const exampleQueries = {
    free: [
      "What are the main benefits of renewable energy?",
      "Explain the basics of machine learning",
      "How does photosynthesis work?"
    ],
    pro: [
      "Compare the latest developments in quantum computing vs classical computing architectures",
      "Analyze the economic impact of AI automation on different industry sectors",
      "What are the technical challenges in scaling nuclear fusion for commercial energy?"
    ],
    enterprise: [
      "Provide a comprehensive analysis of enterprise AI implementation strategies, including ROI calculations, risk assessment, and integration timelines across Fortune 500 companies",
      "Compare and contrast the regulatory frameworks for AI governance across US, EU, and Asian markets, with specific focus on data privacy and algorithmic accountability",
      "Analyze the technical and business implications of quantum-resistant cryptography adoption for enterprise security infrastructure"
    ]
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim() || isProcessing) return;

    setIsProcessing(true);
    
    try {
      const startTime = Date.now();
      
      // Process query using the RAG system hook
      await processQuery({
        query,
        user_id: user?.id || 'anonymous',
        max_sources: maxSources,
        model: selectedModel === 'auto' ? 'llama-3.2-7b' : selectedModel
      });

      const processingTime = Date.now() - startTime;

      // Create demo result for now (will be replaced with actual response)
      const result: QueryResult = {
        id: Date.now().toString(),
        query,
        answer: `Demo response for: "${query}"\n\nThis is a placeholder response. To enable full RAG functionality, please add your API keys (OPENAI_API_KEY, SERPER_API_KEY) to the system.`,
        sources: ['Demo Source 1', 'Demo Source 2'],
        processingTime: processingTime,
        modelUsed: selectedModel,
        gpuAccelerated: enableGPU,
        timestamp: new Date(),
        status: 'success'
      };

      setResults(prev => [result, ...prev]);
      setQuery('');
      
    } catch (error) {
      const errorResult: QueryResult = {
        id: Date.now().toString(),
        query,
        answer: `Error processing query: ${error instanceof Error ? error.message : 'Unknown error'}`,
        sources: [],
        processingTime: 0,
        modelUsed: 'error',
        gpuAccelerated: false,
        timestamp: new Date(),
        status: 'error'
      };
      setResults(prev => [errorResult, ...prev]);
    } finally {
      setIsProcessing(false);
    }
  };

  const insertExampleQuery = (exampleQuery: string) => {
    setQuery(exampleQuery);
    textareaRef.current?.focus();
  };

  const getSubscriptionLimits = () => {
    // Default limits for free tier users
    const defaultLimits = { queries: 50, sources: 5 };
    
    if (!user) return defaultLimits;
    
    // Set limits based on subscription tier
    const tierLimits = {
      free: { queries: 50, sources: 5 },
      pro: { queries: 500, sources: 15 },
      enterprise: { queries: -1, sources: 20 }
    };
    
    return tierLimits[user.subscription_tier as keyof typeof tierLimits] || defaultLimits;
  };

  const limits = getSubscriptionLimits();

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Query Input Section */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Query Interface</h3>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <span>{limits.queries === -1 ? 'Unlimited' : limits.queries} queries/day</span>
            <span>•</span>
            <span>{limits.sources === -1 ? 'Unlimited' : limits.sources} sources max</span>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything... The system will intelligently search knowledge bases and the web to provide comprehensive answers."
              rows={4}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 resize-none"
              disabled={isProcessing}
            />
          </div>

          {/* Advanced Options */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Selection
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                disabled={isProcessing}
              >
                <option value="auto">Auto-select best model</option>
                {availableModels.map((model: any) => (
                  <option key={model.id} value={model.id}>
                    {model.name} - {model.provider}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Sources
              </label>
              <select
                value={maxSources}
                onChange={(e) => setMaxSources(parseInt(e.target.value))}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                disabled={isProcessing}
              >
                {[5, 10, 15, 20].filter(n => n <= limits.sources || limits.sources === -1).map((num) => (
                  <option key={num} value={num}>{num} sources</option>
                ))}
              </select>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  GPU Acceleration
                </label>
                <button
                  type="button"
                  onClick={() => setEnableGPU(!enableGPU)}
                  className={`${
                    enableGPU ? 'bg-blue-600' : 'bg-gray-200'
                  } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
                  disabled={isProcessing || subscription?.tier === 'free'}
                >
                  <span
                    className={`${
                      enableGPU ? 'translate-x-5' : 'translate-x-0'
                    } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              {isProcessing && (
                <div className="flex items-center space-x-2">
                  <LoadingSpinner size="sm" />
                  <span>Processing with {enableGPU && subscription?.tier !== 'free' ? 'GPU acceleration' : 'standard processing'}...</span>
                </div>
              )}
            </div>
            
            <button
              type="submit"
              disabled={!query.trim() || isProcessing}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <PaperAirplaneIcon className="h-4 w-4 mr-2" />
                  Ask Question
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Example Queries */}
      <div className="bg-blue-50 rounded-lg p-4">
        <div className="flex items-center mb-3">
          <LightBulbIcon className="h-5 w-5 text-blue-600 mr-2" />
          <h4 className="text-sm font-medium text-blue-900">
            Example Queries for {subscription?.name || 'Free Tier'}
          </h4>
        </div>
        <div className="space-y-2">
          {exampleQueries[subscription?.tier as keyof typeof exampleQueries]?.map((example, index) => (
            <button
              key={index}
              onClick={() => insertExampleQuery(example)}
              className="text-left w-full p-2 text-sm text-blue-800 hover:bg-blue-100 rounded border border-blue-200 hover:border-blue-300 transition-colors"
              disabled={isProcessing}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Results Section */}
      <div className="space-y-4">
        {results.map((result) => (
          <QueryResultCard key={result.id} result={result} />
        ))}
      </div>
    </div>
  );
};

// Query Result Card Component
const QueryResultCard: React.FC<{ result: QueryResult }> = ({ result }) => {
  const getStatusIcon = () => {
    switch (result.status) {
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      case 'limited':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusBgColor = () => {
    switch (result.status) {
      case 'success':
        return 'bg-white';
      case 'error':
        return 'bg-red-50';
      case 'limited':
        return 'bg-yellow-50';
      default:
        return 'bg-gray-50';
    }
  };

  return (
    <div className={`${getStatusBgColor()} shadow rounded-lg p-6 border border-gray-200`}>
      {/* Query Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-2">
          {getStatusIcon()}
          <h4 className="text-sm font-medium text-gray-900">Query</h4>
        </div>
        <div className="flex items-center space-x-4 text-xs text-gray-500">
          <span>{result.timestamp.toLocaleTimeString()}</span>
          <span>{result.processingTime.toFixed(2)}s</span>
          {result.gpuAccelerated && (
            <div className="flex items-center space-x-1">
              <CpuChipIcon className="h-3 w-3" />
              <span>GPU</span>
            </div>
          )}
        </div>
      </div>

      {/* Query Text */}
      <div className="mb-4 p-3 bg-gray-100 rounded text-sm text-gray-800">
        {result.query}
      </div>

      {/* Answer */}
      <div className="mb-4">
        <h5 className="text-sm font-medium text-gray-900 mb-2">Answer:</h5>
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            components={{
              code({ node, className, children, ...props }: any) {
                const inline = props.inline;
                return !inline ? (
                  <CodeBlock className={className} {...props}>
                    {String(children).replace(/\n$/, '')}
                  </CodeBlock>
                ) : (
                  <code 
                    className={className} 
                    style={{ 
                      backgroundColor: '#f1f5f9', 
                      padding: '2px 4px', 
                      borderRadius: '3px',
                      fontSize: '0.875em'
                    }} 
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
            }}
          >
            {result.answer}
          </ReactMarkdown>
        </div>
      </div>

      {/* Sources */}
      {result.sources.length > 0 && (
        <div>
          <h5 className="text-sm font-medium text-gray-900 mb-2">Sources:</h5>
          <div className="space-y-1">
            {result.sources.map((source, index) => (
              <div key={index} className="text-xs text-blue-600 hover:text-blue-800">
                <a href={source} target="_blank" rel="noopener noreferrer" className="underline">
                  {source.length > 80 ? `${source.substring(0, 80)}...` : source}
                </a>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Info */}
      <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between text-xs text-gray-500">
        <span>Model: {result.modelUsed}</span>
        <span>
          {result.gpuAccelerated ? 'GPU-accelerated' : 'Standard processing'}
        </span>
      </div>
    </div>
  );
};

export default QueryInterface;