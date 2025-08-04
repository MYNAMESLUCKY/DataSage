import React, { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

export default function SimpleRagApp() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setResponse(`Enterprise RAG Response: "${query}"\n\nThis is a demonstration of our enterprise RAG system with:\n• Free LLM models (Llama 3.2, Mistral 7B)\n• Web search integration (Serper API)\n• Business logic protection\n• Subscription-based access control\n\nThe system processes your query through our knowledge base and provides intelligent responses using cutting-edge AI models at zero cost for basic operations.`);
      setIsLoading(false);
    }, 2000);
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="text-2xl font-bold mb-4 text-gray-900">
          Enterprise RAG Intelligence System
        </h1>
        <p className="text-gray-600 mb-4">
          Powered by free LLM models with enterprise-grade business logic protection
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="query" className="text-gray-900 font-semibold">
              Ask a Question:
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question here..."
              className="input textarea mt-2"
              disabled={isLoading}
            />
          </div>
          
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="button"
          >
            {isLoading ? (
              <div className="flex items-center">
                <LoadingSpinner size="sm" />
                <span style={{ marginLeft: '8px' }}>Processing...</span>
              </div>
            ) : (
              'Process Query'
            )}
          </button>
        </form>

        {response && (
          <div className="card" style={{ marginTop: '20px', backgroundColor: '#f9fafb' }}>
            <h3 className="text-lg font-semibold mb-2 text-gray-900">Response:</h3>
            <pre style={{ whiteSpace: 'pre-wrap', color: '#374151', lineHeight: '1.5' }}>
              {response}
            </pre>
          </div>
        )}

        <div className="grid grid-cols-3" style={{ marginTop: '30px' }}>
          <div className="card text-center">
            <div className="text-2xl font-bold text-blue-600">0</div>
            <div className="text-gray-600">Cost Per Query</div>
          </div>
          <div className="card text-center">
            <div className="text-2xl font-bold text-green-600">7+</div>
            <div className="text-gray-600">Free Models</div>
          </div>
          <div className="card text-center">
            <div className="text-2xl font-bold text-blue-600">2.3s</div>
            <div className="text-gray-600">Avg Response</div>
          </div>
        </div>
      </div>
    </div>
  );
}