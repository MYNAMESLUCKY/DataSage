import React from 'react';

interface AnalyticsPageProps {
  user: any;
}

export default function AnalyticsPage({ user }: AnalyticsPageProps) {
  const analytics = {
    queries_today: 12,
    queries_this_month: 143,
    avg_response_time: 2.3,
    success_rate: 96.8,
    cost_saved: 24.50,
    tokens_used: 15420,
    most_used_models: [
      { name: 'Llama 3.2 7B', usage: 45, cost_saved: 12.30 },
      { name: 'Mistral 7B', usage: 32, cost_saved: 8.20 },
      { name: 'Gemma 2B', usage: 23, cost_saved: 4.00 }
    ]
  };

  return (
    <div className="container">
      <div className="card">
        <h2 className="text-xl font-bold mb-4 text-gray-900">Usage Analytics</h2>
        <p className="text-gray-600 mb-6">
          Track your RAG system usage and cost savings
        </p>

        {/* Key Metrics */}
        <div className="grid grid-cols-2" style={{ marginBottom: '30px', gap: '20px' }}>
          <div className="card text-center" style={{ backgroundColor: '#eff6ff' }}>
            <div className="text-2xl font-bold text-blue-600">{analytics.queries_today}</div>
            <div className="text-gray-600">Queries Today</div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
              {50 - analytics.queries_today} remaining
            </div>
          </div>
          
          <div className="card text-center" style={{ backgroundColor: '#f0fdf4' }}>
            <div className="text-2xl font-bold text-green-600">${analytics.cost_saved}</div>
            <div className="text-gray-600">Cost Saved</div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
              vs. paid APIs
            </div>
          </div>
          
          <div className="card text-center" style={{ backgroundColor: '#fef3c7' }}>
            <div className="text-2xl font-bold" style={{ color: '#d97706' }}>{analytics.avg_response_time}s</div>
            <div className="text-gray-600">Avg Response</div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
              Last 30 days
            </div>
          </div>
          
          <div className="card text-center" style={{ backgroundColor: '#f3e8ff' }}>
            <div className="text-2xl font-bold" style={{ color: '#7c3aed' }}>{analytics.success_rate}%</div>
            <div className="text-gray-600">Success Rate</div>
            <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
              Query completion
            </div>
          </div>
        </div>

        {/* Monthly Usage */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Monthly Usage</h3>
          
          <div style={{ marginBottom: '16px' }}>
            <div className="flex justify-between">
              <span className="text-gray-600">Queries this month</span>
              <span className="font-semibold text-gray-900">{analytics.queries_this_month}</span>
            </div>
          </div>
          
          <div style={{ marginBottom: '16px' }}>
            <div className="flex justify-between">
              <span className="text-gray-600">Tokens processed</span>
              <span className="font-semibold text-gray-900">{analytics.tokens_used.toLocaleString()}</span>
            </div>
          </div>
          
          <div style={{ marginBottom: '16px' }}>
            <div className="flex justify-between">
              <span className="text-gray-600">Average query cost</span>
              <span className="font-semibold text-green-600">$0.00</span>
            </div>
          </div>
        </div>

        {/* Model Usage */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Most Used Models</h3>
          
          {analytics.most_used_models.map((model, index) => (
            <div key={index} style={{ marginBottom: '16px', paddingBottom: '16px', borderBottom: '1px solid #f3f4f6' }}>
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold text-gray-900">{model.name}</span>
                <span className="text-blue-600 font-bold">${model.cost_saved}</span>
              </div>
              
              <div style={{ marginBottom: '4px' }}>
                <div className="flex justify-between">
                  <span style={{ fontSize: '14px', color: '#6b7280' }}>Usage: {model.usage} queries</span>
                  <span style={{ fontSize: '14px', color: '#6b7280' }}>Cost saved</span>
                </div>
              </div>
              
              <div style={{ 
                width: '100%', 
                height: '6px', 
                backgroundColor: '#e5e7eb', 
                borderRadius: '3px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  width: `${(model.usage / analytics.queries_this_month) * 100}%`, 
                  height: '100%', 
                  backgroundColor: '#3b82f6',
                  borderRadius: '3px'
                }}></div>
              </div>
            </div>
          ))}
        </div>

        {/* Upgrade Suggestion */}
        <div className="card" style={{ backgroundColor: '#fef9c3', marginTop: '20px' }}>
          <h3 className="text-lg font-semibold mb-2 text-gray-900">💡 Upgrade for More Analytics</h3>
          <p className="text-gray-600 mb-4">
            Get detailed analytics, custom reports, and API usage tracking with Pro or Enterprise plans.
          </p>
          <button className="button" style={{ backgroundColor: '#f59e0b' }}>
            View Upgrade Options
          </button>
        </div>
      </div>
    </div>
  );
}