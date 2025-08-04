import React from 'react';

interface UsageMetricsProps {
  metrics: {
    totalQueries: number;
    averageResponseTime: number;
    successRate: number;
    costSaved: number;
  };
}

export default function UsageMetrics({ metrics }: UsageMetricsProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Usage Metrics</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <p className="text-2xl font-bold text-blue-600">{metrics.totalQueries}</p>
          <p className="text-sm text-gray-600">Total Queries</p>
        </div>
        
        <div className="text-center">
          <p className="text-2xl font-bold text-green-600">{metrics.averageResponseTime}s</p>
          <p className="text-sm text-gray-600">Avg Response</p>
        </div>
        
        <div className="text-center">
          <p className="text-2xl font-bold text-purple-600">{metrics.successRate}%</p>
          <p className="text-sm text-gray-600">Success Rate</p>
        </div>
        
        <div className="text-center">
          <p className="text-2xl font-bold text-orange-600">${metrics.costSaved}</p>
          <p className="text-sm text-gray-600">Cost Saved</p>
        </div>
      </div>
    </div>
  );
}