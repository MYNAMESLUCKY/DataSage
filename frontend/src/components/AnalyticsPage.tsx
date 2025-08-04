import React from 'react';

interface AnalyticsPageProps {
  user: any;
  className?: string;
}

export default function AnalyticsPage({ user, className }: AnalyticsPageProps) {
  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Analytics Dashboard</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-blue-600 uppercase tracking-wide">Total Queries</h3>
            <p className="text-2xl font-bold text-blue-900 mt-2">0</p>
            <p className="text-sm text-blue-600 mt-1">This month</p>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-green-600 uppercase tracking-wide">Documents Processed</h3>
            <p className="text-2xl font-bold text-green-900 mt-2">0</p>
            <p className="text-sm text-green-600 mt-1">Total uploaded</p>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-purple-600 uppercase tracking-wide">Cost Savings</h3>
            <p className="text-2xl font-bold text-purple-900 mt-2">$0.00</p>
            <p className="text-sm text-purple-600 mt-1">Using free models</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Query Performance</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Average Response Time</span>
                <span className="text-sm font-medium text-gray-900">0.0s</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Success Rate</span>
                <span className="text-sm font-medium text-gray-900">0%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">GPU Acceleration Usage</span>
                <span className="text-sm font-medium text-gray-900">0%</span>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Subscription Usage</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Current Plan</span>
                <span className="text-sm font-medium text-gray-900">{user?.subscription_tier || 'Free'}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Queries Used Today</span>
                <span className="text-sm font-medium text-gray-900">0 / 50</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Tokens Used Today</span>
                <span className="text-sm font-medium text-gray-900">0 / 50,000</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}