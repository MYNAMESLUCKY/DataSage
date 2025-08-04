import React from 'react';

interface SubscriptionCardProps {
  subscription: {
    tier: string;
    name: string;
    status: string;
    limits: any;
    usage_today: any;
    features: string[];
  };
}

export default function SubscriptionCard({ subscription }: SubscriptionCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{subscription.name}</h3>
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
          subscription.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
        }`}>
          {subscription.status}
        </span>
      </div>
      
      <div className="space-y-3">
        <div>
          <p className="text-sm text-gray-600">Daily Queries</p>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full" 
              style={{ width: `${(subscription.usage_today.queries / subscription.limits.daily_queries) * 100}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {subscription.usage_today.queries} / {subscription.limits.daily_queries} used
          </p>
        </div>
        
        <div className="pt-3 border-t">
          <p className="text-sm font-medium text-gray-900 mb-2">Features:</p>
          <ul className="text-sm text-gray-600 space-y-1">
            {subscription.features.slice(0, 3).map((feature, index) => (
              <li key={index} className="flex items-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                {feature}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}