import React, { useState, useEffect } from 'react';
import { 
  ChartBarIcon, 
  CpuChipIcon, 
  GlobeAltIcon, 
  BoltIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../hooks/useAuth';
import { useSubscription } from '../hooks/useSubscription';
import { useRAGSystem } from '../hooks/useRAGSystem';
import LoadingSpinner from './LoadingSpinner';
import SubscriptionCard from './SubscriptionCard';
import QueryInterface from './QueryInterface';
import UsageMetrics from './UsageMetrics';

interface DashboardProps {
  className?: string;
}

const Dashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  const { user } = useAuth();
  const { subscription, usage, loading: subscriptionLoading } = useSubscription();
  const { systemStatus, loading: systemLoading } = useRAGSystem();
  
  const [activeTab, setActiveTab] = useState<'query' | 'analytics' | 'settings'>('query');

  if (subscriptionLoading || systemLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Enterprise RAG Intelligence
              </h1>
              <p className="text-gray-600 mt-1">
                Welcome back, {user?.name || 'User'}
              </p>
            </div>
            
            {/* System Status */}
            <div className="flex items-center space-x-4">
              <SystemStatusIndicator status={systemStatus} />
              <SubscriptionBadge subscription={subscription} />
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'query', name: 'Query Interface', icon: BoltIcon },
              { id: 'analytics', name: 'Analytics', icon: ChartBarIcon },
              { id: 'settings', name: 'Settings', icon: CpuChipIcon },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2`}
              >
                <tab.icon className="h-5 w-5" />
                <span>{tab.name}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'query' && (
          <div className="space-y-6">
            {/* Usage Warning */}
            {subscription && usage && (
              <UsageWarningBanner subscription={subscription} usage={usage} />
            )}
            
            {/* Query Interface */}
            <QueryInterface subscription={subscription} />
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <UsageMetrics usage={usage} subscription={subscription} />
            <PerformanceMetrics systemStatus={systemStatus} />
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="space-y-6">
            <SubscriptionCard subscription={subscription} />
            <SystemConfiguration />
          </div>
        )}
      </main>
    </div>
  );
};

// System Status Indicator Component
const SystemStatusIndicator: React.FC<{ status: any }> = ({ status }) => {
  const isHealthy = status?.gpu_providers_available > 0 && status?.api_gateway_healthy;
  
  return (
    <div className="flex items-center space-x-2">
      {isHealthy ? (
        <>
          <CheckCircleIcon className="h-5 w-5 text-green-500" />
          <span className="text-sm text-green-600 font-medium">System Online</span>
        </>
      ) : (
        <>
          <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
          <span className="text-sm text-yellow-600 font-medium">Limited Service</span>
        </>
      )}
    </div>
  );
};

// Subscription Badge Component
const SubscriptionBadge: React.FC<{ subscription: any }> = ({ subscription }) => {
  const getBadgeColor = (tier: string) => {
    switch (tier) {
      case 'free': return 'bg-gray-100 text-gray-800';
      case 'pro': return 'bg-blue-100 text-blue-800';
      case 'enterprise': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (!subscription) return null;

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getBadgeColor(subscription.tier)}`}>
      {subscription.name}
    </span>
  );
};

// Usage Warning Banner Component
const UsageWarningBanner: React.FC<{ subscription: any; usage: any }> = ({ subscription, usage }) => {
  const warningThreshold = 0.8; // 80%
  const criticalThreshold = 0.95; // 95%
  
  const calculateUsagePercentage = (used: number, limit: number) => {
    if (limit === -1) return 0; // Unlimited
    return used / limit;
  };

  const queryUsage = calculateUsagePercentage(usage?.queries || 0, subscription?.limits?.queries_per_day || 50);
  const isCritical = queryUsage >= criticalThreshold;
  const isWarning = queryUsage >= warningThreshold;

  if (!isWarning) return null;

  return (
    <div className={`rounded-md p-4 ${isCritical ? 'bg-red-50' : 'bg-yellow-50'}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          <ExclamationTriangleIcon className={`h-5 w-5 ${isCritical ? 'text-red-400' : 'text-yellow-400'}`} />
        </div>
        <div className="ml-3">
          <h3 className={`text-sm font-medium ${isCritical ? 'text-red-800' : 'text-yellow-800'}`}>
            {isCritical ? 'Usage Limit Almost Reached' : 'High Usage Warning'}
          </h3>
          <div className={`mt-2 text-sm ${isCritical ? 'text-red-700' : 'text-yellow-700'}`}>
            <p>
              You've used {Math.round(queryUsage * 100)}% of your daily query limit. 
              {isCritical ? ' Consider upgrading your plan to avoid service interruption.' : ' Monitor your usage to avoid limits.'}
            </p>
          </div>
          {(isCritical || isWarning) && subscription.tier === 'free' && (
            <div className="mt-4">
              <div className="-mx-2 -my-1.5 flex">
                <button
                  type="button"
                  className={`px-2 py-1.5 rounded-md text-sm font-medium ${
                    isCritical 
                      ? 'bg-red-100 text-red-800 hover:bg-red-200' 
                      : 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200'
                  }`}
                >
                  Upgrade to Pro
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Performance Metrics Component
const PerformanceMetrics: React.FC<{ systemStatus: any }> = ({ systemStatus }) => {
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">System Performance</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center">
            <CpuChipIcon className="h-8 w-8 text-blue-600" />
            <div className="ml-3">
              <p className="text-sm font-medium text-blue-900">GPU Providers</p>
              <p className="text-2xl font-bold text-blue-600">
                {systemStatus?.gpu_providers_available || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center">
            <BoltIcon className="h-8 w-8 text-green-600" />
            <div className="ml-3">
              <p className="text-sm font-medium text-green-900">Avg Response</p>
              <p className="text-2xl font-bold text-green-600">
                {systemStatus?.avg_response_time || '0.0'}s
              </p>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center">
            <GlobeAltIcon className="h-8 w-8 text-purple-600" />
            <div className="ml-3">
              <p className="text-sm font-medium text-purple-900">Success Rate</p>
              <p className="text-2xl font-bold text-purple-600">
                {systemStatus?.success_rate || '0'}%
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// System Configuration Component
const SystemConfiguration: React.FC = () => {
  const [config, setConfig] = useState({
    enableGPU: true,
    maxSources: 10,
    responseFormat: 'detailed',
    autoUpgrade: false
  });

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">System Configuration</h3>
      
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-gray-900">GPU Acceleration</h4>
            <p className="text-sm text-gray-500">Enable GPU-powered query processing</p>
          </div>
          <button
            type="button"
            onClick={() => setConfig(prev => ({ ...prev, enableGPU: !prev.enableGPU }))}
            className={`${
              config.enableGPU ? 'bg-blue-600' : 'bg-gray-200'
            } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
          >
            <span
              className={`${
                config.enableGPU ? 'translate-x-5' : 'translate-x-0'
              } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
            />
          </button>
        </div>

        <div>
          <label htmlFor="maxSources" className="block text-sm font-medium text-gray-700">
            Max Sources per Query
          </label>
          <select
            id="maxSources"
            value={config.maxSources}
            onChange={(e) => setConfig(prev => ({ ...prev, maxSources: parseInt(e.target.value) }))}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          >
            <option value={5}>5 sources</option>
            <option value={10}>10 sources</option>
            <option value={15}>15 sources</option>
            <option value={20}>20 sources</option>
          </select>
        </div>

        <div>
          <label htmlFor="responseFormat" className="block text-sm font-medium text-gray-700">
            Response Format
          </label>
          <select
            id="responseFormat"
            value={config.responseFormat}
            onChange={(e) => setConfig(prev => ({ ...prev, responseFormat: e.target.value }))}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          >
            <option value="concise">Concise</option>
            <option value="detailed">Detailed</option>
            <option value="bullet_points">Bullet Points</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;