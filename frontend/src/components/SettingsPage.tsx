import React, { useState } from 'react';

interface SettingsPageProps {
  user: any;
  className?: string;
}

export default function SettingsPage({ user, className }: SettingsPageProps) {
  const [settings, setSettings] = useState({
    enableGpuAcceleration: false,
    maxSources: 5,
    searchWeb: true,
    defaultModel: 'llama-3.2-7b',
    theme: 'light'
  });

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Settings</h2>
        
        <div className="space-y-8">
          {/* Account Settings */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Account Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={user?.name || ''}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  disabled
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={user?.email || ''}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  disabled
                />
              </div>
            </div>
          </div>

          {/* Query Settings */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Query Settings</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Enable GPU Acceleration</label>
                  <p className="text-sm text-gray-500">Use GPU for faster query processing</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.enableGpuAcceleration}
                  onChange={(e) => handleSettingChange('enableGpuAcceleration', e.target.checked)}
                  className="h-4 w-4 text-blue-600 rounded"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Enable Web Search</label>
                  <p className="text-sm text-gray-500">Include real-time web results in queries</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.searchWeb}
                  onChange={(e) => handleSettingChange('searchWeb', e.target.checked)}
                  className="h-4 w-4 text-blue-600 rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Maximum Sources</label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={settings.maxSources}
                  onChange={(e) => handleSettingChange('maxSources', parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-sm text-gray-500">
                  <span>1</span>
                  <span>Current: {settings.maxSources}</span>
                  <span>20</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Default Model</label>
                <select
                  value={settings.defaultModel}
                  onChange={(e) => handleSettingChange('defaultModel', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="llama-3.2-7b">Llama 3.2 7B (Free)</option>
                  <option value="mistral-7b">Mistral 7B (Free)</option>
                  <option value="gemma-7b">Gemma 7B (Free)</option>
                  <option value="qwen-7b">Qwen 7B (Free)</option>
                  <option value="deepseek-coder">DeepSeek Coder (Free)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Subscription */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Subscription</h3>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium text-gray-900">Current Plan: {user?.subscription_tier || 'Free'}</p>
                  <p className="text-sm text-gray-600">50 queries per day, Free LLM models</p>
                </div>
                <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  Upgrade Plan
                </button>
              </div>
            </div>
          </div>

          {/* Save Button */}
          <div className="pt-4 border-t">
            <button className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700">
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}