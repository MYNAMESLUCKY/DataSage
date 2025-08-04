import React, { useState } from 'react';

interface SettingsPageProps {
  user: any;
}

export default function SettingsPage({ user }: SettingsPageProps) {
  const [settings, setSettings] = useState({
    email_notifications: true,
    api_notifications: false,
    max_sources: 5,
    default_model: 'llama-3.2-7b',
    enable_web_search: true,
    response_format: 'detailed'
  });

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    // Save settings
    alert('Settings saved successfully!');
  };

  return (
    <div className="container">
      <div className="card">
        <h2 className="text-xl font-bold mb-4 text-gray-900">Account Settings</h2>
        <p className="text-gray-600 mb-6">
          Manage your account preferences and RAG system configuration
        </p>

        {/* Account Information */}
        <div className="card" style={{ marginBottom: '24px' }}>
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Account Information</h3>
          
          <div className="grid grid-cols-2" style={{ gap: '20px' }}>
            <div>
              <label className="text-gray-900 font-semibold">Name:</label>
              <input
                type="text"
                defaultValue={user?.name || 'Demo User'}
                className="input"
                style={{ marginTop: '4px' }}
              />
            </div>
            
            <div>
              <label className="text-gray-900 font-semibold">Email:</label>
              <input
                type="email"
                defaultValue={user?.email || 'demo@example.com'}
                className="input"
                style={{ marginTop: '4px' }}
              />
            </div>
          </div>
          
          <div style={{ marginTop: '16px' }}>
            <label className="text-gray-900 font-semibold">Subscription Plan:</label>
            <div style={{ 
              padding: '12px', 
              backgroundColor: '#f3f4f6', 
              borderRadius: '6px',
              marginTop: '4px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span className="font-semibold text-gray-900">
                {user?.subscription_tier?.toUpperCase() || 'FREE'} PLAN
              </span>
              <button 
                className="button"
                style={{ padding: '6px 12px', fontSize: '12px' }}
              >
                Upgrade
              </button>
            </div>
          </div>
        </div>

        {/* RAG Configuration */}
        <div className="card" style={{ marginBottom: '24px' }}>
          <h3 className="text-lg font-semibold mb-4 text-gray-900">RAG Configuration</h3>
          
          <div className="space-y-4">
            <div>
              <label className="text-gray-900 font-semibold">Default Model:</label>
              <select 
                value={settings.default_model}
                onChange={(e) => handleSettingChange('default_model', e.target.value)}
                className="input"
                style={{ marginTop: '4px' }}
              >
                <option value="llama-3.2-7b">Llama 3.2 7B (Free)</option>
                <option value="mistral-7b">Mistral 7B (Free)</option>
                <option value="gemma-2b">Gemma 2B (Free)</option>
                <option value="qwen-2.5-7b">Qwen 2.5 7B (Free)</option>
              </select>
            </div>
            
            <div>
              <label className="text-gray-900 font-semibold">Max Sources per Query:</label>
              <select 
                value={settings.max_sources}
                onChange={(e) => handleSettingChange('max_sources', parseInt(e.target.value))}
                className="input"
                style={{ marginTop: '4px' }}
              >
                <option value={3}>3 sources</option>
                <option value={5}>5 sources (Recommended)</option>
                <option value={10}>10 sources (Pro)</option>
                <option value={20}>20 sources (Enterprise)</option>
              </select>
            </div>
            
            <div>
              <label className="text-gray-900 font-semibold">Response Format:</label>
              <select 
                value={settings.response_format}
                onChange={(e) => handleSettingChange('response_format', e.target.value)}
                className="input"
                style={{ marginTop: '4px' }}
              >
                <option value="concise">Concise</option>
                <option value="detailed">Detailed (Recommended)</option>
                <option value="technical">Technical</option>
              </select>
            </div>
          </div>
        </div>

        {/* Preferences */}
        <div className="card" style={{ marginBottom: '24px' }}>
          <h3 className="text-lg font-semibold mb-4 text-gray-900">Preferences</h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <div className="font-semibold text-gray-900">Enable Web Search</div>
                <div style={{ fontSize: '14px', color: '#6b7280' }}>
                  Include real-time web results in responses
                </div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: '44px', height: '24px' }}>
                <input
                  type="checkbox"
                  checked={settings.enable_web_search}
                  onChange={(e) => handleSettingChange('enable_web_search', e.target.checked)}
                  style={{ opacity: 0, width: 0, height: 0 }}
                />
                <span style={{
                  position: 'absolute',
                  cursor: 'pointer',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: settings.enable_web_search ? '#3b82f6' : '#ccc',
                  transition: '0.4s',
                  borderRadius: '24px'
                }}>
                  <span style={{
                    position: 'absolute',
                    content: '""',
                    height: '18px',
                    width: '18px',
                    left: settings.enable_web_search ? '23px' : '3px',
                    bottom: '3px',
                    backgroundColor: 'white',
                    transition: '0.4s',
                    borderRadius: '50%'
                  }}></span>
                </span>
              </label>
            </div>
            
            <div className="flex justify-between items-center">
              <div>
                <div className="font-semibold text-gray-900">Email Notifications</div>
                <div style={{ fontSize: '14px', color: '#6b7280' }}>
                  Receive updates about your usage and new features
                </div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: '44px', height: '24px' }}>
                <input
                  type="checkbox"
                  checked={settings.email_notifications}
                  onChange={(e) => handleSettingChange('email_notifications', e.target.checked)}
                  style={{ opacity: 0, width: 0, height: 0 }}
                />
                <span style={{
                  position: 'absolute',
                  cursor: 'pointer',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: settings.email_notifications ? '#3b82f6' : '#ccc',
                  transition: '0.4s',
                  borderRadius: '24px'
                }}>
                  <span style={{
                    position: 'absolute',
                    content: '""',
                    height: '18px',
                    width: '18px',
                    left: settings.email_notifications ? '23px' : '3px',
                    bottom: '3px',
                    backgroundColor: 'white',
                    transition: '0.4s',
                    borderRadius: '50%'
                  }}></span>
                </span>
              </label>
            </div>
            
            <div className="flex justify-between items-center">
              <div>
                <div className="font-semibold text-gray-900">API Notifications</div>
                <div style={{ fontSize: '14px', color: '#6b7280' }}>
                  Get notified about API rate limits and errors
                </div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: '44px', height: '24px' }}>
                <input
                  type="checkbox"
                  checked={settings.api_notifications}
                  onChange={(e) => handleSettingChange('api_notifications', e.target.checked)}
                  style={{ opacity: 0, width: 0, height: 0 }}
                />
                <span style={{
                  position: 'absolute',
                  cursor: 'pointer',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: settings.api_notifications ? '#3b82f6' : '#ccc',
                  transition: '0.4s',
                  borderRadius: '24px'
                }}>
                  <span style={{
                    position: 'absolute',
                    content: '""',
                    height: '18px',
                    width: '18px',
                    left: settings.api_notifications ? '23px' : '3px',
                    bottom: '3px',
                    backgroundColor: 'white',
                    transition: '0.4s',
                    borderRadius: '50%'
                  }}></span>
                </span>
              </label>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex justify-between items-center">
          <button 
            onClick={handleSave}
            className="button"
            style={{ padding: '12px 24px' }}
          >
            Save Settings
          </button>
          
          <button 
            className="button"
            style={{ 
              backgroundColor: '#dc2626',
              padding: '12px 24px'
            }}
          >
            Delete Account
          </button>
        </div>
      </div>
    </div>
  );
}