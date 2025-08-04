import React, { useState } from 'react';

interface NavigationBarProps {
  user: any;
  currentTab: string;
  onTabChange: (tab: string) => void;
  onLogout: () => void;
}

export default function NavigationBar({ user, currentTab, onTabChange, onLogout }: NavigationBarProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);

  const tabs = [
    { id: 'query', label: 'RAG Query', icon: '🤖' },
    { id: 'documents', label: 'Documents', icon: '📄' },
    { id: 'analytics', label: 'Analytics', icon: '📊' },
    { id: 'settings', label: 'Settings', icon: '⚙️' }
  ];

  return (
    <nav style={{
      backgroundColor: 'white',
      borderBottom: '1px solid #e5e7eb',
      padding: '12px 0',
      boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
    }}>
      <div className="container flex justify-between items-center">
        {/* Logo and Title */}
        <div className="flex items-center" style={{ gap: '16px' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1f2937' }}>
            🧠 Enterprise RAG
          </div>
          <div style={{ 
            backgroundColor: '#dbeafe', 
            color: '#1e40af', 
            padding: '4px 8px', 
            borderRadius: '12px', 
            fontSize: '12px',
            fontWeight: '500'
          }}>
            v2.0 Free Models
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex" style={{ gap: '8px' }}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              style={{
                padding: '8px 16px',
                borderRadius: '6px',
                border: 'none',
                backgroundColor: currentTab === tab.id ? '#3b82f6' : 'transparent',
                color: currentTab === tab.id ? 'white' : '#6b7280',
                fontWeight: currentTab === tab.id ? '500' : 'normal',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                if (currentTab !== tab.id) {
                  e.currentTarget.style.backgroundColor = '#f3f4f6';
                }
              }}
              onMouseLeave={(e) => {
                if (currentTab !== tab.id) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              <span style={{ marginRight: '8px' }}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* User Menu */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            style={{
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid #d1d5db',
              backgroundColor: 'white',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              backgroundColor: '#3b82f6',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '14px',
              fontWeight: 'bold'
            }}>
              {user?.name?.charAt(0)?.toUpperCase() || 'U'}
            </div>
            <div style={{ textAlign: 'left' }}>
              <div style={{ fontSize: '14px', fontWeight: '500', color: '#1f2937' }}>
                {user?.name || 'User'}
              </div>
              <div style={{ fontSize: '12px', color: '#6b7280' }}>
                {user?.subscription_tier || 'free'} plan
              </div>
            </div>
            <span style={{ fontSize: '12px', color: '#6b7280' }}>▼</span>
          </button>

          {showUserMenu && (
            <div style={{
              position: 'absolute',
              right: 0,
              top: '100%',
              marginTop: '4px',
              backgroundColor: 'white',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              minWidth: '200px',
              zIndex: 50
            }}>
              <div style={{ padding: '12px', borderBottom: '1px solid #e5e7eb' }}>
                <div style={{ fontSize: '14px', fontWeight: '500', color: '#1f2937' }}>
                  {user?.email}
                </div>
                <div style={{ fontSize: '12px', color: '#6b7280', marginTop: '2px' }}>
                  Subscription: {user?.subscription_tier || 'Free'}
                </div>
              </div>
              
              <div style={{ padding: '8px 0' }}>
                <button
                  onClick={() => {
                    onTabChange('settings');
                    setShowUserMenu(false);
                  }}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    textAlign: 'left',
                    border: 'none',
                    backgroundColor: 'transparent',
                    cursor: 'pointer',
                    fontSize: '14px',
                    color: '#374151'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  ⚙️ Account Settings
                </button>
                
                <button
                  onClick={() => {
                    onTabChange('analytics');
                    setShowUserMenu(false);
                  }}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    textAlign: 'left',
                    border: 'none',
                    backgroundColor: 'transparent',
                    cursor: 'pointer',
                    fontSize: '14px',
                    color: '#374151'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  📊 Usage Analytics
                </button>
                
                <hr style={{ margin: '8px 0', border: 'none', borderTop: '1px solid #e5e7eb' }} />
                
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    onLogout();
                  }}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    textAlign: 'left',
                    border: 'none',
                    backgroundColor: 'transparent',
                    cursor: 'pointer',
                    fontSize: '14px',
                    color: '#dc2626'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#fef2f2'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  🚪 Logout
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}