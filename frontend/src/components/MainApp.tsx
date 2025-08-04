import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import NavigationBar from './layout/NavigationBar';
import LoginPage from './auth/LoginPage';
import SignupPage from './auth/SignupPage';
import QueryInterface from './QueryInterface';
import DocumentManager from './DocumentManager';
import AnalyticsPage from './AnalyticsPage';
import SettingsPage from './SettingsPage';
import LoadingSpinner from './LoadingSpinner';

export default function MainApp() {
  const { user, isLoading, isAuthenticated, login, logout } = useAuth();
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [currentTab, setCurrentTab] = useState('query');

  // Check for existing token on app load
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token && !isAuthenticated) {
      // Auto-login with demo user if token exists
      const demoUser = {
        id: 'demo-user-123',
        email: 'demo@example.com',
        name: 'Demo User',
        subscription_tier: 'free'
      };
      login(demoUser);
    }
  }, []);

  const handleLogin = (userData: any) => {
    login(userData);
  };

  const handleSignup = (userData: any) => {
    login(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    logout();
    setCurrentTab('query');
  };

  if (isLoading) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        minHeight: '100vh' 
      }}>
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb' }}>
        {authMode === 'login' ? (
          <LoginPage 
            onLogin={handleLogin}
            onSwitchToSignup={() => setAuthMode('signup')}
          />
        ) : (
          <SignupPage 
            onSignup={handleSignup}
            onSwitchToLogin={() => setAuthMode('login')}
          />
        )}
      </div>
    );
  }

  const renderCurrentTab = () => {
    switch (currentTab) {
      case 'query':
        return <QueryInterface user={user} />;
      case 'documents':
        return <DocumentManager user={user} />;
      case 'analytics':
        return <AnalyticsPage user={user} />;
      case 'settings':
        return <SettingsPage user={user} />;
      default:
        return <QueryInterface user={user} />;
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb' }}>
      <NavigationBar
        user={user}
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        onLogout={handleLogout}
      />
      
      <main style={{ paddingTop: '20px' }}>
        {renderCurrentTab()}
      </main>
    </div>
  );
}