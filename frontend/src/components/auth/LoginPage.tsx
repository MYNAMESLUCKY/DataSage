import React, { useState } from 'react';
import { apiClient } from '../../utils/apiClient';

interface LoginPageProps {
  onLogin: (user: any) => void;
  onSwitchToSignup: () => void;
}

export default function LoginPage({ onLogin, onSwitchToSignup }: LoginPageProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          password
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed');
      }

      const data = await response.json();
      
      const { user, token } = data;
      localStorage.setItem('auth_token', token);
      onLogin(user);
    } catch (err: any) {
      console.error('Login error:', err);
      setError(err.message || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoLogin = () => {
    // Demo login for testing
    const demoUser = {
      id: 'demo-user-123',
      email: 'demo@example.com',
      name: 'Demo User',
      subscription_tier: 'free'
    };
    localStorage.setItem('auth_token', 'demo-token');
    onLogin(demoUser);
  };

  return (
    <div className="container" style={{ maxWidth: '400px', margin: '50px auto' }}>
      <div className="card">
        <h1 className="text-2xl font-bold mb-4 text-gray-900">Login to Enterprise RAG</h1>
        
        {error && (
          <div style={{ 
            backgroundColor: '#fee2e2', 
            color: '#dc2626', 
            padding: '12px', 
            borderRadius: '6px', 
            marginBottom: '16px' 
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="text-gray-900 font-semibold">Email:</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              className="input"
              required
              disabled={isLoading}
            />
          </div>

          <div>
            <label htmlFor="password" className="text-gray-900 font-semibold">Password:</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="input"
              required
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="button"
            style={{ width: '100%' }}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div style={{ textAlign: 'center', margin: '20px 0' }}>
          <span style={{ color: '#6b7280' }}>or</span>
        </div>

        <button
          onClick={handleDemoLogin}
          className="button"
          style={{ 
            width: '100%', 
            backgroundColor: '#10b981',
            marginBottom: '16px'
          }}
        >
          Try Demo Account
        </button>

        <div style={{ textAlign: 'center' }}>
          <span className="text-gray-600">Don't have an account? </span>
          <button
            onClick={onSwitchToSignup}
            style={{
              background: 'none',
              border: 'none',
              color: '#3b82f6',
              textDecoration: 'underline',
              cursor: 'pointer'
            }}
          >
            Sign up
          </button>
        </div>
      </div>
    </div>
  );
}