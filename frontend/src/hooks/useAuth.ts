import { useState, useEffect } from 'react';

interface User {
  id: string;
  email: string;
  name: string;
  subscription_tier: string;
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

export function useAuth() {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    isLoading: true,
    isAuthenticated: false,
  });

  useEffect(() => {
    // Check for existing token on load
    const token = localStorage.getItem('auth_token');
    if (token) {
      setAuthState({
        user: {
          id: 'demo-user-123',
          email: 'demo@example.com',
          name: 'Demo User',
          subscription_tier: 'free'
        },
        isLoading: false,
        isAuthenticated: true,
      });
    } else {
      setAuthState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  const login = async (userData: any) => {
    setAuthState({
      user: userData,
      isLoading: false,
      isAuthenticated: true,
    });
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setAuthState({
      user: null,
      isLoading: false,
      isAuthenticated: false,
    });
  };

  return {
    ...authState,
    login,
    logout,
  };
}