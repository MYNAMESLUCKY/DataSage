import axios from 'axios';

// Create axios instance with base configuration
export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    // Get auth token from localStorage or your auth system
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common error scenarios
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_token');
        window.location.href = '/login';
      }
    } else if (error.response?.status === 403) {
      // Forbidden - subscription issue
      console.error('Access forbidden - check subscription');
    } else if (error.response?.status === 429) {
      // Too many requests - rate limited
      console.error('Rate limit exceeded');
    } else if (error.response?.status >= 500) {
      // Server error
      console.error('Server error occurred');
    }

    return Promise.reject(error);
  }
);

// API endpoints
export const endpoints = {
  // Authentication
  auth: {
    login: '/auth/login',
    logout: '/auth/logout',
    refresh: '/auth/refresh',
    register: '/auth/register',
  },

  // RAG System
  query: {
    process: '/query/process',
    history: '/query/history',
  },

  // Models
  models: {
    available: '/models/available',
    status: '/models/status',
  },

  // System
  system: {
    status: '/system/status',
    health: '/system/health',
    metrics: '/system/metrics',
  },

  // Subscription
  subscription: {
    current: '/subscription/current',
    plans: '/subscription/plans',
    usage: '/subscription/usage',
    upgrade: '/subscription/upgrade',
    cancel: '/subscription/cancel',
  },

  // Admin
  admin: {
    dashboard: '/admin/dashboard',
    users: '/admin/users',
    analytics: '/admin/analytics',
  },
};

// Utility functions for common API patterns
export const apiUtils = {
  // Handle file uploads
  uploadFile: async (endpoint: string, file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData();
    formData.append('file', file);

    return apiClient.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });
  },

  // Handle pagination
  getPaginated: async (endpoint: string, page: number = 1, limit: number = 10, filters?: Record<string, any>) => {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...filters,
    });

    return apiClient.get(`${endpoint}?${params}`);
  },

  // Handle batch operations
  batch: async (requests: Array<{ method: string; url: string; data?: any }>) => {
    const promises = requests.map(req => {
      switch (req.method.toLowerCase()) {
        case 'get':
          return apiClient.get(req.url);
        case 'post':
          return apiClient.post(req.url, req.data);
        case 'put':
          return apiClient.put(req.url, req.data);
        case 'delete':
          return apiClient.delete(req.url);
        default:
          throw new Error(`Unsupported method: ${req.method}`);
      }
    });

    return Promise.allSettled(promises);
  },

  // Retry mechanism for failed requests
  retry: async <T>(
    apiCall: () => Promise<T>,
    maxRetries: number = 3,
    delay: number = 1000
  ): Promise<T> => {
    let lastError: any;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await apiCall();
      } catch (error) {
        lastError = error;
        
        if (attempt === maxRetries) {
          break;
        }

        // Don't retry on client errors (4xx)
        if (axios.isAxiosError(error) && error.response?.status && error.response.status < 500) {
          break;
        }

        // Wait before retrying with exponential backoff
        await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, attempt - 1)));
      }
    }

    throw lastError;
  },
};

export default apiClient;