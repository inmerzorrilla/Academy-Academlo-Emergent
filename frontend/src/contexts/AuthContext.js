import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('academy_token'));
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      // Set default authorization header
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      // Verify token and get user info
      verifyToken();
    } else {
      setLoading(false);
    }

    // Remove Emergent badge periodically
    const removeEmergentBadge = () => {
      const selectors = [
        'div[style*="Made with Emergent"]',
        '[data-emergent-badge]',
        '.emergent-badge',
        '#emergent-badge',
        'div[style*="position: fixed"][style*="bottom"][style*="right"]:not(.chat-widget)',
        'div[style*="z-index: 9999"]'
      ];
      
      selectors.forEach(selector => {
        const elements = document.querySelectorAll(selector);
        elements.forEach(el => {
          if (el && !el.classList.contains('chat-widget') && !el.dataset.testid?.includes('chat')) {
            el.style.display = 'none';
            el.remove();
          }
        });
      });
    };

    // Run immediately and then every 2 seconds
    removeEmergentBadge();
    const interval = setInterval(removeEmergentBadge, 2000);

    return () => clearInterval(interval);
  }, [token]);

  const verifyToken = async () => {
    try {
      const response = await axios.get(`${API}/user/profile`);
      setUser(response.data);
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Token verification failed:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      const response = await axios.post(`${API}/auth/login`, {
        email,
        password
      });
      
      const { user: userData, token: userToken } = response.data;
      
      setUser(userData);
      setToken(userToken);
      setIsAuthenticated(true);
      
      localStorage.setItem('academy_token', userToken);
      axios.defaults.headers.common['Authorization'] = `Bearer ${userToken}`;
      
      return { success: true, message: response.data.message };
    } catch (error) {
      return { 
        success: false, 
        message: error.response?.data?.detail || 'Login failed' 
      };
    }
  };

  const register = async (name, email, phone, password) => {
    try {
      const response = await axios.post(`${API}/auth/register`, {
        name,
        email,
        phone,
        password
      });
      
      const { user: userData, token: userToken } = response.data;
      
      setUser(userData);
      setToken(userToken);
      setIsAuthenticated(true);
      
      localStorage.setItem('academy_token', userToken);
      axios.defaults.headers.common['Authorization'] = `Bearer ${userToken}`;
      
      return { success: true, message: response.data.message };
    } catch (error) {
      return { 
        success: false, 
        message: error.response?.data?.detail || 'Registration failed' 
      };
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    setIsAuthenticated(false);
    localStorage.removeItem('academy_token');
    delete axios.defaults.headers.common['Authorization'];
  };

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    login,
    register,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
