import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { toast } from 'sonner';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export const AuthPage = () => {
  const { login, register } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    password: ''
  });

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      let result;
      if (isLogin) {
        result = await login(formData.email, formData.password);
      } else {
        result = await register(formData.name, formData.email, formData.phone, formData.password);
      }

      if (result.success) {
        toast.success(result.message);
        navigate('/dashboard');
      } else {
        toast.error(result.message);
      }
    } catch (error) {
      toast.error('Error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen futuristic-bg relative flex items-center justify-center">
      <div className="particles"></div>
      
      {/* Header */}
      <div className="absolute top-6 left-6 right-6 z-10">
        <div className="flex justify-between items-center">
          <Link to="/" className="flex items-center space-x-2">
            <img 
              src="https://customer-assets.emergentagent.com/job_a9e7c5a2-be4f-4a1e-8ab4-c5b1c9d69107/artifacts/tfw28vr3_Logo.png" 
              alt="Academy Logo" 
              className="h-10 w-10 rounded-lg"
            />
            <span className="text-xl font-bold text-gradient">ACADEMY</span>
          </Link>
          
          <div className="flex items-center space-x-4">
            <button 
              onClick={toggleLanguage}
              className="btn-ghost text-sm"
              data-testid="language-toggle"
            >
              <i className="fas fa-globe mr-2"></i>
              {language === 'es' ? 'EN' : 'ES'}
            </button>
            
            <button 
              onClick={toggleTheme}
              className="btn-ghost text-sm"
              data-testid="theme-toggle"
            >
              <i className={`fas ${theme === 'dark' ? 'fa-sun' : 'fa-moon'} mr-2`}></i>
            </button>
          </div>
        </div>
      </div>

      {/* Auth Form */}
      <div className="relative z-10 w-full max-w-md mx-auto px-6">
        <Card className="glass-card border-cyan-500/20">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl font-bold text-gradient mb-2">
              {isLogin ? t('loginTitle') : t('registerTitle')}
            </CardTitle>
            <p className="text-gray-400">
              {isLogin ? 'Ingresa tus credenciales' : 'Crea tu cuenta para comenzar'}
            </p>
          </CardHeader>
          
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {!isLogin && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">
                    {t('name')}
                  </label>
                  <Input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    placeholder="Tu nombre completo"
                    required={!isLogin}
                    className="bg-black/20 border-cyan-500/30 focus:border-cyan-500 text-white"
                    data-testid="name-input"
                  />
                </div>
              )}
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">
                  {t('email')}
                </label>
                <Input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  placeholder="tu@email.com"
                  required
                  className="bg-black/20 border-cyan-500/30 focus:border-cyan-500 text-white"
                  data-testid="email-input"
                />
              </div>
              
              {!isLogin && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">
                    {t('phone')}
                  </label>
                  <Input
                    type="tel"
                    name="phone"
                    value={formData.phone}
                    onChange={handleInputChange}
                    placeholder="+52 123 456 7890"
                    required={!isLogin}
                    className="bg-black/20 border-cyan-500/30 focus:border-cyan-500 text-white"
                    data-testid="phone-input"
                  />
                </div>
              )}
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">
                  {t('password')}
                </label>
                <Input
                  type="password"
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  placeholder="••••••••"
                  required
                  className="bg-black/20 border-cyan-500/30 focus:border-cyan-500 text-white"
                  data-testid="password-input"
                />
              </div>
              
              <Button 
                type="submit" 
                disabled={loading}
                className="w-full btn-futuristic py-3 text-lg"
                data-testid="auth-submit-btn"
              >
                {loading ? (
                  <div className="spinner"></div>
                ) : (
                  <>
                    <i className={`fas ${isLogin ? 'fa-sign-in-alt' : 'fa-user-plus'} mr-2`}></i>
                    {isLogin ? t('login') : t('register')}
                  </>
                )}
              </Button>
            </form>
            
            <div className="mt-6 text-center">
              <p className="text-gray-400">
                {isLogin ? '¿No tienes cuenta?' : '¿Ya tienes cuenta?'}
              </p>
              <button
                onClick={() => setIsLogin(!isLogin)}
                className="text-cyan-400 hover:text-cyan-300 font-medium mt-2 transition-colors"
                data-testid="toggle-auth-mode"
              >
                {isLogin ? t('register') : t('login')}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
