import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { ChatWidget } from './ChatWidget';
import { Footer } from './Footer';
import { Progress } from './ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const Dashboard = () => {
  const { user, logout } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  
  const [progress, setProgress] = useState({
    teorico_progress: 0,
    escucha_progress: 0,
    prompt_progress: 0,
    proyecto_progress: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProgress();
  }, []);

  const fetchProgress = async () => {
    try {
      const response = await axios.get(`${API}/user/progress`);
      setProgress(response.data);
    } catch (error) {
      console.error('Error fetching progress:', error);
      toast.error('Error al cargar el progreso');
    } finally {
      setLoading(false);
    }
  };

  const getTotalProgress = () => {
    return Math.round((progress.teorico_progress + progress.escucha_progress + 
                     progress.prompt_progress + progress.proyecto_progress) / 4);
  };

  const downloadCertificate = async () => {
    if (getTotalProgress() === 100) {
      try {
        const response = await axios.get(`${API}/user/certificate`, {
          responseType: 'blob'
        });
        
        // Create download link
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'academy_certificate.pdf');
        document.body.appendChild(link);
        link.click();
        link.remove();
        
        toast.success('¡Certificado descargado exitosamente! 🎉');
      } catch (error) {
        console.error('Error downloading certificate:', error);
        toast.error('Error al descargar el certificado');
      }
    } else {
      toast.error('Completa todos los módulos para obtener tu certificado');
    }
  };

  const modules = [
    {
      id: 'teorico',
      title: t('teorico'),
      description: t('teoricoDesc'),
      icon: 'fa-brain',
      progress: progress.teorico_progress,
      color: 'from-blue-500 to-cyan-500',
      route: '/module/teorico'
    },
    {
      id: 'escucha',
      title: t('escucha'),
      description: t('escuchaDesc'),
      icon: 'fa-headphones',
      progress: progress.escucha_progress,
      color: 'from-green-500 to-teal-500',
      route: '/module/escucha'
    },
    {
      id: 'prompt',
      title: t('prompt'),
      description: t('promptDesc'),
      icon: 'fa-terminal',
      progress: progress.prompt_progress,
      color: 'from-purple-500 to-pink-500',
      route: '/module/prompt'
    },
    {
      id: 'proyecto',
      title: t('proyecto'),
      description: t('proyectoDesc'),
      icon: 'fa-project-diagram',
      progress: progress.proyecto_progress,
      color: 'from-orange-500 to-red-500',
      route: '/module/proyecto'
    }
  ];

  if (loading) {
    return (
      <div className="min-h-screen futuristic-bg flex items-center justify-center">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen futuristic-bg relative" data-testid="student-dashboard">
      <div className="particles"></div>
      
      {/* Header */}
      <header className="relative z-10 p-6 border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <img 
              src="https://customer-assets.emergentagent.com/job_a9e7c5a2-be4f-4a1e-8ab4-c5b1c9d69107/artifacts/tfw28vr3_Logo.png" 
              alt="Academy Logo" 
              className="h-12 w-12 rounded-lg"
            />
            <div>
              <h1 className="text-2xl font-bold text-gradient">ACADEMY</h1>
              <p className="text-sm text-cyan-400">Bienvenido, {user?.name}</p>
            </div>
          </div>
          
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
            
            {user?.is_admin && (
              <Link to="/admin" className="btn-ghost text-sm">
                <i className="fas fa-cog mr-2"></i>
                {t('admin')}
              </Link>
            )}
            
            <button 
              onClick={logout}
              className="btn-ghost text-sm text-red-400 border-red-400 hover:bg-red-400"
            >
              <i className="fas fa-sign-out-alt mr-2"></i>
              {t('logout')}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Welcome Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-gradient glitch" data-text={t('welcomeCommander')}>
            {t('welcomeCommander')}
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            {t('academloEmergentWelcome')}
          </p>
          
          {/* Overall Progress */}
          <Card className="glass-card max-w-2xl mx-auto">
            <CardHeader>
              <CardTitle className="text-2xl text-gradient">
                <span className="text-primary">{t('overallProgress')}: </span>
                <span className="text-cyan-400">{getTotalProgress()}%</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-6">
                <Progress value={getTotalProgress()} className="h-4" />
              </div>
              <p className="text-gray-400 mb-4">
                {t('progressDescription')}
              </p>
              
              {getTotalProgress() === 100 ? (
                <Button 
                  onClick={downloadCertificate}
                  className="btn-futuristic"
                  data-testid="download-certificate-btn"
                >
                  <i className="fas fa-download mr-2"></i>
                  {t('downloadCertificate')}
                </Button>
              ) : (
                <p className="text-cyan-400">
                  {t('continueProgress')}
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Modules Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          {modules.map((module) => (
            <Link 
              key={module.id} 
              to={module.route}
              className="block"
              data-testid={`module-${module.id}`}
            >
              <Card className={`module-card h-full ${module.progress === 100 ? 'completed' : ''}`}>
                <CardHeader className="text-center">
                  <div className={`w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r ${module.color} flex items-center justify-center`}>
                    <i className={`fas ${module.icon} text-2xl text-white`}></i>
                  </div>
                  <CardTitle className="text-xl text-gradient">
                    {module.progress}% {module.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-4">
                    <Progress value={module.progress} className="h-2" />
                  </div>
                  <p className="text-gray-400 text-sm mb-4">{module.description}</p>
                  
                  {module.progress === 100 ? (
                    <div className="flex items-center justify-center text-green-400">
                      <i className="fas fa-check-circle mr-2"></i>
                      Completado
                    </div>
                  ) : (
                    <div className="text-center">
                      <Button className="btn-ghost text-sm w-full">
                        Continuar
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>

        {/* Academlo Advertisement */}
        <Card className="glass-card mb-8">
          <CardContent className="p-8 text-center">
            <h3 className="text-2xl font-bold mb-4 text-gradient">
              {t('completeProgrAcademlo')}
            </h3>
            <p className="text-xl mb-6 text-gray-300">
              {t('studyAcademlo')}
            </p>
            <a 
              href="https://www.academlo.com/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-futuristic"
              data-testid="academlo-ad-link"
            >
              <i className="fas fa-external-link-alt mr-2"></i>
              {t('visitAcademlo')}
            </a>
          </CardContent>
        </Card>
      </main>

      <Footer />

      <ChatWidget />
    </div>
  );
};
