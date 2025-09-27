import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { ChatWidget } from '../ChatWidget';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const ModuleProyecto = () => {
  const { user } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const [progress, setProgress] = useState({ proyecto_progress: 0, proyecto_url: '' });
  const [loading, setLoading] = useState(true);
  const [projectUrl, setProjectUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchProgress();
  }, []);

  const fetchProgress = async () => {
    try {
      const response = await axios.get(`${API}/user/progress`);
      setProgress(response.data);
      if (response.data.proyecto_url) {
        setProjectUrl(response.data.proyecto_url);
      }
    } catch (error) {
      console.error('Error fetching progress:', error);
    } finally {
      setLoading(false);
    }
  };

  const submitProject = async () => {
    if (!projectUrl.trim()) {
      toast.error(t('pleaseEnterProjectUrl'));
      return;
    }

    // Validate Emergent URL
    if (!projectUrl.includes('emergent') && !projectUrl.includes('app.emergent.sh')) {
      toast.error(t('urlMustBeFromEmergent'));
      return;
    }

    setSubmitting(true);
    try {
      const response = await axios.post(`${API}/user/progress`, {
        module: 'proyecto',
        proyecto_url: projectUrl
      });
      
      setProgress(response.data);
      toast.success(t('projectSubmittedSuccessfully'));
    } catch (error) {
      console.error('Error updating progress:', error);
      toast.error(t('errorSubmittingProject'));
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen futuristic-bg flex items-center justify-center">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen futuristic-bg relative">
      <div className="particles"></div>
      
      {/* Header */}
      <header className="relative z-10 p-6 border-b border-gray-800">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <Link to="/dashboard" className="btn-ghost text-sm">
              <i className="fas fa-arrow-left mr-2"></i>
              Volver al Dashboard
            </Link>
          </div>
          
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gradient">Módulo Proyecto</h1>
            <p className="text-sm text-cyan-400">Proyecto Final</p>
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
            
            <Badge className="bg-orange-500">
              {progress.proyecto_progress}% Completado
            </Badge>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="relative z-10 max-w-4xl mx-auto px-6 py-8">
        {/* Progress Section */}
        <Card className="glass-card mb-8">
          <CardHeader>
            <CardTitle className="text-xl text-gradient flex items-center">
              <i className="fas fa-project-diagram mr-2"></i>
              Progreso del Módulo Proyecto
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <Progress value={progress.proyecto_progress} className="h-3" />
            </div>
            <p className="text-gray-400">
              {progress.proyecto_progress === 100 ? 'Proyecto completado y enviado' : 'Pendiente de completar el proyecto'}
            </p>
          </CardContent>
        </Card>

        {/* Instructions */}
        <Card className="glass-card mb-8">
          <CardHeader>
            <CardTitle className="text-2xl text-gradient flex items-center">
              <i className="fas fa-rocket mr-2"></i>
              {t('createProjectEmergent')}
            </CardTitle>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 p-6 rounded-lg border border-cyan-500/20">
              <h3 className="text-xl font-semibold text-cyan-400 mb-4">
                🎆 {t('timeToCreateSomethingIncredible')}
              </h3>
              
              <div className="space-y-4 text-gray-300">
                <p className="text-lg leading-relaxed">
                  🚀 <strong>{t('yourMission')}:</strong> {t('useEmergentToCreateProject')} <span className="text-cyan-400 font-semibold">Emergent</span> {t('toDemonstrateWhatYouLearned')}.
                </p>
                
                <div className="bg-gray-800/50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-400 mb-3 flex items-center">
                    <i className="fas fa-lightbulb mr-2"></i>
                    {t('projectIdeas')}:
                  </h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start">
                      <span className="text-cyan-400 mr-2">•</span>
                      {t('projectIdea1')}
                    </li>
                    <li className="flex items-start">
                      <span className="text-cyan-400 mr-2">•</span>
                      {t('projectIdea2')}
                    </li>
                    <li className="flex items-start">
                      <span className="text-cyan-400 mr-2">•</span>
                      {t('projectIdea3')}
                    </li>
                    <li className="flex items-start">
                      <span className="text-cyan-400 mr-2">•</span>
                      {t('projectIdea4')}
                    </li>
                  </ul>
                </div>
                
                <div className="bg-purple-900/30 p-4 rounded-lg border border-purple-500/30">
                  <h4 className="font-semibold text-purple-400 mb-2 flex items-center">
                    <i className="fas fa-star mr-2"></i>
                    {t('requirements')}:
                  </h4>
                  <ul className="space-y-1 text-sm">
                    <li>✓ {t('requirement1')}</li>
                    <li>✓ {t('requirement2')}</li>
                    <li>✓ {t('requirement3')}</li>
                  </ul>
                </div>
              </div>
            </div>
            
            {/* Emergent Access */}
            <div className="text-center">
              <a 
                href="https://app.emergent.sh/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-futuristic text-lg px-8 py-4 inline-block"
                data-testid="emergent-link"
              >
                <i className="fas fa-external-link-alt mr-3"></i>
                {t('goToEmergentToCreateProject')}
              </a>
              <p className="text-gray-400 mt-3 text-sm">
                Una vez que hayas creado tu proyecto, regresa aquí para enviar la URL
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Project Submission */}
        <Card className={`glass-card mb-8 ${progress.proyecto_progress === 100 ? 'border-green-500/50' : ''}`}>
          <CardHeader>
            <CardTitle className="text-xl text-gradient flex items-center">
              <i className="fas fa-upload mr-2"></i>
              Enviar tu Proyecto
            </CardTitle>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {progress.proyecto_progress === 100 ? (
              <div className="text-center space-y-4">
                <div className="text-6xl text-green-400 mb-4">
                  ✅
                </div>
                <h3 className="text-2xl font-bold text-green-400">
                  ¡Proyecto Enviado Exitosamente!
                </h3>
                <p className="text-gray-300">
                  Tu proyecto ha sido registrado correctamente:
                </p>
                <div className="bg-gray-800/50 p-4 rounded-lg">
                  <a 
                    href={progress.proyecto_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 break-all"
                  >
                    {progress.proyecto_url}
                  </a>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    URL de tu proyecto en Emergent:
                  </label>
                  <Input
                    type="url"
                    value={projectUrl}
                    onChange={(e) => setProjectUrl(e.target.value)}
                    placeholder="https://tu-proyecto.emergent.sh/"
                    className="bg-gray-900 border-gray-600 text-white"
                    data-testid="project-url-input"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Debe ser una URL válida de un proyecto creado en Emergent
                  </p>
                </div>
                
                <div className="text-center">
                  <Button 
                    onClick={submitProject}
                    disabled={submitting || !projectUrl.trim()}
                    className="btn-futuristic"
                    data-testid="submit-project-btn"
                  >
                    {submitting ? (
                      <div className="flex items-center">
                        <div className="spinner mr-2"></div>
                        Enviando...
                      </div>
                    ) : (
                      <>
                        <i className="fas fa-check mr-2"></i>
                        Enviar Proyecto
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Module Complete */}
        {progress.proyecto_progress === 100 && (
          <Card className="glass-card mt-8 border-green-500/50">
            <CardContent className="p-8 text-center">
              <div className="text-6xl mb-4">
                🎆🎉🚀
              </div>
              <h3 className="text-3xl font-bold mb-4 text-gradient">
                ¡Felicidades, Programador del Futuro!
              </h3>
              <p className="text-xl mb-6 text-gray-300">
                Has completado todos los módulos de ACADEMY. ¡Ya estás listo para obtener tu certificado!
              </p>
              <Link to="/dashboard" className="btn-futuristic text-lg px-8 py-4">
                <i className="fas fa-trophy mr-2"></i>
                Ver Certificado en Dashboard
              </Link>
            </CardContent>
          </Card>
        )}

        {/* Academlo Advertisement */}
        <Card className="glass-card mt-12">
          <CardContent className="p-8 text-center">
            <h3 className="text-2xl font-bold mb-4 text-gradient">
              ¿Quieres ser un programador completo?
            </h3>
            <p className="text-xl mb-6 text-gray-300">
              Estudia en Academlo y complementa tu educación con ACADEMY
            </p>
            <a 
              href="https://www.academlo.com/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-futuristic"
              data-testid="academlo-ad-link"
            >
              <i className="fas fa-external-link-alt mr-2"></i>
              Visitar Academlo
            </a>
          </CardContent>
        </Card>
      </main>

      <ChatWidget />
    </div>
  );
};
