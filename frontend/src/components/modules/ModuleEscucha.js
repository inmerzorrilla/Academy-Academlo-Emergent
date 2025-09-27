import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { ChatWidget } from '../ChatWidget';
import { ModuleFooter } from '../ModuleFooter';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const ModuleEscucha = () => {
  const { user } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const [videos, setVideos] = useState([]);
  const [progress, setProgress] = useState({ escucha_completed: [], escucha_progress: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchContent();
    fetchProgress();
  }, [language]);

  const fetchContent = async () => {
    try {
      const response = await axios.get(`${API}/content/escucha?lang=${language}`);
      setVideos(response.data);
    } catch (error) {
      console.error('Error fetching content:', error);
      toast.error(t('errorLoadingContent'));
    }
  };

  const fetchProgress = async () => {
    try {
      const response = await axios.get(`${API}/user/progress`);
      setProgress(response.data);
    } catch (error) {
      console.error('Error fetching progress:', error);
    } finally {
      setLoading(false);
    }
  };

  const markVideoComplete = async (videoId) => {
    try {
      const response = await axios.post(`${API}/user/progress`, {
        module: 'escucha',
        item_id: videoId
      });
      
      setProgress(response.data);
      toast.success(t('videoCompletedProgress', { videoId, progress: response.data.escucha_progress }));
    } catch (error) {
      console.error('Error updating progress:', error);
      toast.error(t('errorUpdatingProgress'));
    }
  };

  const isVideoCompleted = (videoId) => {
    return progress.escucha_completed.includes(videoId);
  };

  const getYouTubeEmbedUrl = (url) => {
    // Convert YouTube URLs to embed format
    if (url.includes('youtube.com/watch?v=')) {
      const videoId = url.split('v=')[1].split('&')[0];
      const timeParam = url.includes('&t=') ? `?start=${url.split('&t=')[1].replace('s', '')}` : '';
      return `https://www.youtube.com/embed/${videoId}${timeParam}`;
    } else if (url.includes('youtube.com/shorts/')) {
      const videoId = url.split('shorts/')[1].split('?')[0];
      return `https://www.youtube.com/embed/${videoId}`;
    }
    return url;
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
              {t('backToDashboard')}
            </Link>
          </div>
          
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gradient">{t('listeningModule')}</h1>
            <p className="text-sm text-cyan-400">{t('educationalVideos')}</p>
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
            
            <Badge className="bg-green-500">
              {progress.escucha_progress}% {t('completed')}
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
              <i className="fas fa-headphones mr-2"></i>
              {t('listeningModuleProgress')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <Progress value={progress.escucha_progress} className="h-3" />
            </div>
            <p className="text-gray-400">
              {progress.escucha_completed.length} {t('of')} {videos.length} {t('videosCompleted')}
            </p>
          </CardContent>
        </Card>

        {/* Videos */}
        <div className="space-y-8">
          {videos.map((video) => {
            const isCompleted = isVideoCompleted(video.id);
            
            return (
              <Card key={video.id} className={`glass-card ${isCompleted ? 'border-green-500/50' : ''}`}>
                <CardHeader>
                  <CardTitle className="text-xl flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-cyan-400 mr-3">#{video.id}</span>
                      <span className="text-primary font-semibold">{video.title}</span>
                      {isCompleted && (
                        <i className="fas fa-check-circle text-green-500 ml-3"></i>
                      )}
                    </div>
                  </CardTitle>
                  <p className="text-primary">{video.description}</p>
                </CardHeader>
                
                <CardContent className="space-y-6">
                  {/* Video Embed */}
                  <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden">
                    <iframe
                      src={getYouTubeEmbedUrl(video.url)}
                      title={video.title}
                      className="w-full h-full"
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      data-testid={`video-${video.id}`}
                    ></iframe>
                  </div>
                  
                  {/* Video Actions */}
                  <div className="flex justify-between items-center">
                    <a 
                      href={video.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-ghost text-sm"
                      data-testid={`open-video-${video.id}`}
                    >
                      <i className="fab fa-youtube mr-2"></i>
                      {t('watchOnYoutube')}
                    </a>
                    
                    {!isCompleted ? (
                      <Button 
                        onClick={() => markVideoComplete(video.id)}
                        className="btn-futuristic"
                        data-testid={`complete-video-${video.id}`}
                      >
                        <i className="fas fa-check mr-2"></i>
                        {t('markAsWatched')}
                      </Button>
                    ) : (
                      <div className="flex items-center text-green-400">
                        <i className="fas fa-check-circle mr-2"></i>
                        {t('videoCompleted')}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Module Complete */}
        {progress.escucha_progress === 100 && (
          <Card className="glass-card mt-8 border-green-500/50">
            <CardContent className="p-8 text-center">
              <div className="text-6xl text-green-400 mb-4">
                🎉
              </div>
              <h3 className="text-2xl font-bold mb-4 text-gradient">
                {t('listeningModuleCompleted')}
              </h3>
              <p className="text-xl mb-6 text-gray-300">
                {t('allVideosWatched')}
              </p>
              <Link to="/dashboard" className="btn-futuristic">
                <i className="fas fa-arrow-right mr-2"></i>
                {t('continueToDashboard')}
              </Link>
            </CardContent>
          </Card>
        )}

        {/* Academlo Advertisement */}
        <Card className="glass-card mt-12">
          <CardContent className="p-8 text-center">
            <h3 className="text-2xl font-bold mb-4 text-gradient">
              {t('wantToBeCompleteProgrammer')}
            </h3>
            <p className="text-xl mb-6 text-gray-300">
              {t('studyAtAcademlo')}
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

      <ChatWidget />
      <ModuleFooter />
    </div>
  );
};
