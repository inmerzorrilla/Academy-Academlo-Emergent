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
import { ModuleHeader } from '../ModuleHeader';
import { Footer } from '../Footer';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const ModuleTeorico = () => {
  const { user } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const [questions, setQuestions] = useState([]);
  const [progress, setProgress] = useState({ teorico_completed: [], teorico_progress: 0 });
  const [loading, setLoading] = useState(true);
  const [expandedQuestion, setExpandedQuestion] = useState(null);

  useEffect(() => {
    fetchContent();
    fetchProgress();
  }, []);
  
  useEffect(() => {
    fetchContent(); // Re-fetch content when language changes
  }, [language]);

  const fetchContent = async () => {
    try {
      const response = await axios.get(`${API}/content/teorico?lang=${language}`);
      setQuestions(response.data);
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

  const markQuestionComplete = async (questionId) => {
    try {
      const response = await axios.post(`${API}/user/progress`, {
        module: 'teorico',
        item_id: questionId
      });
      
      setProgress(response.data);
      toast.success(t('questionCompletedProgress', { questionId, progress: response.data.teorico_progress }));
    } catch (error) {
      console.error('Error updating progress:', error);
      toast.error(t('errorUpdatingProgress'));
    }
  };

  const isQuestionCompleted = (questionId) => {
    return progress.teorico_completed.includes(questionId);
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
      <ModuleHeader 
        moduleName={t('theoreticalModule')}
        moduleSubtitle={t('fundamentalConcepts')}
        progress={progress.teorico_progress}
      />

      {/* Content */}
      <main className="relative z-10 max-w-4xl mx-auto px-6 py-8">
        {/* Progress Section */}
        <Card className="glass-card mb-8">
          <CardHeader>
            <CardTitle className="text-xl text-gradient flex items-center">
              <i className="fas fa-brain mr-2"></i>
              {t('theoreticalModuleProgress')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <Progress value={progress.teorico_progress} className="h-3" />
            </div>
            <p className="text-gray-400">
              {progress.teorico_completed.length} {language === 'es' ? 'de' : 'of'} {questions.length} {t('questionsCompleted')}
            </p>
          </CardContent>
        </Card>

        {/* Questions */}
        <div className="space-y-6">
          {questions.map((question) => {
            const isCompleted = isQuestionCompleted(question.id);
            const isExpanded = expandedQuestion === question.id;
            
            return (
              <Card key={question.id} className={`glass-card ${isCompleted ? 'border-green-500/50' : ''}`}>
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-lg flex items-center">
                      <span className="text-cyan-400 mr-3">#{question.id}</span>
                      <span className="text-primary font-semibold">{question.question}</span>
                      {isCompleted && (
                        <i className="fas fa-check-circle text-green-500 ml-3"></i>
                      )}
                    </CardTitle>
                    <Button
                      onClick={() => setExpandedQuestion(isExpanded ? null : question.id)}
                      className="btn-ghost text-sm"
                      data-testid={`toggle-question-${question.id}`}
                    >
                      <i className={`fas fa-chevron-${isExpanded ? 'up' : 'down'}`}></i>
                    </Button>
                  </div>
                </CardHeader>
                
                {isExpanded && (
                  <CardContent className="space-y-6">
                    {/* Answer */}
                    <div className="bg-gray-800/50 p-4 rounded-lg">
                      <h4 className="text-green-400 font-semibold mb-3 flex items-center">
                        <i className="fas fa-lightbulb mr-2"></i>
                        {t('answer')}:
                      </h4>
                      <p className="text-primary leading-relaxed">{question.answer}</p>
                    </div>
                    
                    {/* Code Example */}
                    {question.code && (
                      <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                        <h4 className="text-purple-400 font-semibold mb-3 flex items-center">
                          <i className="fas fa-code mr-2"></i>
                          {t('codeExample')}:
                        </h4>
                        <pre className="text-green-300 text-sm overflow-x-auto">
                          <code>{question.code}</code>
                        </pre>
                      </div>
                    )}
                    
                    {/* Complete Button */}
                    <div className="flex justify-center">
                      {!isCompleted ? (
                        <Button 
                          onClick={() => markQuestionComplete(question.id)}
                          className="btn-futuristic"
                          data-testid={`complete-question-${question.id}`}
                        >
                          <i className="fas fa-check mr-2"></i>
                          Marcar como Completada
                        </Button>
                      ) : (
                        <div className="flex items-center text-green-400">
                          <i className="fas fa-check-circle mr-2"></i>
                          ¡Pregunta Completada!
                        </div>
                      )}
                    </div>
                  </CardContent>
                )}
              </Card>
            );
          })}
        </div>

        {/* Module Complete */}
        {progress.teorico_progress === 100 && (
          <Card className="glass-card mt-8 border-green-500/50">
            <CardContent className="p-8 text-center">
              <div className="text-6xl text-green-400 mb-4">
                🎉
              </div>
              <h3 className="text-2xl font-bold mb-4 text-gradient">
                {t('theoreticalCompleted')}
              </h3>
              <p className="text-xl mb-6 text-gray-300">
                {t('continueNextModule')}
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

      <Footer />
      <ChatWidget />
    </div>
  );
};
