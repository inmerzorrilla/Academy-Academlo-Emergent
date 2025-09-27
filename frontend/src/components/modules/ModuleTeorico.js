import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
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

  const fetchContent = async () => {
    try {
      const response = await axios.get(`${API}/content/teorico`);
      setQuestions(response.data);
    } catch (error) {
      console.error('Error fetching content:', error);
      toast.error('Error al cargar el contenido');
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
      toast.success(`¡Pregunta ${questionId} completada! Progreso: ${response.data.teorico_progress}%`);
    } catch (error) {
      console.error('Error updating progress:', error);
      toast.error('Error al actualizar el progreso');
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
      <header className="relative z-10 p-6 border-b border-gray-800">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <Link to="/dashboard" className="btn-ghost text-sm">
              <i className="fas fa-arrow-left mr-2"></i>
              Volver al Dashboard
            </Link>
          </div>
          
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gradient">Módulo Teórico</h1>
            <p className="text-sm text-cyan-400">Conceptos Fundamentales</p>
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
            
            <Badge className="bg-blue-500">
              {progress.teorico_progress}% Completado
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
              <i className="fas fa-brain mr-2"></i>
              Progreso del Módulo Teórico
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <Progress value={progress.teorico_progress} className="h-3" />
            </div>
            <p className="text-gray-400">
              {progress.teorico_completed.length} de {questions.length} preguntas completadas
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
                    <CardTitle className="text-lg flex items-center text-primary">
                      <span className="text-cyan-400 mr-3">#{question.id}</span>
                      <span className="text-primary">{question.question}</span>
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
                        Respuesta:
                      </h4>
                      <p className="text-gray-300 leading-relaxed">{question.answer}</p>
                    </div>
                    
                    {/* Code Example */}
                    {question.code && (
                      <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                        <h4 className="text-purple-400 font-semibold mb-3 flex items-center">
                          <i className="fas fa-code mr-2"></i>
                          Ejemplo de Código:
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

        {/* Module Complete */}
        {progress.teorico_progress === 100 && (
          <Card className="glass-card mt-8 border-green-500/50">
            <CardContent className="p-8 text-center">
              <div className="text-6xl text-green-400 mb-4">
                🎉
              </div>
              <h3 className="text-2xl font-bold mb-4 text-gradient">
                ¡Módulo Teórico Completado!
              </h3>
              <p className="text-xl mb-6 text-gray-300">
                Has dominado los conceptos fundamentales. ¡Continúa con el siguiente módulo!
              </p>
              <Link to="/dashboard" className="btn-futuristic">
                <i className="fas fa-arrow-right mr-2"></i>
                Continuar al Dashboard
              </Link>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};
