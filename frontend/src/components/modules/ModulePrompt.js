import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Textarea } from '../ui/textarea';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const ModulePrompt = () => {
  const { user } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const [examples, setExamples] = useState([]);
  const [progress, setProgress] = useState({ 
    prompt_completed: false, 
    prompt_progress: 0,
    prompt_tips_completed: false,
    prompt_examples_completed: [],
    prompt_practice_completed: false
  });
  const [loading, setLoading] = useState(true);
  const [userPrompt, setUserPrompt] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    fetchContent();
    fetchProgress();
  }, []);

  const fetchContent = async () => {
    try {
      const response = await axios.get(`${API}/content/prompt`);
      setExamples(response.data);
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

  const markExampleRead = (exampleId) => {
    if (!readExamples.includes(exampleId)) {
      setReadExamples([...readExamples, exampleId]);
      toast.success(`Ejemplo ${exampleId} leído`);
    }
  };

  const completeModule = async () => {
    if (readExamples.length === examples.length && userPrompt.trim()) {
      try {
        const response = await axios.post(`${API}/user/progress`, {
          module: 'prompt'
        });
        
        setProgress(response.data);
        toast.success('¡Módulo Prompt completado al 100%! 🎉');
      } catch (error) {
        console.error('Error updating progress:', error);
        toast.error('Error al actualizar el progreso');
      }
    } else {
      toast.error('Lee todos los ejemplos y practica escribiendo un prompt para completar el módulo');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          // In a real app, you would send this to a speech-to-text service
          toast.info('Grabación finalizada. En una versión completa, esto se convertiría a texto.');
        }
      };
      
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      toast.success('Grabación iniciada. Habla tu prompt...');
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast.error('Error al acceder al micrófono');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      setMediaRecorder(null);
      setIsRecording(false);
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
            <h1 className="text-2xl font-bold text-gradient">Módulo Prompt</h1>
            <p className="text-sm text-cyan-400">Practica con IA</p>
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
            
            <Badge className="bg-purple-500">
              {progress.prompt_progress}% Completado
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
              <i className="fas fa-terminal mr-2"></i>
              Progreso del Módulo Prompt
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <Progress value={progress.prompt_progress} className="h-3" />
            </div>
            <p className="text-gray-400">
              {readExamples.length} de {examples.length} ejemplos leídos
              {userPrompt.trim() && ' • Prompt de práctica escrito'}
            </p>
          </CardContent>
        </Card>

        {/* Prompt Tips Section */}
        <Card className={`glass-card mb-8 ${progress.prompt_tips_completed ? 'border-green-500/50' : ''}`}>
          <CardHeader>
            <CardTitle className="text-xl text-gradient flex items-center">
              <i className="fas fa-lightbulb mr-2"></i>
              3 Elementos Básicos de un Prompt Efectivo
              {progress.prompt_tips_completed && (
                <i className="fas fa-check-circle text-green-500 ml-3"></i>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 p-4 rounded-lg border border-blue-500/20">
                <h4 className="font-semibold text-blue-400 mb-2">1. Contexto Claro</h4>
                <p className="text-sm text-gray-300">Define el rol y el contexto específico para obtener respuestas más precisas.</p>
              </div>
              <div className="bg-gradient-to-r from-green-500/10 to-teal-500/10 p-4 rounded-lg border border-green-500/20">
                <h4 className="font-semibold text-green-400 mb-2">2. Instrucciones Específicas</h4>
                <p className="text-sm text-gray-300">Sé específico sobre qué quieres que haga y cómo quieres el resultado.</p>
              </div>
              <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 p-4 rounded-lg border border-purple-500/20">
                <h4 className="font-semibold text-purple-400 mb-2">3. Ejemplos o Formato</h4>
                <p className="text-sm text-gray-300">Proporciona ejemplos del resultado esperado o especifica el formato deseado.</p>
              </div>
            </div>
            
            <div className="text-center mt-6">
              {!progress.prompt_tips_completed ? (
                <Button 
                  onClick={() => markSectionComplete('tips')}
                  className="btn-futuristic"
                  data-testid="complete-tips"
                >
                  <i className="fas fa-check mr-2"></i>
                  Marcar Consejos como Leídos
                </Button>
              ) : (
                <div className="flex items-center justify-center text-green-400">
                  <i className="fas fa-check-circle mr-2"></i>
                  ¡Consejos Completados!
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Examples */}
        <div className="space-y-6 mb-8">
          <h2 className="text-2xl font-bold text-gradient mb-6">Ejemplos de Prompts Geniales</h2>
          
          {examples.map((example) => {
            const isRead = readExamples.includes(example.id);
            
            return (
              <Card key={example.id} className={`glass-card ${isRead ? 'border-green-500/50' : ''}`}>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-cyan-400 mr-3">#{example.id}</span>
                      {example.title}
                      {isRead && (
                        <i className="fas fa-check-circle text-green-500 ml-3"></i>
                      )}
                    </div>
                  </CardTitle>
                  <p className="text-gray-400">{example.description}</p>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  {/* Prompt Example */}
                  <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                    <h4 className="text-purple-400 font-semibold mb-3 flex items-center">
                      <i className="fas fa-magic mr-2"></i>
                      Prompt de Ejemplo:
                    </h4>
                    <p className="text-green-300 leading-relaxed italic">
                      "{example.prompt}"
                    </p>
                  </div>
                  
                  {/* Read Button */}
                  <div className="flex justify-center">
                    {!isRead ? (
                      <Button 
                        onClick={() => markExampleRead(example.id)}
                        className="btn-futuristic"
                        data-testid={`read-example-${example.id}`}
                      >
                        <i className="fas fa-check mr-2"></i>
                        Marcar como Leído
                      </Button>
                    ) : (
                      <div className="flex items-center text-green-400">
                        <i className="fas fa-check-circle mr-2"></i>
                        ¡Ejemplo Leído!
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Practice Section */}
        <Card className="glass-card mb-8">
          <CardHeader>
            <CardTitle className="text-xl text-gradient flex items-center">
              <i className="fas fa-edit mr-2"></i>
              Área de Práctica
            </CardTitle>
            <p className="text-gray-400">Escribe tu propio prompt o usa el micrófono para dictarlo</p>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <Textarea
                ref={textareaRef}
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder="Escribe aquí tu prompt personalizado...\n\nPor ejemplo: 'Actúa como un experto en desarrollo web. Ayudame a crear una aplicación de e-commerce completa usando React y Node.js. Incluye autenticación, carrito de compras y pasarela de pagos...'"
                className="min-h-[150px] bg-gray-900 border-gray-600 text-white resize-none"
                data-testid="prompt-textarea"
              />
              
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-4">
                  <Button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`btn-ghost ${isRecording ? 'text-red-400 border-red-400' : 'text-green-400 border-green-400'}`}
                    data-testid="voice-record-btn"
                  >
                    <i className={`fas ${isRecording ? 'fa-stop' : 'fa-microphone'} mr-2`}></i>
                    {isRecording ? 'Detener Grabación' : 'Usar Micrófono'}
                  </Button>
                  
                  {isRecording && (
                    <div className="flex items-center text-red-400">
                      <div className="w-3 h-3 bg-red-400 rounded-full animate-pulse mr-2"></div>
                      Grabando...
                    </div>
                  )}
                </div>
                
                <div className="text-sm text-gray-400">
                  {userPrompt.length} caracteres
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Complete Module */}
        {!progress.prompt_completed && (
          <Card className="glass-card mb-8">
            <CardContent className="p-8 text-center">
              <h3 className="text-2xl font-bold mb-4 text-gradient">
                ¿Listo para completar el módulo?
              </h3>
              <p className="text-gray-300 mb-6">
                Asegúrate de haber leído todos los ejemplos y practicado escribiendo tu propio prompt
              </p>
              <Button 
                onClick={completeModule}
                className="btn-futuristic"
                disabled={readExamples.length !== examples.length || !userPrompt.trim()}
                data-testid="complete-module-btn"
              >
                <i className="fas fa-check mr-2"></i>
                Completar Módulo
              </Button>
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

        {/* Module Complete */}
        {progress.prompt_completed && (
          <Card className="glass-card mt-8 border-green-500/50">
            <CardContent className="p-8 text-center">
              <div className="text-6xl text-green-400 mb-4">
                🎉
              </div>
              <h3 className="text-2xl font-bold mb-4 text-gradient">
                ¡Módulo Prompt Completado!
              </h3>
              <p className="text-xl mb-6 text-gray-300">
                Has dominado el arte de crear prompts efectivos. ¡Continúa con el último módulo!
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
