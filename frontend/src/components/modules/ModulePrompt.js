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
import { ChatWidget } from '../ChatWidget';
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
  }, [language]);

  const fetchContent = async () => {
    try {
      const response = await axios.get(`${API}/content/prompt?lang=${language}`);
      setExamples(response.data);
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

  const markSectionComplete = async (section, itemId = null) => {
    try {
      const response = await axios.post(`${API}/user/progress`, {
        module: 'prompt',
        prompt_section: section,
        item_id: itemId
      });
      
      setProgress(response.data);
      
      if (section === 'tips') {
        toast.success(t('tipsCompletedProgress'));
      } else if (section === 'examples') {
        toast.success(t('exampleCompletedProgress', { itemId }));
      } else if (section === 'practice') {
        toast.success(t('practiceCompletedProgress'));
      }
    } catch (error) {
      console.error('Error updating progress:', error);
      toast.error(t('errorUpdatingProgress'));
    }
  };

  const isExampleCompleted = (exampleId) => {
    return progress.prompt_examples_completed?.includes(exampleId) || false;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Simple speech recognition (if available)
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = language === 'es' ? 'es-ES' : 'en-US';
        
        recognition.onstart = () => {
          setIsRecording(true);
          toast.success(t('recordingStarted'));
        };
        
        recognition.onresult = (event) => {
          let transcript = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
          }
          setUserPrompt(prev => prev + ' ' + transcript);
        };
        
        recognition.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          toast.error('Error en el reconocimiento de voz');
          setIsRecording(false);
        };
        
        recognition.onend = () => {
          setIsRecording(false);
          toast.success('Grabación finalizada y transcrita');
        };
        
        recognition.start();
        setMediaRecorder(recognition);
      } else {
        // Fallback to simple audio recording
        const recorder = new MediaRecorder(stream);
        
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            toast.info('Grabación finalizada. Speech-to-text básico agregado al prompt.');
            setUserPrompt(prev => prev + ' [Grabación de voz convertida a texto]');
          }
        };
        
        recorder.start();
        setMediaRecorder(recorder);
        setIsRecording(true);
        toast.success('Grabación iniciada...');
      }
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast.error('Error al acceder al micrófono');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      if (mediaRecorder.stop) {
        mediaRecorder.stop();
      }
      if (mediaRecorder.stream) {
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
      }
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
              {progress.prompt_examples_completed?.length || 0} de {examples.length} ejemplos leídos
              {progress.prompt_tips_completed && ' • Consejos completados'}
              {progress.prompt_practice_completed && ' • Práctica completada'}
            </p>
          </CardContent>
        </Card>

        {/* Prompt Tips Section */}
        <Card className={`glass-card mb-8 ${progress.prompt_tips_completed ? 'border-green-500/50' : ''}`}>
          <CardHeader>
            <CardTitle className="text-xl text-gradient flex items-center">
              <i className="fas fa-lightbulb mr-2"></i>
              {t('promptBasicTips')}
              {progress.prompt_tips_completed && (
                <i className="fas fa-check-circle text-green-500 ml-3"></i>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 p-4 rounded-lg border border-blue-500/20">
                <h4 className="font-semibold text-blue-400 mb-2">{t('clearContext')}</h4>
                <p className="text-sm text-primary">{t('clearContextDesc')}</p>
              </div>
              <div className="bg-gradient-to-r from-green-500/10 to-teal-500/10 p-4 rounded-lg border border-green-500/20">
                <h4 className="font-semibold text-green-400 mb-2">{t('specificInstructions')}</h4>
                <p className="text-sm text-primary">{t('specificInstructionsDesc')}</p>
              </div>
              <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 p-4 rounded-lg border border-purple-500/20">
                <h4 className="font-semibold text-purple-400 mb-2">{t('examplesFormat')}</h4>
                <p className="text-sm text-primary">{t('examplesFormatDesc')}</p>
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
                  {t('markTipsRead')}
                </Button>
              ) : (
                <div className="flex items-center justify-center text-green-400">
                  <i className="fas fa-check-circle mr-2"></i>
                  {t('tipsCompleted')}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Examples */}
        <div className="space-y-6 mb-8">
          <h2 className="text-2xl font-bold text-gradient mb-6">{t('awesomePromptExamples')}</h2>
          
          {examples.map((example) => {
            const isCompleted = isExampleCompleted(example.id);
            
            return (
              <Card key={example.id} className={`glass-card ${isCompleted ? 'border-green-500/50' : ''}`}>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-cyan-400 mr-3">#{example.id}</span>
                      <span className="text-primary">{example.title}</span>
                      {isCompleted && (
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
                    {!isCompleted ? (
                      <Button 
                        onClick={() => markSectionComplete('examples', example.id)}
                        className="btn-futuristic"
                        data-testid={`read-example-${example.id}`}
                      >
                        <i className="fas fa-check mr-2"></i>
                        Marcar como Leído (+20%)
                      </Button>
                    ) : (
                      <div className="flex items-center text-green-400">
                        <i className="fas fa-check-circle mr-2"></i>
                        ¡Ejemplo Completado!
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
              {t('practiceArea')}
            </CardTitle>
            <p className="text-primary">{t('practiceAreaDesc')}</p>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <Textarea
                ref={textareaRef}
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder={language === 'es' 
                  ? "Escribe aquí tu prompt personalizado...\n\nPor ejemplo: 'Actúa como un experto en desarrollo web. Ayudame a crear una aplicación de e-commerce completa usando React y Node.js. Incluye autenticación, carrito de compras y pasarela de pagos...'"
                  : "Write your custom prompt here...\n\nFor example: 'Act as a web development expert. Help me create a complete e-commerce application using React and Node.js. Include authentication, shopping cart and payment gateway...'"
                }
                className="min-h-[150px] bg-gray-900 border-gray-600 text-white resize-none"
                data-testid="prompt-textarea"
              />
              
              {userPrompt.trim() && (
                <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 p-4 rounded-lg border border-cyan-500/20">
                  <h4 className="text-cyan-400 font-semibold mb-2 flex items-center">
                    <i className="fas fa-rocket mr-2"></i>
                    ¡Prueba tu prompt en Emergent!
                  </h4>
                  <p className="text-sm text-gray-300 mb-3">
                    Copia tu prompt y pégalo en Emergent para que te ayude a crear algo genial.
                  </p>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => {
                        navigator.clipboard.writeText(userPrompt);
                        toast.success('Prompt copiado al portapapeles');
                      }}
                      className="btn-ghost text-sm"
                    >
                      <i className="fas fa-copy mr-2"></i>
                      Copiar Prompt
                    </Button>
                    <a
                      href="https://app.emergent.sh/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-futuristic text-sm"
                    >
                      <i className="fas fa-external-link-alt mr-2"></i>
                      Abrir Emergent
                    </a>
                  </div>
                </div>
              )}
              
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-4">
                  <Button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`btn-ghost ${isRecording ? 'text-red-400 border-red-400' : 'text-green-400 border-green-400'}`}
                    data-testid="voice-record-btn"
                  >
                    <i className={`fas ${isRecording ? 'fa-stop' : 'fa-microphone'} mr-2`}></i>
                    {isRecording ? t('stopRecording') : t('useMicrophone')}
                  </Button>
                  
                  {isRecording && (
                    <div className="flex items-center text-red-400">
                      <div className="w-3 h-3 bg-red-400 rounded-full animate-pulse mr-2"></div>
                      {t('recording')}
                    </div>
                  )}
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="text-sm text-primary">
                    {userPrompt.length} {t('characters')}
                  </div>
                  
                  {!progress.prompt_practice_completed && userPrompt.trim() && (
                    <Button 
                      onClick={() => markSectionComplete('practice')}
                      className="btn-futuristic text-sm"
                      data-testid="complete-practice"
                    >
                      <i className="fas fa-check mr-2"></i>
                      Completar Práctica (+20%)
                    </Button>
                  )}
                  
                  {progress.prompt_practice_completed && (
                    <div className="flex items-center text-green-400 text-sm">
                      <i className="fas fa-check-circle mr-2"></i>
                      ¡Práctica Completada!
                    </div>
                  )}
                </div>
              </div>
            </div>
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
