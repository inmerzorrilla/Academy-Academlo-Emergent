import React, { createContext, useContext, useState } from 'react';

const LanguageContext = createContext();

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

const translations = {
  en: {
    // Landing Page
    welcome: 'Welcome to the Future of Programming',
    subtitle: 'Master AI Development with ACADEMY',
    description: 'Join the next generation of developers. Learn Deep Agents, AI programming, and create real projects with cutting-edge technology.',
    getStarted: 'Get Started',
    learnMore: 'Learn More',
    
    // Auth
    login: 'Login',
    register: 'Register',
    name: 'Name',
    email: 'Email',
    phone: 'Phone',
    password: 'Password',
    loginTitle: 'Welcome Back, Future Developer',
    registerTitle: 'Join the Academy',
    
    // Dashboard
    dashboard: 'Dashboard',
    progress: 'Progress',
    modules: 'Modules',
    certificate: 'Certificate',
    welcomeCommander: 'Welcome, AI Commander',
    academloEmergentWelcome: 'Welcome by Academlo - Emergent',
    overallProgress: 'Overall Progress',
    progressDescription: '25% Theoretical - 25% Listen - 25% Prompt - 25% Project',
    downloadCertificate: 'Download Certificate',
    continueProgress: 'Keep going to get your certificate as Future Developer!',
    admin: 'Admin',
    logout: 'Logout',
    
    // Modules
    teorico: 'Theoretical',
    escucha: 'Listen',
    prompt: 'Prompt',
    proyecto: 'Project',
    
    // Module Descriptions
    teoricoDesc: 'Fundamental concepts about Deep Agents and AI programming',
    escuchaDesc: 'Carefully selected educational videos',
    promptDesc: 'Practice with advanced AI prompts',
    proyectoDesc: 'Create real projects with Emergent',
    
    // Module Progress
    backToDashboard: 'Back to Dashboard',
    fundamentalConcepts: 'Fundamental Concepts',
    educationalVideos: 'Educational Videos',
    aiPractice: 'AI Practice',
    finalProject: 'Final Project',
    
    // Common
    complete: 'Complete',
    completed: 'Completed',
    next: 'Next',
    back: 'Back',
    close: 'Close',
    continue: 'Continue',
    
    // Completion Messages
    moduleCompleted: 'Module Completed!',
    continueNextModule: 'You have mastered the fundamentals. Continue with the next module!',
    continueToDashboard: 'Continue to Dashboard',
    
    // Chat
    chatPlaceholder: 'Ask anything about ACADEMY...',
    
    // Footer
    urgentProject: 'Urgent Project?',
    urgentHelp: 'Urgent Help?',
    socialNetworks: 'Social Networks',
    followUs: 'Follow Us!',
    configuration: 'Configuration',
    
    // Academlo Ad
    completeProgrAcademlo: 'Want to be a complete programmer?',
    studyAcademlo: 'Study at Academlo and complement your education with ACADEMY',
    visitAcademlo: 'Visit Academlo'
  },
  es: {
    // Landing Page
    welcome: 'Bienvenido al Futuro de la Programación',
    subtitle: 'Domina el Desarrollo con IA en ACADEMY',
    description: 'Únete a la próxima generación de desarrolladores. Aprende sobre Deep Agents, programación con IA y crea proyectos reales con tecnología de vanguardia.',
    getStarted: 'Comenzar',
    learnMore: 'Saber Más',
    
    // Auth
    login: 'Iniciar Sesión',
    register: 'Registrarse',
    name: 'Nombre',
    email: 'Correo',
    phone: 'Teléfono',
    password: 'Contraseña',
    loginTitle: 'Bienvenido de Vuelta, Desarrollador del Futuro',
    registerTitle: 'Únete a la Academia',
    
    // Dashboard
    dashboard: 'Panel',
    progress: 'Progreso',
    modules: 'Módulos',
    certificate: 'Certificado',
    welcomeCommander: 'Bienvenido, Comandante de la IA',
    academloEmergentWelcome: 'Bienvenido por Academlo - Emergent',
    overallProgress: 'Progreso General',
    progressDescription: '25% Teórico - 25% Escucha - 25% Prompt - 25% Proyecto',
    downloadCertificate: 'Descargar Certificado',
    continueProgress: '¡Sigue avanzando para obtener tu certificado como Programador del Futuro!',
    admin: 'Admin',
    logout: 'Salir',
    
    // Modules
    teorico: 'Teórico',
    escucha: 'Escucha',
    prompt: 'Prompt',
    proyecto: 'Proyecto',
    
    // Module Descriptions
    teoricoDesc: 'Conceptos fundamentales sobre Deep Agents y programación con IA',
    escuchaDesc: 'Videos educativos cuidadosamente seleccionados',
    promptDesc: 'Practica con prompts avanzados de IA',
    proyectoDesc: 'Crea proyectos reales con Emergent',
    
    // Module Progress
    backToDashboard: 'Volver al Dashboard',
    fundamentalConcepts: 'Conceptos Fundamentales',
    educationalVideos: 'Videos Educativos',
    aiPractice: 'Práctica con IA',
    finalProject: 'Proyecto Final',
    
    // Common
    complete: 'Completar',
    completed: 'Completado',
    next: 'Siguiente',
    back: 'Regresar',
    close: 'Cerrar',
    continue: 'Continuar',
    
    // Completion Messages
    moduleCompleted: 'Módulo Completado!',
    continueNextModule: 'Has dominado los conceptos fundamentales. ¡Continúa con el siguiente módulo!',
    continueToDashboard: 'Continuar al Dashboard',
    
    // Chat
    chatPlaceholder: 'Pregunta cualquier cosa sobre ACADEMY...',
    
    // Footer
    urgentProject: '¿Proyecto urgente?',
    urgentHelp: '¿Ayuda urgente?',
    socialNetworks: 'Redes Sociales',
    followUs: '¡Síguenos!',
    configuration: 'Configuración',
    
    // Academlo Ad
    completeProgrAcademlo: '¿Quieres ser un programador completo?',
    studyAcademlo: 'Estudia en Academlo y complementa tu educación con ACADEMY',
    visitAcademlo: 'Visitar Academlo'
  }
};

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState(localStorage.getItem('academy_language') || 'es');

  const toggleLanguage = () => {
    const newLanguage = language === 'es' ? 'en' : 'es';
    setLanguage(newLanguage);
    localStorage.setItem('academy_language', newLanguage);
  };

  const t = (key) => {
    return translations[language][key] || key;
  };

  const value = {
    language,
    setLanguage,
    toggleLanguage,
    t
  };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
};
