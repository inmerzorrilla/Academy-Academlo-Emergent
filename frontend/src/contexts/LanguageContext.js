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
    visitAcademlo: 'Visit Academlo',
    
    // Prompt Module
    promptBasicTips: '3 Basic Elements of an Effective Prompt',
    clearContext: '1. Clear Context',
    clearContextDesc: 'Define the role and specific context to get more precise responses.',
    specificInstructions: '2. Specific Instructions', 
    specificInstructionsDesc: 'Be specific about what you want it to do and how you want the result.',
    examplesFormat: '3. Examples or Format',
    examplesFormatDesc: 'Provide examples of expected result or specify desired format.',
    markTipsRead: 'Mark Tips as Read',
    tipsCompleted: 'Tips Completed!',
    awesomePromptExamples: 'Awesome Prompt Examples',
    practiceArea: 'Practice Area',
    practiceAreaDesc: 'Write your own prompt or use the microphone to dictate it',
    writePromptHere: 'Write your custom prompt here...',
    useMicrophone: 'Use Microphone',
    stopRecording: 'Stop Recording',
    recording: 'Recording...',
    characters: 'characters',
    testPromptEmergent: 'Test your prompt in Emergent!',
    copyPromptDesc: 'Copy your prompt and paste it in Emergent to help you create something awesome.',
    copyPrompt: 'Copy Prompt',
    openEmergent: 'Open Emergent',
    completePractice: 'Complete Practice (+20%)',
    practiceCompleted: 'Practice Completed!',
    markAsRead: 'Mark as Read (+20%)',
    exampleCompleted: 'Example Completed!',
    
    // Prompt Examples (English)
    codeGenerationPrompt: 'Prompt for Code Generation',
    codeGenerationPromptText: 'Act as an expert Python developer. Create a function that receives a list of numbers and returns the average, median and mode. Include error handling and complete documentation.',
    codeGenerationDesc: 'This prompt is ideal for generating functional and well-documented code',
    
    dataAnalysisPrompt: 'Prompt for Data Analysis',
    dataAnalysisPromptText: 'You are an expert data scientist. Analyze the following sales dataset and provide key insights, important trends and strategic recommendations. Include suggested visualizations and relevant metrics.',
    dataAnalysisDesc: 'Perfect for getting deep business data analysis',
    
    softwareArchPrompt: 'Prompt for Software Architecture',
    softwareArchPromptText: 'As a senior software architect, design a scalable architecture for an e-commerce application that handles 100k concurrent users. Include design patterns, recommended technologies, component diagram and security considerations.',
    softwareArchDesc: 'Ideal for getting robust and professional architectural designs',
    
    webDevPrompt: 'Prompt for Complete Web Development',
    webDevPromptText: 'Act as a full-stack web development expert. Create a complete web application using React and Node.js that includes: JWT authentication system, responsive dashboard, user CRUD, MongoDB database integration, documented REST API, and cloud deployment. Provide complete code, file structure and installation guide.',
    webDevDesc: 'Perfect for generating complete and modern web applications',
    
    // Chat responses
    chatResponse1: 'ACADEMY is a futuristic educational platform designed to train future programmers. We offer courses on Deep Agents, AI, programming and project development with emerging technology.',
    chatWhatsapp: 'For any questions, here is our direct WhatsApp number: +528136037100',
    
    // Module Headers
    theoreticalModule: 'Theoretical Module',
    listenModule: 'Listen Module', 
    promptModule: 'Prompt Module',
    projectModule: 'Project Module',
    
    // Module Progress Labels
    questionsCompleted: 'questions completed',
    videosCompleted: 'videos completed',
    practiceCompleted: 'practice completed',
    
    // Module Completion
    theoreticalCompleted: 'Theoretical Module Completed!',
    listenCompleted: 'Listen Module Completed!',
    promptCompleted: 'Prompt Module Completed!',
    projectCompleted: 'Project Module Completed!',
    
    // Videos
    videoTitle1: 'Introduction to AI',
    videoTitle2: 'AI Development',
    videoTitle3: 'The Future of Programming',
    videoTitle4: 'Why is it important to study at Academlo?',
    videoTitle5: 'Emergent Tutorial',
    
    videoDesc1: 'Short video about the basic concepts of artificial intelligence',
    videoDesc2: 'Learn how AI is transforming software development',
    videoDesc3: 'Discover where programming is heading in the AI era',
    videoDesc4: 'Learn about the advantages of studying at Academlo and how it can transform your career',
    videoDesc5: 'Learn to use Emergent to create incredible projects with AI',
    
    // Buttons and Actions
    markAsWatched: 'Mark as Watched',
    videoCompleted: 'Video Completed!',
    watchOnYoutube: 'Watch on YouTube',
    
    // Module Specific
    listeningModule: 'Listen Module',
    listeningModuleProgress: 'Listen Module Progress',
    listeningModuleCompleted: 'Listen Module Completed!',
    allVideosWatched: 'You have watched all educational videos. Continue with the next module!',
    of: 'of',
    
    // Error Messages
    errorLoadingContent: 'Error loading content',
    errorUpdatingProgress: 'Error updating progress',
    videoCompletedProgress: 'Video {{videoId}} completed! Progress: {{progress}}%',
    
    // Academlo Ad
    wantToBeCompleteProgrammer: 'Want to be a complete programmer?',
    studyAtAcademlo: 'Study at Academlo and complement your education with ACADEMY',
    
    // Landing Page
    learningModules: 'Learning Modules',
    poweredBy: 'Powered by Emergent + Academlo',
    
    // Footer
    footerDescription: 'Futuristic educational platform to train future programmers with AI and Deep Agents.',
    quickActions: 'Quick Actions',
    ourPartners: 'Our Partners',
    joinDeveloperCommunity: 'Join our developer community',
    madeWith: 'Made with',
    secure: 'Secure',
    responsive: 'Responsive',
    
    // Dashboard Error Messages
    errorLoadingProgress: 'Error loading progress',
    certificateDownloadedSuccess: 'Certificate downloaded successfully! 🎉',
    errorDownloadingCertificate: 'Error downloading certificate',
    completeAllModulesForCertificate: 'Complete all modules to get your certificate',
    
    // Auth Messages
    errorOccurred: 'Error occurred. Please try again.',
    
    // Module Messages
    pleaseEnterProjectUrl: 'Please enter your project URL',
    urlMustBeFromEmergent: 'URL must be from a project created on Emergent',
    projectSubmittedSuccessfully: 'Project submitted successfully! Module completed 100% 🎉',
    errorSubmittingProject: 'Error submitting project',
    
    // Prompt Module Messages
    tipsCompletedProgress: 'Tips completed! +20%',
    exampleCompletedProgress: 'Example {{itemId}} completed! +20%',
    practiceCompletedProgress: 'Practice completed! +20%',
    recordingStarted: 'Recording started. Speak your prompt...',
    speechRecognitionError: 'Speech recognition error',
    recordingFinishedTranscribed: 'Recording finished and transcribed',
    recordingFinishedBasicSpeechToText: 'Recording finished. Basic speech-to-text added to prompt.',
    recordingStartedBasic: 'Recording started...',
    microphoneAccessError: 'Error accessing microphone',
    promptCopiedToClipboard: 'Prompt copied to clipboard',
    
    // Teorico Module Messages
    questionCompletedProgress: 'Question {{questionId}} completed! Progress: {{progress}}%',
    
    // Admin Messages
    errorLoadingUsers: 'Error loading users',
    userDeletedSuccessfully: 'User deleted successfully',
    errorDeletingUser: 'Error deleting user',
    
    // Project Module Messages
    createProjectEmergent: 'Create your Project with Emergent',
    timeToCreateSomethingIncredible: 'It\'s time to create something incredible!',
    yourMission: 'Your mission',
    useEmergentToCreateProject: 'Use the platform',
    toDemonstrateWhatYouLearned: 'to create a project that demonstrates everything you have learned at ACADEMY',
    projectIdeas: 'Project ideas',
    projectIdea1: 'An intelligent chatbot with AI',
    projectIdea2: 'A web application with advanced functionalities',
    projectIdea3: 'An intelligent automation system',
    projectIdea4: 'A tool that solves a real problem',
    requirements: 'Requirements',
    requirement1: 'Must be created on the Emergent platform',
    requirement2: 'Must apply concepts learned in the course',
    requirement3: 'Must be functional and demonstrable',
    goToEmergentToCreateProject: 'Go to Emergent to create your project',
    onceProjectCreatedComeBack: 'Once you have created your project, come back here to submit the URL',
    submitYourProject: 'Submit your Project',
    projectSubmittedSuccessfullyTitle: 'Project Submitted Successfully!',
    projectHasBeenRegistered: 'Your project has been registered correctly',
    projectUrlLabel: 'URL of your project on Emergent',
    projectUrlPlaceholder: 'https://your-project.emergent.sh/',
    validEmergentUrlRequired: 'Must be a valid URL of a project created on Emergent',
    submitting: 'Submitting',
    submitProject: 'Submit Project',
    congratsFutureProgrammer: 'Congratulations, Future Programmer!',
    completedAllModulesReadyForCertificate: 'You have completed all ACADEMY modules. You are ready to get your certificate!',
    viewCertificateInDashboard: 'View Certificate in Dashboard',
    
    // Project Module
    createYourProject: 'Create your Project with Emergent',
    yourMission: 'Your mission',
    projectIdeas: 'Project ideas:',
    requirements: 'Requirements:',
    submitYourProject: 'Submit your Project',
    projectSubmitted: 'Project Submitted Successfully!',
    projectUrl: 'URL of your project in Emergent:',
    submitProject: 'Submit Project',
    
    // Instructions
    useEmergentPlatform: 'Use the Emergent platform to create a project that demonstrates everything you have learned at ACADEMY.',
    intelligentChatbot: 'An intelligent chatbot with AI',
    webAppAdvanced: 'A web application with advanced functionalities', 
    intelligentAutomation: 'An intelligent automation system',
    toolSolvesProblems: 'A tool that solves a real problem',
    createdInEmergent: 'Must be created on the Emergent platform',
    applyLearned: 'Must apply concepts learned in the course',
    functionalDemo: 'Must be functional and demonstrable',
    goToEmergent: 'Go to Emergent to create your project',
    onceCreated: 'Once you have created your project, return here to submit the URL',
    
    // Final Congratulations
    congratsFutureProgrammer: 'Congratulations, Future Programmer!',
    completedAllModules: 'You have completed all ACADEMY modules. You are ready to get your certificate!',
    viewCertificateInDashboard: 'View Certificate in Dashboard'
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
    visitAcademlo: 'Visitar Academlo',
    
    // Prompt Module
    promptBasicTips: '3 Elementos Básicos de un Prompt Efectivo',
    clearContext: '1. Contexto Claro',
    clearContextDesc: 'Define el rol y el contexto específico para obtener respuestas más precisas.',
    specificInstructions: '2. Instrucciones Específicas',
    specificInstructionsDesc: 'Sé específico sobre qué quieres que haga y cómo quieres el resultado.',
    examplesFormat: '3. Ejemplos o Formato',
    examplesFormatDesc: 'Proporciona ejemplos del resultado esperado o especifica el formato deseado.',
    markTipsRead: 'Marcar Consejos como Leídos',
    tipsCompleted: '¡Consejos Completados!',
    awesomePromptExamples: 'Ejemplos de Prompts Geniales',
    practiceArea: 'Área de Práctica',
    practiceAreaDesc: 'Escribe tu propio prompt o usa el micrófono para dictarlo',
    writePromptHere: 'Escribe aquí tu prompt personalizado...',
    useMicrophone: 'Usar Micrófono',
    stopRecording: 'Detener Grabación',
    recording: 'Grabando...',
    characters: 'caracteres',
    testPromptEmergent: '¡Prueba tu prompt en Emergent!',
    copyPromptDesc: 'Copia tu prompt y pégalo en Emergent para que te ayude a crear algo genial.',
    copyPrompt: 'Copiar Prompt',
    openEmergent: 'Abrir Emergent',
    completePractice: 'Completar Práctica (+20%)',
    practiceCompleted: '¡Práctica Completada!',
    markAsRead: 'Marcar como Leído (+20%)',
    exampleCompleted: '¡Ejemplo Completado!',
    
    // Prompt Examples (Spanish)
    codeGenerationPrompt: 'Prompt para Generación de Código',
    codeGenerationPromptText: 'Actúa como un experto desarrollador Python. Crea una función que reciba una lista de números y devuelva el promedio, la mediana y la moda. Incluye manejo de errores y documentación completa.',
    codeGenerationDesc: 'Este prompt es ideal para generar código funcional y bien documentado',
    
    dataAnalysisPrompt: 'Prompt para Análisis de Datos',
    dataAnalysisPromptText: 'Eres un científico de datos experto. Analiza el siguiente dataset de ventas y proporciona insights clave, tendencias importantes y recomendaciones estratégicas. Incluye visualizaciones sugeridas y métricas relevantes.',
    dataAnalysisDesc: 'Perfecto para obtener análisis profundos de datos empresariales',
    
    softwareArchPrompt: 'Prompt para Arquitectura de Software',
    softwareArchPromptText: 'Como arquitecto de software senior, diseña una arquitectura escalable para una aplicación de e-commerce que maneje 100k usuarios concurrentes. Incluye patrones de diseño, tecnologías recomendadas, diagrama de componentes y consideraciones de seguridad.',
    softwareArchDesc: 'Ideal para obtener diseños arquitectónicos robustos y profesionales',
    
    webDevPrompt: 'Prompt para Desarrollo Web Completo',
    webDevPromptText: 'Actúa como un experto en desarrollo web full-stack. Crea una aplicación web completa usando React y Node.js que incluya: sistema de autenticación JWT, dashboard responsivo, CRUD de usuarios, integración con base de datos MongoDB, API REST documentada, y deployment en la nube. Proporciona el código completo, estructura de archivos y guía de instalación.',
    webDevDesc: 'Perfecto para generar aplicaciones web completas y modernas',
    
    // Chat responses
    chatResponse1: 'ACADEMY es una plataforma educativa futurista diseñada para formar a los programadores del futuro. Ofrecemos cursos sobre Deep Agents, IA, programación y desarrollo de proyectos con tecnología emergente.',
    chatWhatsapp: 'Para cualquier duda, te dejo nuestro número de WhatsApp directo: +528136037100',
    
    // Module Headers
    theoreticalModule: 'Módulo Teórico',
    listenModule: 'Módulo Escucha',
    promptModule: 'Módulo Prompt', 
    projectModule: 'Módulo Proyecto',
    
    // Module Progress Labels
    questionsCompleted: 'preguntas completadas',
    videosCompleted: 'videos completados',
    practiceCompleted: 'práctica completada',
    
    // Module Completion
    theoreticalCompleted: '¡Módulo Teórico Completado!',
    listenCompleted: '¡Módulo Escucha Completado!',
    promptCompleted: '¡Módulo Prompt Completado!',
    projectCompleted: '¡Módulo Proyecto Completado!',
    
    // Videos
    videoTitle1: 'Introducción a la IA',
    videoTitle2: 'Desarrollo con IA',
    videoTitle3: 'El futuro de la programación',
    videoTitle4: '¿Por qué es importante estudiar en Academlo?',
    videoTitle5: 'Tutorial de Emergent',
    
    videoDesc1: 'Video corto sobre los conceptos básicos de inteligencia artificial',
    videoDesc2: 'Aprende cómo la IA está transformando el desarrollo de software',
    videoDesc3: 'Descubre hacia dónde se dirige la programación en la era de la IA',
    videoDesc4: 'Conoce las ventajas de estudiar en Academlo y cómo puede transformar tu carrera',
    videoDesc5: 'Aprende a usar Emergent para crear proyectos increíbles con IA',
    
    // Buttons and Actions
    markAsWatched: 'Marcar como Visto',
    videoCompleted: '¡Video Completado!',
    watchOnYoutube: 'Ver en YouTube',
    
    // Module Specific
    listeningModule: 'Módulo Escucha',
    listeningModuleProgress: 'Progreso del Módulo Escucha',
    listeningModuleCompleted: '¡Módulo Escucha Completado!',
    allVideosWatched: 'Has visto todos los videos educativos. ¡Continúa con el siguiente módulo!',
    of: 'de',
    
    // Error Messages  
    errorLoadingContent: 'Error al cargar el contenido',
    errorUpdatingProgress: 'Error al actualizar el progreso',
    videoCompletedProgress: '¡Video {{videoId}} completado! Progreso: {{progress}}%',
    
    // Academlo Ad
    wantToBeCompleteProgrammer: '¿Quieres ser un programador completo?',
    studyAtAcademlo: 'Estudia en Academlo y complementa tu educación con ACADEMY',
    
    // Landing Page
    learningModules: 'Módulos de Aprendizaje',
    poweredBy: 'Powered by Emergent + Academlo',
    
    // Footer
    footerDescription: 'Plataforma educativa futurista para formar programadores del futuro con IA y Deep Agents.',
    quickActions: 'Acciones Rápidas',
    ourPartners: 'Nuestros Socios',
    joinDeveloperCommunity: 'Únete a nuestra comunidad de desarrolladores',
    madeWith: 'Hecho con',
    secure: 'Seguro',
    responsive: 'Responsivo',
    
    // Dashboard Error Messages
    errorLoadingProgress: 'Error al cargar el progreso',
    certificateDownloadedSuccess: '¡Certificado descargado exitosamente! 🎉',
    errorDownloadingCertificate: 'Error al descargar el certificado',
    completeAllModulesForCertificate: 'Completa todos los módulos para obtener tu certificado',
    
    // Auth Messages
    errorOccurred: 'Ocurrió un error. Por favor intenta de nuevo.',
    
    // Module Messages
    pleaseEnterProjectUrl: 'Por favor ingresa la URL de tu proyecto',
    urlMustBeFromEmergent: 'La URL debe ser de un proyecto creado en Emergent',
    projectSubmittedSuccessfully: '¡Proyecto enviado exitosamente! Módulo completado al 100% 🎉',
    errorSubmittingProject: 'Error al enviar el proyecto',
    
    // Prompt Module Messages
    tipsCompletedProgress: '¡Consejos completados! +20%',
    exampleCompletedProgress: '¡Ejemplo {{itemId}} completado! +20%',
    practiceCompletedProgress: '¡Práctica completada! +20%',
    recordingStarted: 'Grabación iniciada. Habla tu prompt...',
    speechRecognitionError: 'Error en el reconocimiento de voz',
    recordingFinishedTranscribed: 'Grabación finalizada y transcrita',
    recordingFinishedBasicSpeechToText: 'Grabación finalizada. Speech-to-text básico agregado al prompt.',
    recordingStartedBasic: 'Grabación iniciada...',
    microphoneAccessError: 'Error al acceder al micrófono',
    promptCopiedToClipboard: 'Prompt copiado al portapapeles',
    
    // Teorico Module Messages
    questionCompletedProgress: '¡Pregunta {{questionId}} completada! Progreso: {{progress}}%',
    
    // Admin Messages
    errorLoadingUsers: 'Error al cargar usuarios',
    userDeletedSuccessfully: 'Usuario eliminado exitosamente',
    errorDeletingUser: 'Error al eliminar usuario',
    
    // Project Module
    createYourProject: 'Crea tu Proyecto con Emergent',
    yourMission: 'Tu misión',
    projectIdeas: 'Ideas de proyectos:',
    requirements: 'Requisitos:',
    submitYourProject: 'Enviar tu Proyecto',
    projectSubmitted: '¡Proyecto Enviado Exitosamente!',
    projectUrl: 'URL de tu proyecto en Emergent:',
    submitProject: 'Enviar Proyecto',
    
    // Instructions
    useEmergentPlatform: 'Utiliza la plataforma Emergent para crear un proyecto que demuestre todo lo que has aprendido en ACADEMY.',
    intelligentChatbot: 'Un chatbot inteligente con IA',
    webAppAdvanced: 'Una aplicación web con funcionalidades avanzadas',
    intelligentAutomation: 'Un sistema de automatización inteligente',
    toolSolvesProblems: 'Una herramienta que resuelva un problema real',
    createdInEmergent: 'Debe ser creado en la plataforma Emergent',
    applyLearned: 'Debe aplicar conceptos aprendidos en el curso',
    functionalDemo: 'Debe ser funcional y demostrable',
    goToEmergent: 'Ir a Emergent para crear tu proyecto',
    onceCreated: 'Una vez que hayas creado tu proyecto, regresa aquí para enviar la URL',
    
    // Final Congratulations
    congratsFutureProgrammer: '¡Felicidades, Programador del Futuro!',
    completedAllModules: 'Has completado todos los módulos de ACADEMY. ¡Ya estás listo para obtener tu certificado!',
    viewCertificateInDashboard: 'Ver Certificado en Dashboard'
  }
};

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState(localStorage.getItem('academy_language') || 'es');

  const toggleLanguage = () => {
    const newLanguage = language === 'es' ? 'en' : 'es';
    setLanguage(newLanguage);
    localStorage.setItem('academy_language', newLanguage);
  };

  const t = (key, variables = {}) => {
    let translation = translations[language][key] || key;
    
    // Replace variables in the format {{variableName}}
    Object.keys(variables).forEach(variable => {
      const regex = new RegExp(`{{${variable}}}`, 'g');
      translation = translation.replace(regex, variables[variable]);
    });
    
    return translation;
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
