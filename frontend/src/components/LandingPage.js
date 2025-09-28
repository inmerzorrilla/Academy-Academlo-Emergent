import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { ChatWidget } from './ChatWidget';
import { SocialLinks } from './SocialLinks';

export const LandingPage = () => {
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const [showAuth, setShowAuth] = useState(false);

  return (
    <div className="min-h-screen futuristic-bg relative">
      <div className="particles"></div>
      
      {/* Header */}
      <header className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <img 
              src="https://customer-assets.emergentagent.com/job_a9e7c5a2-be4f-4a1e-8ab4-c5b1c9d69107/artifacts/tfw28vr3_Logo.png" 
              alt="Academy Logo" 
              className="h-12 w-12 rounded-lg"
            />
            <div>
              <h1 className="text-2xl font-bold text-gradient glitch" data-text="ACADEMY">
                ACADEMY
              </h1>
              <p className="text-sm text-cyan-400">{t('poweredBy')}</p>
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
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 pt-20 pb-32">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <div 
            className="mb-8 animate-float"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1674027444485-cec3da58eef4?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1Nzl8MHwxfHNlYXJjaHwyfHxhcnRpZmljaWFsJTIwaW50ZWxsaWdlbmNlfGVufDB8fHx8MTc1ODk5MjQwNnww&ixlib=rb-4.1.0&q=85')`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              borderRadius: '50%',
              width: '200px',
              height: '200px',
              margin: '0 auto',
              border: '4px solid var(--primary-cyan)',
              boxShadow: '0 0 50px rgba(0, 212, 255, 0.5)'
            }}
          ></div>
          
          <h1 className="text-6xl md:text-8xl font-bold mb-6 text-gradient leading-tight">
            {t('welcome')}
          </h1>
          
          <h2 className="text-2xl md:text-4xl font-semibold mb-8 text-cyan-300">
            {t('subtitle')}
          </h2>
          
          <p className="text-xl md:text-2xl mb-12 max-w-4xl mx-auto text-gray-300 leading-relaxed">
            {t('description')}
          </p>
          
          <div className="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-8">
            <Link 
              to="/auth" 
              className="btn-futuristic text-lg px-8 py-4"
              data-testid="get-started-btn"
            >
              <i className="fas fa-rocket mr-3"></i>
              {t('getStarted')}
            </Link>
            
            <button className="btn-ghost text-lg px-8 py-4">
              <i className="fas fa-play mr-3"></i>
              {t('learnMore')}
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 py-20">
        <div className="max-w-7xl mx-auto px-6">
          <h3 className="text-4xl font-bold text-center mb-16 text-gradient">
            {t('learningModules')}
          </h3>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: 'fa-brain',
                title: t('teorico'),
                description: t('teoricoDesc'),
                color: 'from-blue-500 to-cyan-500'
              },
              {
                icon: 'fa-headphones',
                title: t('escucha'),
                description: 'Videos educativos cuidadosamente seleccionados para tu aprendizaje',
                color: 'from-green-500 to-teal-500'
              },
              {
                icon: 'fa-terminal',
                title: t('prompt'),
                description: 'Practica con prompts avanzados y desarrolla habilidades de IA',
                color: 'from-purple-500 to-pink-500'
              },
              {
                icon: 'fa-project-diagram',
                title: t('proyecto'),
                description: t('proyectoDesc'),
                color: 'from-orange-500 to-red-500'
              }
            ].map((module, index) => (
              <div key={index} className="glass-card p-6 text-center hover:scale-105 transition-transform">
                <div className={`w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r ${module.color} flex items-center justify-center`}>
                  <i className={`fas ${module.icon} text-2xl text-white`}></i>
                </div>
                <h4 className="text-xl font-semibold mb-3 text-gradient">{module.title}</h4>
                <p className="text-gray-400">{module.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-20">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="glass-card p-12">
            <h3 className="text-4xl font-bold mb-6 text-gradient">
              {t('wantToBeCompleteProgrammer')}
            </h3>
            <p className="text-xl mb-8 text-gray-300">
              {t('studyAcademlo')}
            </p>
            <a 
              href="https://www.academlo.com/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-futuristic text-lg px-8 py-4 inline-block"
              data-testid="academlo-link"
            >
              <i className="fas fa-external-link-alt mr-3"></i>
              {t('visitAcademlo')}
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 py-12 border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col lg:flex-row justify-between items-center space-y-6 lg:space-y-0 lg:space-x-8">
            <div className="flex flex-col sm:flex-row items-center space-y-4 sm:space-y-0 sm:space-x-8">
              <a 
                href="https://synapsys-technology.abacusai.app/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-ghost"
                data-testid="synapsys-link"
              >
                <i className="fas fa-bolt mr-2"></i>
                {t('urgentProject')} - Synapsys
              </a>
              
              <a 
                href="https://wa.me/528136037100" 
                target="_blank" 
                rel="noopener noreferrer"
                className="btn-ghost text-green-400 border-green-400 hover:bg-green-400"
                data-testid="whatsapp-link"
              >
                <i className="fab fa-whatsapp mr-2"></i>
                {t('urgentHelp')}
              </a>
            </div>
            
            {/* PayPal Donation Section - Middle */}
            <div className="text-center">
              <h4 className="text-lg font-semibold text-cyan-400 mb-3">
                {t('useful')} {t('buyMeACoffee')} ☕
              </h4>
              <div className="flex justify-center mb-2">
                <a 
                  href="https://paypal.me/pastorinmerzorrilla" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white px-4 py-2 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                >
                  <svg 
                    className="w-5 h-5 mr-2" 
                    viewBox="0 0 24 24" 
                    fill="currentColor"
                  >
                    <path d="M7.076 21.337H2.47a.641.641 0 0 1-.633-.74L4.944.901C5.026.382 5.474 0 5.998 0h7.46c2.57 0 4.578.543 5.69 1.81 1.01 1.15 1.304 2.42 1.012 4.287-.023.143-.047.288-.077.437-.983 5.05-4.349 6.797-8.647 6.797h-2.19c-.524 0-.968.382-1.05.9l-1.12 7.106zm1.262-8.24h2.798c1.31 0 2.417-.718 2.89-1.866.473-1.148.265-2.553-.543-3.665-.808-1.112-2.240-1.789-3.738-1.789H7.392l1.946 7.32z"/>
                  </svg>
                  <span className="font-medium text-sm">
                    PayPal
                  </span>
                </a>
              </div>
              <p className="text-xs text-gray-500">
                {t('supportMessage')}
              </p>
            </div>
            
            <SocialLinks />
          </div>
          
          <div className="mt-8 pt-8 border-t border-gray-800 text-center">
            <p className="text-gray-400">
              © 2025 ACADEMY - {t('poweredBy')}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Quantum Intelligence • Digital Autonomy • Augmented Reality
            </p>
          </div>
        </div>
      </footer>

      <ChatWidget />
    </div>
  );
};
