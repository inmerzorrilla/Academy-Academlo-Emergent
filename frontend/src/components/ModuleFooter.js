import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';

export const ModuleFooter = () => {
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();

  return (
    <footer className="relative z-10 bg-gradient-to-r from-gray-900/90 to-black/90 backdrop-blur-md border-t border-cyan-500/20 mt-12">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="grid md:grid-cols-3 gap-6 mb-6">
          {/* Language and Theme Controls */}
          <div className="flex flex-col space-y-4">
            <h4 className="text-lg font-semibold text-cyan-400 mb-2">
              {t('language') === 'es' ? 'Configuración' : 'Settings'}
            </h4>
            <div className="flex space-x-4">
              <button 
                onClick={toggleLanguage}
                className="btn-ghost text-sm flex items-center"
                data-testid="footer-language-toggle"
              >
                <i className="fas fa-globe mr-2"></i>
                <span className="font-semibold">{language === 'es' ? 'English' : 'Español'}</span>
              </button>
              
              <button 
                onClick={toggleTheme}
                className="btn-ghost text-sm flex items-center"
                data-testid="footer-theme-toggle"
              >
                <i className={`fas ${theme === 'dark' ? 'fa-sun' : 'fa-moon'} mr-2`}></i>
                <span className="font-semibold">
                  {theme === 'dark' 
                    ? (language === 'es' ? 'Claro' : 'Light')
                    : (language === 'es' ? 'Oscuro' : 'Dark')
                  }
                </span>
              </button>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-col space-y-3">
            <h4 className="text-lg font-semibold text-cyan-400 mb-2">
              {t('language') === 'es' ? 'Ayuda Rápida' : 'Quick Help'}
            </h4>
            <a 
              href="https://wa.me/528136037100" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center text-gray-300 hover:text-green-400 transition-colors group text-sm"
              data-testid="whatsapp-footer"
            >
              <i className="fab fa-whatsapp mr-3 text-green-400 group-hover:text-green-300"></i>
              <span>{t('urgentHelp')}</span>
            </a>
            
            <a 
              href="https://synapsys-technology.abacusai.app/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center text-gray-300 hover:text-cyan-400 transition-colors group text-sm"
              data-testid="synapsys-footer"
            >
              <i className="fas fa-bolt mr-3 text-yellow-400 group-hover:text-yellow-300"></i>
              <span>{t('urgentProject')} - Synapsys</span>
            </a>
          </div>

          {/* Partners */}
          <div className="flex flex-col space-y-3">
            <h4 className="text-lg font-semibold text-cyan-400 mb-2">
              {t('language') === 'es' ? 'Nuestros Socios' : 'Our Partners'}
            </h4>
            <a 
              href="https://www.academlo.com/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center text-gray-300 hover:text-cyan-400 transition-colors group text-sm"
            >
              <i className="fas fa-graduation-cap mr-3 text-blue-400 group-hover:text-blue-300"></i>
              <span>Academlo</span>
            </a>
            <a 
              href="https://app.emergent.sh/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center text-gray-300 hover:text-cyan-400 transition-colors group text-sm"
            >
              <i className="fas fa-rocket mr-3 text-purple-400 group-hover:text-purple-300"></i>
              <span>Emergent</span>
            </a>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-gray-700 pt-4">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-2 md:space-y-0">
            <div className="text-center md:text-left">
              <p className="text-gray-400 text-sm">
                © 2025 ACADEMY - {t('language') === 'es' ? 'Desarrollado por' : 'Powered by'} 
                <span className="text-cyan-400 mx-1">Emergent</span> + 
                <span className="text-blue-400 mx-1">Academlo</span>
              </p>
            </div>
            
            <div className="flex items-center space-x-4 text-xs text-gray-400">
              <span className="flex items-center">
                <i className="fas fa-shield-alt mr-1 text-green-400"></i>
                {t('language') === 'es' ? 'Seguro' : 'Secure'}
              </span>
              <span className="flex items-center">
                <i className="fas fa-mobile-alt mr-1 text-purple-400"></i>
                {t('language') === 'es' ? 'Responsivo' : 'Responsive'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};