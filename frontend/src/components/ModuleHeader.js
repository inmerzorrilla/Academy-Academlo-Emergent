import React from 'react';
import { Link } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { Badge } from './ui/badge';

export const ModuleHeader = ({ 
  moduleName, 
  moduleSubtitle, 
  progress = 0, 
  showLanguageButtons = true, 
  showThemeButton = true 
}) => {
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="relative z-10 p-3 sm:p-6 border-b border-gray-800 module-header-mobile">
      <div className="max-w-7xl mx-auto">
        {/* Module Title - Moved to top for mobile */}
        <div className="text-center mb-3 sm:mb-4">
          <h1 className="text-lg sm:text-2xl font-bold text-gradient text-wrap-mobile">{moduleName}</h1>
          <p className="text-xs sm:text-sm text-cyan-400">{moduleSubtitle}</p>
        </div>
        
        {/* Navigation and Controls */}
        <div className="flex flex-col space-y-3 sm:space-y-0 sm:flex-row sm:justify-between sm:items-center">
          {/* Back button */}
          <div className="flex items-center justify-center sm:justify-start">
            <Link to="/dashboard" className="btn-ghost text-xs sm:text-sm">
              <i className="fas fa-arrow-left mr-1 sm:mr-2"></i>
              <span className="hidden sm:inline">{t('backToDashboard')}</span>
              <span className="sm:hidden">{t('back')}</span>
            </Link>
          </div>
          
          {/* Control buttons */}
          <div className="module-header-buttons">
            {showLanguageButtons && (
              <>
                <button 
                  onClick={toggleLanguage}
                  className={`px-2 py-1 text-xs rounded transition-all ${
                    language === 'es' 
                      ? 'bg-cyan-500 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  data-testid="language-toggle-es"
                >
                  ES
                </button>
                <button 
                  onClick={toggleLanguage}
                  className={`px-2 py-1 text-xs rounded transition-all ${
                    language === 'en' 
                      ? 'bg-cyan-500 text-white' 
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  data-testid="language-toggle-en"
                >
                  EN
                </button>
              </>
            )}
            
            {showThemeButton && (
              <button 
                onClick={toggleTheme}
                className="btn-ghost text-xs px-2 py-1"
                data-testid="theme-toggle"
              >
                <i className={`fas ${theme === 'dark' ? 'fa-sun' : 'fa-moon'}`}></i>
              </button>
            )}
            
            <Badge className="bg-orange-500 text-xs badge px-2 py-1">
              {progress}%
            </Badge>
          </div>
        </div>
      </div>
    </header>
  );
};