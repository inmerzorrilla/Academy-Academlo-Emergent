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
    <header className="relative z-10 p-4 sm:p-6 border-b border-gray-800">
      <div className="max-w-7xl mx-auto">
        {/* Top Navigation */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-3 sm:space-y-0 mb-4">
          <div className="flex items-center">
            <Link to="/dashboard" className="btn-ghost text-sm">
              <i className="fas fa-arrow-left mr-2"></i>
              {t('backToDashboard')}
            </Link>
          </div>
          
          <div className="flex items-center space-x-2 flex-wrap">
            {showLanguageButtons && (
              <>
                <button 
                  onClick={toggleLanguage}
                  className={`px-3 py-1 text-xs rounded transition-all ${
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
                  className={`px-3 py-1 text-xs rounded transition-all ${
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
                className="btn-ghost text-sm"
                data-testid="theme-toggle"
              >
                <i className={`fas ${theme === 'dark' ? 'fa-sun' : 'fa-moon'} mr-2`}></i>
              </button>
            )}
            
            <Badge className="bg-orange-500 text-xs">
              {progress}% {t('completed')}
            </Badge>
          </div>
        </div>
        
        {/* Module Title */}
        <div className="text-center">
          <h1 className="text-xl sm:text-2xl font-bold text-gradient">{moduleName}</h1>
          <p className="text-xs sm:text-sm text-cyan-400">{moduleSubtitle}</p>
        </div>
      </div>
    </header>
  );
};