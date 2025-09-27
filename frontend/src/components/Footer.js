import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { SocialLinks } from './SocialLinks';

export const Footer = () => {
  const { t } = useLanguage();

  return (
    <footer className="relative z-10 bg-gradient-to-r from-gray-900/90 to-black/90 backdrop-blur-md border-t border-cyan-500/20">
      {/* Main Footer Content */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          {/* Logo Section */}
          <div className="md:col-span-1">
            <div className="flex items-center space-x-3 mb-4">
              <img 
                src="https://customer-assets.emergentagent.com/job_a9e7c5a2-be4f-4a1e-8ab4-c5b1c9d69107/artifacts/tfw28vr3_Logo.png" 
                alt="Academy Logo" 
                className="h-12 w-12 rounded-lg"
              />
              <div>
                <h3 className="text-2xl font-bold text-gradient">ACADEMY</h3>
                <p className="text-sm text-cyan-400">Future Developers</p>
              </div>
            </div>
            <p className="text-gray-400 text-sm leading-relaxed">
              {t('footerDescription')}
            </p>
          </div>

          {/* Quick Actions */}
          <div className="md:col-span-1">
            <h4 className="text-lg font-semibold text-cyan-400 mb-4">
              {t('quickActions')}
            </h4>
            <div className="space-y-3">
              <a 
                href="https://synapsys-technology.abacusai.app/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center text-gray-300 hover:text-cyan-400 transition-colors group"
                data-testid="synapsys-link"
              >
                <i className="fas fa-bolt mr-3 text-yellow-400 group-hover:text-yellow-300"></i>
                <span className="text-sm">{t('urgentProject')} - Synapsys</span>
              </a>
              
              <a 
                href="https://wa.me/528136037100" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center text-gray-300 hover:text-green-400 transition-colors group"
                data-testid="whatsapp-link"
              >
                <i className="fab fa-whatsapp mr-3 text-green-400 group-hover:text-green-300"></i>
                <span className="text-sm">{t('urgentHelp')}</span>
              </a>
            </div>
          </div>

          {/* Partners */}
          <div className="md:col-span-1">
            <h4 className="text-lg font-semibold text-cyan-400 mb-4">
              {t('ourPartners')}
            </h4>
            <div className="space-y-3">
              <a 
                href="https://www.academlo.com/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center text-gray-300 hover:text-cyan-400 transition-colors group"
              >
                <i className="fas fa-graduation-cap mr-3 text-blue-400 group-hover:text-blue-300"></i>
                <span className="text-sm">Academlo</span>
              </a>
              <a 
                href="https://app.emergent.sh/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center text-gray-300 hover:text-cyan-400 transition-colors group"
              >
                <i className="fas fa-rocket mr-3 text-purple-400 group-hover:text-purple-300"></i>
                <span className="text-sm">Emergent</span>
              </a>
            </div>
          </div>

          {/* Social Networks */}
          <div className="md:col-span-1">
            <h4 className="text-lg font-semibold text-cyan-400 mb-4 flex items-center">
              <i className="fas fa-heart mr-2 text-red-400"></i>
              {t('followUs')}
            </h4>
            <div className="flex flex-col space-y-3">
              <SocialLinks />
              <p className="text-xs text-gray-500 mt-3">
                {t('joinDeveloperCommunity')}
              </p>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-gray-700 mb-6"></div>

        {/* Bottom Section */}
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <div className="text-center md:text-left">
            <p className="text-gray-400 text-sm">
              © 2025 ACADEMY - Powered by 
              <span className="text-cyan-400 mx-1">Emergent</span> + 
              <span className="text-blue-400 mx-1">Academlo</span>
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Quantum Intelligence • Digital Autonomy • Augmented Reality
            </p>
          </div>
          
          <div className="flex items-center space-x-6 text-sm text-gray-400">
            <span className="flex items-center">
              <i className="fas fa-code mr-2 text-cyan-400"></i>
              {t('language') === 'es' ? 'Hecho con' : 'Made with'} ❤️
            </span>
            <span className="flex items-center">
              <i className="fas fa-shield-alt mr-2 text-green-400"></i>
              {t('language') === 'es' ? 'Seguro' : 'Secure'}
            </span>
            <span className="flex items-center">
              <i className="fas fa-mobile-alt mr-2 text-purple-400"></i>
              {t('language') === 'es' ? 'Responsivo' : 'Responsive'}
            </span>
          </div>
        </div>
      </div>

      {/* Animated Background Effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-cyan-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>
    </footer>
  );
};