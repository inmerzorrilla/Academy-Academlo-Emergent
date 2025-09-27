import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import { LandingPage } from './components/LandingPage';
import { AuthPage } from './components/AuthPage';
import { Dashboard } from './components/Dashboard';
import { AdminDashboard } from './components/AdminDashboard';
import { ModuleTeorico } from './components/modules/ModuleTeorico';
import { ModuleEscucha } from './components/modules/ModuleEscucha';
import { ModulePrompt } from './components/modules/ModulePrompt';
import { ModuleProyecto } from './components/modules/ModuleProyecto';
import { LanguageProvider } from './contexts/LanguageContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Toaster } from 'sonner';

function AppContent() {
  const { user, isAuthenticated } = useAuth();

  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/auth" element={<AuthPage />} />
        <Route 
          path="/dashboard" 
          element={isAuthenticated ? <Dashboard /> : <Navigate to="/auth" />} 
        />
        <Route 
          path="/admin" 
          element={isAuthenticated && user?.is_admin ? <AdminDashboard /> : <Navigate to="/dashboard" />} 
        />
        <Route 
          path="/module/teorico" 
          element={isAuthenticated ? <ModuleTeorico /> : <Navigate to="/auth" />} 
        />
        <Route 
          path="/module/escucha" 
          element={isAuthenticated ? <ModuleEscucha /> : <Navigate to="/auth" />} 
        />
        <Route 
          path="/module/prompt" 
          element={isAuthenticated ? <ModulePrompt /> : <Navigate to="/auth" />} 
        />
        <Route 
          path="/module/proyecto" 
          element={isAuthenticated ? <ModuleProyecto /> : <Navigate to="/auth" />} 
        />
      </Routes>
      <Toaster position="top-right" richColors />
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <LanguageProvider>
        <ThemeProvider>
          <Router>
            <AppContent />
          </Router>
        </ThemeProvider>
      </LanguageProvider>
    </AuthProvider>
  );
}

export default App;
