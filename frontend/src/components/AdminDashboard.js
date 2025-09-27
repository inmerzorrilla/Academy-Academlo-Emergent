import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const AdminDashboard = () => {
  const { user, logout } = useAuth();
  const { t, language, toggleLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalUsers: 0,
    completedCourses: 0,
    averageProgress: 0
  });

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await axios.get(`${API}/admin/users`);
      const usersData = response.data;
      setUsers(usersData);
      
      // Calculate stats
      const totalUsers = usersData.length;
      const completedCourses = usersData.filter(userData => {
        const progress = userData.progress;
        return (progress.teorico_progress + progress.escucha_progress + 
                progress.prompt_progress + progress.proyecto_progress) / 4 === 100;
      }).length;
      
      const averageProgress = usersData.reduce((acc, userData) => {
        const progress = userData.progress;
        const totalProgress = (progress.teorico_progress + progress.escucha_progress + 
                             progress.prompt_progress + progress.proyecto_progress) / 4;
        return acc + totalProgress;
      }, 0) / totalUsers || 0;
      
      setStats({ totalUsers, completedCourses, averageProgress: Math.round(averageProgress) });
    } catch (error) {
      console.error('Error fetching users:', error);
      toast.error(t('errorLoadingUsers'));
    } finally {
      setLoading(false);
    }
  };

  const deleteUser = async (userId) => {
    if (window.confirm('¿Estás seguro de que quieres eliminar este usuario?')) {
      try {
        await axios.delete(`${API}/admin/users/${userId}`);
        toast.success(t('userDeletedSuccessfully'));
        fetchUsers(); // Refresh the list
      } catch (error) {
        console.error('Error deleting user:', error);
        toast.error(t('errorDeletingUser'));
      }
    }
  };

  const getTotalProgress = (progress) => {
    return Math.round((progress.teorico_progress + progress.escucha_progress + 
                     progress.prompt_progress + progress.proyecto_progress) / 4);
  };

  if (loading) {
    return (
      <div className="min-h-screen futuristic-bg flex items-center justify-center">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen futuristic-bg relative" data-testid="admin-dashboard">
      <div className="particles"></div>
      
      {/* Header */}
      <header className="relative z-10 p-6 border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <img 
              src="https://customer-assets.emergentagent.com/job_a9e7c5a2-be4f-4a1e-8ab4-c5b1c9d69107/artifacts/tfw28vr3_Logo.png" 
              alt="Academy Logo" 
              className="h-12 w-12 rounded-lg"
            />
            <div>
              <h1 className="text-2xl font-bold text-gradient">ACADEMY - Admin</h1>
              <p className="text-sm text-cyan-400">Panel de Administración</p>
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
            </button>
            
            <Link to="/dashboard" className="btn-ghost text-sm">
              <i className="fas fa-user mr-2"></i>
              Mi Dashboard
            </Link>
            
            <button 
              onClick={logout}
              className="btn-ghost text-sm text-red-400 border-red-400 hover:bg-red-400"
            >
              <i className="fas fa-sign-out-alt mr-2"></i>
              Salir
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Stats Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="text-lg text-gradient flex items-center">
                <i className="fas fa-users mr-2"></i>
                Total de Usuarios
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-cyan-400">{stats.totalUsers}</div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="text-lg text-gradient flex items-center">
                <i className="fas fa-graduation-cap mr-2"></i>
                Cursos Completados
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-400">{stats.completedCourses}</div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="text-lg text-gradient flex items-center">
                <i className="fas fa-chart-line mr-2"></i>
                Progreso Promedio
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-400">{stats.averageProgress}%</div>
            </CardContent>
          </Card>
        </div>

        {/* Users Table */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-2xl text-gradient">
              Gestión de Usuarios
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-4 text-gray-300">Usuario</th>
                    <th className="text-left py-3 px-4 text-gray-300">Contacto</th>
                    <th className="text-left py-3 px-4 text-gray-300">Progreso</th>
                    <th className="text-left py-3 px-4 text-gray-300">Módulos</th>
                    <th className="text-left py-3 px-4 text-gray-300">Registro</th>
                    <th className="text-left py-3 px-4 text-gray-300">Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(({ user: userData, progress }) => {
                    const totalProgress = getTotalProgress(progress);
                    return (
                      <tr key={userData.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="py-4 px-4">
                          <div>
                            <div className="font-semibold text-white">{userData.name}</div>
                            {userData.is_admin && (
                              <Badge className="mt-1 bg-purple-500">Admin</Badge>
                            )}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="text-sm">
                            <div className="text-gray-300">{userData.email}</div>
                            <div className="text-gray-400">{userData.phone}</div>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center space-x-2">
                            <Progress value={totalProgress} className="w-20 h-2" />
                            <span className="text-sm font-medium text-cyan-400">
                              {totalProgress}%
                            </span>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex space-x-2">
                            <Badge 
                              variant={progress.teorico_progress === 100 ? "default" : "secondary"}
                              className={progress.teorico_progress === 100 ? "bg-green-500" : ""}
                            >
                              T: {progress.teorico_progress}%
                            </Badge>
                            <Badge 
                              variant={progress.escucha_progress === 100 ? "default" : "secondary"}
                              className={progress.escucha_progress === 100 ? "bg-green-500" : ""}
                            >
                              E: {progress.escucha_progress}%
                            </Badge>
                            <Badge 
                              variant={progress.prompt_progress === 100 ? "default" : "secondary"}
                              className={progress.prompt_progress === 100 ? "bg-green-500" : ""}
                            >
                              P: {progress.prompt_progress}%
                            </Badge>
                            <Badge 
                              variant={progress.proyecto_progress === 100 ? "default" : "secondary"}
                              className={progress.proyecto_progress === 100 ? "bg-green-500" : ""}
                            >
                              Pr: {progress.proyecto_progress}%
                            </Badge>
                          </div>
                        </td>
                        <td className="py-4 px-4 text-sm text-gray-400">
                          {new Date(userData.created_at).toLocaleDateString('es-ES')}
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex space-x-2">
                            <Button 
                              onClick={() => window.open(`mailto:${userData.email}`, '_blank')}
                              size="sm"
                              className="btn-ghost text-xs"
                              data-testid={`email-user-${userData.id}`}
                            >
                              <i className="fas fa-envelope"></i>
                            </Button>
                            <Button 
                              onClick={() => deleteUser(userData.id)}
                              size="sm"
                              className="btn-ghost text-xs text-red-400 border-red-400 hover:bg-red-400"
                              data-testid={`delete-user-${userData.id}`}
                            >
                              <i className="fas fa-trash"></i>
                            </Button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};
