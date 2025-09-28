import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import axios from 'axios';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false); // New state for expanded mode
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      message: '¡Hola! Soy el asistente de ACADEMY 🚀 Estoy aquí para resolver todas tus dudas sobre nuestros cursos de programación e IA. ¿En qué puedo ayudarte hoy?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = {
      type: 'user',
      message: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const messageToSend = inputMessage;
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(
        `${API}/chat`, 
        { message: messageToSend },
        {
          headers: {
            'Content-Type': 'application/json'
          },
          timeout: 10000
        }
      );

      const botMessage = {
        type: 'bot',
        message: response.data.response || 'No pude generar una respuesta.',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        type: 'bot',
        message: 'Lo siento, ocurrió un error temporal. El servicio de chat está experimentando problemas. ¿Puedes intentar de nuevo?',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-widget">
      <Card className={`fixed bottom-20 ${isExpanded ? 'right-4 w-96 h-[500px]' : 'right-4 w-80 h-96'} z-40 flex flex-col glass-card shadow-2xl border-cyan-500/30 ${isOpen ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0 pointer-events-none'} transition-all duration-300`}>
          <CardHeader className="pb-2">
            <div className="flex justify-between items-center">
              <CardTitle className="text-lg text-gradient flex items-center">
                <i className="fas fa-robot mr-2"></i>
                ACADEMY Assistant
              </CardTitle>
              <div className="flex items-center space-x-2">
                <Button 
                  onClick={() => setIsExpanded(!isExpanded)}
                  size="sm"
                  className="btn-ghost text-xs"
                  data-testid="expand-chat"
                  title={isExpanded ? 'Contraer chat' : 'Expandir chat'}
                >
                  <i className={`fas ${isExpanded ? 'fa-compress' : 'fa-expand'}`}></i>
                </Button>
                <Button 
                  onClick={() => setIsOpen(false)}
                  size="sm"
                  className="btn-ghost text-xs"
                  data-testid="close-chat"
                >
                  <i className="fas fa-times"></i>
                </Button>
              </div>
            </div>
            <div className="flex items-center text-xs text-green-400">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
              En línea 24/7
            </div>
          </CardHeader>
          
          <CardContent className="flex-1 flex flex-col p-4">
            <div className="flex-1 overflow-y-auto mb-4 space-y-3" style={{ maxHeight: isExpanded ? '360px' : '240px' }}>
              {messages.map((msg, index) => (
                <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div 
                    className={`${isExpanded ? 'max-w-sm' : 'max-w-xs'} p-3 rounded-lg text-sm ${
                      msg.type === 'user' 
                        ? 'bg-cyan-500 text-white' 
                        : 'bg-gray-700 text-gray-100'
                    }`}
                  >
                    {msg.message}
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-700 text-gray-100 max-w-xs p-3 rounded-lg text-sm">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
            
            <form onSubmit={sendMessage} className="flex space-x-2">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Escribe tu pregunta..."
                className="flex-1 text-sm bg-gray-800 border-gray-600 focus:border-cyan-500 text-white placeholder:text-gray-400"
                disabled={isLoading}
                data-testid="chat-input"
              />
              <Button 
                type="submit" 
                size="sm"
                disabled={isLoading || !inputMessage.trim()}
                className="btn-futuristic px-3"
                data-testid="send-message"
              >
                <i className="fas fa-paper-plane"></i>
              </Button>
            </form>
          </CardContent>
        </Card>
      
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="chat-button"
        data-testid="chat-toggle"
      >
        <i className={`fas ${isOpen ? 'fa-times' : 'fa-comments'}`}></i>
      </button>
    </div>
  );
};
