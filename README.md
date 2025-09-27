# ACADEMY - Educational Platform 🚀

**ACADEMY** is a cutting-edge educational web application designed to train the future programmers in AI, Deep Agents, and emerging technologies. Built with a modern tech stack, it offers an immersive learning experience in both Spanish and English.

## 🌟 Features

### 🎓 Complete Learning System
- **4 Progressive Modules**: Theoretical, Listen, Prompt, and Project
- **Bilingual Support**: Full Spanish/English interface with dynamic translation
- **Progress Tracking**: Real-time progress monitoring with visual indicators
- **Certificate Generation**: Professional PDF certificates upon course completion

### 🤖 AI-Powered Features
- **Intelligent Chatbot**: Claude-powered assistant for comprehensive support
- **Speech-to-Text**: Voice input for prompt engineering practice
- **Smart Content**: Contextual learning materials and examples

### 🎨 Modern UI/UX
- **Futuristic Design**: Dark/Light theme with glass morphism effects
- **Fully Responsive**: Optimized for all devices (desktop, tablet, mobile)
- **Interactive Elements**: Engaging animations and transitions
- **Accessibility**: WCAG compliant with proper contrast and keyboard navigation

### 👥 Admin Features
- **User Management**: Complete CRUD operations for user administration
- **Analytics Dashboard**: Progress tracking and course completion statistics
- **Super Admin Panel**: Advanced administrative capabilities

## 🛠 Tech Stack

### Frontend
- **React 18** - Modern React with hooks and context
- **Tailwind CSS** - Utility-first CSS framework
- **Shadcn/UI** - High-quality component library
- **React Router** - Client-side routing
- **Axios** - HTTP client for API calls
- **Sonner** - Toast notifications

### Backend
- **FastAPI** - High-performance Python web framework
- **MongoDB** - NoSQL database with Motor (async driver)
- **JWT Authentication** - Secure user authentication
- **ReportLab** - PDF certificate generation
- **Emergent Integrations** - LLM integration (Claude)

### DevOps & Deployment
- **Docker** - Containerization
- **Kubernetes** - Container orchestration
- **Supervisor** - Process management
- **Nginx** - Reverse proxy (via Kubernetes ingress)

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.11+
- MongoDB
- Yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/inmerzorrilla/Academy-Academlo-Emergent.git
   cd Academy-Academlo-Emergent
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   cp .env.example .env  # Configure your environment variables
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   yarn install
   cp .env.example .env  # Configure your environment variables
   ```

4. **Database Setup**
   ```bash
   # Make sure MongoDB is running
   cd backend
   python create_admin.py  # Create initial admin user
   ```

5. **Start the application**
   ```bash
   # Backend (from backend directory)
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   
   # Frontend (from frontend directory)
   yarn start
   ```

## 📚 Course Modules

### 1. Módulo Teórico (Theoretical Module)
- Deep Agents fundamentals
- AI programming concepts
- Interactive Q&A system
- Progress tracking

### 2. Módulo Escucha (Listen Module)
- Curated educational videos
- YouTube integration
- Video completion tracking
- Expert insights

### 3. Módulo Prompt (Prompt Module)
- Prompt engineering best practices
- Interactive examples
- Speech-to-text practice
- Real-world applications

### 4. Módulo Proyecto (Project Module)
- Hands-on project development
- Emergent platform integration
- Portfolio building
- Final assessment

## 🌍 Internationalization

The platform supports full bilingual functionality:
- **Spanish (es)**: Default language
- **English (en)**: Complete translation coverage
- **Dynamic switching**: Real-time language toggle
- **Persistent preferences**: Language choice saved in localStorage

## 🔧 Configuration

### Environment Variables

**Frontend (.env)**
```
REACT_APP_BACKEND_URL=your_backend_url
WDS_SOCKET_PORT=443
```

**Backend (.env)**
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=academy_database
CORS_ORIGINS=*
JWT_SECRET=your_jwt_secret
EMERGENT_LLM_KEY=your_emergent_llm_key
```

## 🎨 Design System

### Color Palette
- **Primary Cyan**: #00d4ff
- **Primary Dark**: #0a0e1a
- **Secondary Dark**: #1a1f2e
- **Accent Colors**: Various gradients for visual hierarchy

### Typography
- **Font Family**: Space Grotesk
- **Font Weights**: 300, 400, 500, 600, 700
- **Responsive scaling**: Fluid typography for all screen sizes

## 📱 Mobile Optimization

- **Responsive breakpoints**: sm (640px), md (768px), lg (1024px), xl (1280px)
- **Touch-friendly**: Optimized touch targets and gestures
- **Performance**: Lazy loading and code splitting
- **PWA-ready**: Service worker and manifest configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 Partners

- **[Academlo](https://www.academlo.com/)** - Educational Excellence
- **[Emergent](https://app.emergent.sh/)** - AI Platform Integration

## 📞 Support

- **WhatsApp**: +528136037100
- **Email**: support@academy.com
- **Website**: https://academy.emergent.sh

## 🎯 Roadmap

- [ ] Advanced AI integrations
- [ ] Real-time collaboration features
- [ ] Mobile app development
- [ ] Extended language support
- [ ] Advanced analytics dashboard

---

**Built with ❤️ by the ACADEMY Team**

*Quantum Intelligence • Digital Autonomy • Augmented Reality*
