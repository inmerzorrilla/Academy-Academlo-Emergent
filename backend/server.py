from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import bcrypt
import jwt
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Settings
SECRET_KEY = os.environ.get('JWT_SECRET', 'academy-secret-key-change-in-production')
ALGORITHM = "HS256"

# Create the main app without a prefix
app = FastAPI(title="ACADEMY API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()

# Models
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    phone: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    phone: str
    is_admin: bool = False
    is_super_admin: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None

class UserProgress(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    teorico_progress: int = 0  # 0-100
    escucha_progress: int = 0  # 0-100
    prompt_progress: int = 0   # 0-100
    proyecto_progress: int = 0 # 0-100
    teorico_completed: List[int] = []  # completed question numbers
    escucha_completed: List[int] = []  # completed video numbers
    prompt_completed: bool = False
    prompt_tips_completed: bool = False
    prompt_examples_completed: List[int] = []  # completed example numbers
    prompt_practice_completed: bool = False
    proyecto_url: Optional[str] = None
    certificate_generated: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProgressUpdate(BaseModel):
    module: str  # 'teorico', 'escucha', 'prompt', 'proyecto'
    item_id: Optional[int] = None  # for specific questions/videos
    proyecto_url: Optional[str] = None
    prompt_section: Optional[str] = None  # 'tips', 'examples', 'practice'

class ChatMessage(BaseModel):
    message: str
    response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Utility functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc).timestamp() + 86400  # 24 hours
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one({"id": user_id})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return User(**user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Auth Routes
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        name=user_data.name,
        email=user_data.email,
        phone=user_data.phone
    )
    
    user_dict = user.dict()
    user_dict["password"] = hash_password(user_data.password)
    
    await db.users.insert_one(user_dict)
    
    # Create initial progress
    progress = UserProgress(user_id=user.id)
    await db.user_progress.insert_one(progress.dict())
    
    # Create JWT token
    token = create_jwt_token(user.id, user.email)
    
    return {"user": user, "token": token, "message": "Registration successful"}

@api_router.post("/auth/login")
async def login(login_data: UserLogin):
    user = await db.users.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    await db.users.update_one(
        {"id": user["id"]},
        {"$set": {"last_login": datetime.now(timezone.utc)}}
    )
    
    # Create JWT token
    token = create_jwt_token(user["id"], user["email"])
    
    user_obj = User(**user)
    return {"user": user_obj, "token": token, "message": "Login successful"}

# User Routes
@api_router.get("/user/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user

@api_router.get("/user/progress")
async def get_progress(current_user: User = Depends(get_current_user)):
    progress = await db.user_progress.find_one({"user_id": current_user.id})
    if not progress:
        # Create initial progress if not exists
        progress = UserProgress(user_id=current_user.id)
        await db.user_progress.insert_one(progress.dict())
        return progress
    return UserProgress(**progress)

@api_router.post("/user/progress")
async def update_progress(update: ProgressUpdate, current_user: User = Depends(get_current_user)):
    progress = await db.user_progress.find_one({"user_id": current_user.id})
    if not progress:
        progress = UserProgress(user_id=current_user.id).dict()
    
    if update.module == "teorico" and update.item_id:
        if update.item_id not in progress["teorico_completed"]:
            progress["teorico_completed"].append(update.item_id)
            progress["teorico_progress"] = min(100, len(progress["teorico_completed"]) * 10)  # 10 questions = 100%
    
    elif update.module == "escucha" and update.item_id:
        if update.item_id not in progress["escucha_completed"]:
            progress["escucha_completed"].append(update.item_id)
            # 5 videos = 20% each = 100%
            progress["escucha_progress"] = min(100, len(progress["escucha_completed"]) * 20)
    
    elif update.module == "prompt":
        if update.prompt_section == "tips":
            progress["prompt_tips_completed"] = True
        elif update.prompt_section == "examples" and update.item_id:
            if update.item_id not in progress.get("prompt_examples_completed", []):
                if "prompt_examples_completed" not in progress:
                    progress["prompt_examples_completed"] = []
                progress["prompt_examples_completed"].append(update.item_id)
        elif update.prompt_section == "practice":
            progress["prompt_practice_completed"] = True
        
        # Calculate progress: 20% tips + 20% each example (4 examples) + 20% practice = 100%
        prompt_progress = 0
        if progress.get("prompt_tips_completed", False):
            prompt_progress += 20
        prompt_progress += len(progress.get("prompt_examples_completed", [])) * 20  # 20% per example
        if progress.get("prompt_practice_completed", False):
            prompt_progress += 20
        
        progress["prompt_progress"] = min(100, prompt_progress)
        if progress["prompt_progress"] == 100:
            progress["prompt_completed"] = True
    
    elif update.module == "proyecto" and update.proyecto_url:
        progress["proyecto_url"] = update.proyecto_url
        progress["proyecto_progress"] = 100
    
    progress["updated_at"] = datetime.now(timezone.utc)
    
    await db.user_progress.update_one(
        {"user_id": current_user.id},
        {"$set": progress},
        upsert=True
    )
    
    return UserProgress(**progress)

# Admin Routes
@api_router.get("/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = await db.users.find().to_list(1000)
    users_with_progress = []
    
    for user in users:
        user_obj = User(**user)
        progress = await db.user_progress.find_one({"user_id": user["id"]})
        if progress:
            progress_obj = UserProgress(**progress)
        else:
            progress_obj = UserProgress(user_id=user["id"])
        
        users_with_progress.append({
            "user": user_obj,
            "progress": progress_obj
        })
    
    return users_with_progress

@api_router.delete("/admin/users/{user_id}")
async def delete_user(user_id: str, current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Delete user and their progress
    await db.users.delete_one({"id": user_id})
    await db.user_progress.delete_one({"user_id": user_id})
    
    return {"message": "User deleted successfully"}

# Content Routes
@api_router.get("/content/teorico")
async def get_teorico_content():
    questions = [
        {
            "id": 1,
            "question": "¿Qué es un Deep Agent?",
            "answer": "Un Deep Agent es un sistema de inteligencia artificial que utiliza redes neuronales profundas para procesar información y tomar decisiones de manera autónoma. Combina aprendizaje profundo con agentes inteligentes para resolver problemas complejos.",
            "code": "# Ejemplo básico de un agente\nclass DeepAgent:\n    def __init__(self, model):\n        self.model = model\n        self.memory = []\n    \n    def perceive(self, environment):\n        return self.model.predict(environment)\n    \n    def act(self, action):\n        return self.execute(action)"
        },
        {
            "id": 2,
            "question": "¿Cómo se codifica un agente con Python?",
            "answer": "Se puede codificar usando clases que definan percepciones, acciones y memoria. Se utilizan librerías como TensorFlow, PyTorch para el modelo neuronal.",
            "code": "import tensorflow as tf\nimport numpy as np\n\nclass NeuralAgent:\n    def __init__(self):\n        self.model = tf.keras.Sequential([\n            tf.keras.layers.Dense(64, activation='relu'),\n            tf.keras.layers.Dense(32, activation='relu'),\n            tf.keras.layers.Dense(4, activation='softmax')\n        ])\n    \n    def train(self, X, y):\n        self.model.compile(optimizer='adam', loss='categorical_crossentropy')\n        self.model.fit(X, y, epochs=100)"
        },
        {
            "id": 3,
            "question": "¿Qué es un chatbot?",
            "answer": "Un chatbot es un programa de computadora diseñado para simular conversaciones con usuarios humanos, especialmente a través de internet. Utiliza procesamiento de lenguaje natural (NLP) para entender y responder preguntas.",
            "code": "import openai\n\nclass ChatBot:\n    def __init__(self, api_key):\n        openai.api_key = api_key\n    \n    def respond(self, message):\n        response = openai.ChatCompletion.create(\n            model='gpt-3.5-turbo',\n            messages=[{'role': 'user', 'content': message}]\n        )\n        return response.choices[0].message.content"
        },
        {
            "id": 4,
            "question": "¿Qué son las redes neuronales?",
            "answer": "Las redes neuronales son modelos computacionales inspirados en el cerebro humano. Están compuestas por nodos (neuronas artificiales) interconectados que procesan información mediante pesos y funciones de activación.",
            "code": "import numpy as np\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\n\nclass NeuralNetwork:\n    def __init__(self, input_size, hidden_size, output_size):\n        self.W1 = np.random.randn(input_size, hidden_size)\n        self.W2 = np.random.randn(hidden_size, output_size)\n    \n    def forward(self, X):\n        self.z1 = np.dot(X, self.W1)\n        self.a1 = sigmoid(self.z1)\n        self.z2 = np.dot(self.a1, self.W2)\n        return sigmoid(self.z2)"
        },
        {
            "id": 5,
            "question": "¿Qué es Machine Learning?",
            "answer": "Machine Learning es una rama de la inteligencia artificial que permite a las máquinas aprender y mejorar automáticamente a partir de la experiencia sin ser explícitamente programadas para cada tarea específica.",
            "code": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\n\n# Ejemplo de ML\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n\nmodel = RandomForestClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)\n\naccuracy = model.score(X_test, y_test)\nprint(f'Accuracy: {accuracy}')"
        },
        {
            "id": 6,
            "question": "¿Qué es el procesamiento de lenguaje natural (NLP)?",
            "answer": "NLP es una rama de la IA que ayuda a las computadoras a entender, interpretar y manipular el lenguaje humano. Combina lingüística computacional con machine learning para procesar texto y voz.",
            "code": "import nltk\nfrom textblob import TextBlob\n\n# Análisis de sentimientos\ntext = 'Me encanta la programación'\nblob = TextBlob(text)\n\nsentiment = blob.sentiment.polarity\nif sentiment > 0:\n    print('Sentimiento positivo')\nelif sentiment < 0:\n    print('Sentimiento negativo')\nelse:\n    print('Sentimiento neutral')"
        },
        {
            "id": 7,
            "question": "¿Qué es la visión por computadora?",
            "answer": "La visión por computadora es un campo de la IA que entrena computadoras para interpretar y entender el mundo visual. Utiliza cámaras, datos y algoritmos para identificar y analizar contenido visual.",
            "code": "import cv2\nimport numpy as np\n\n# Detección de bordes\nimage = cv2.imread('image.jpg')\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n\n# Aplicar filtro Canny\nedges = cv2.Canny(gray, 50, 150)\n\ncv2.imshow('Original', image)\ncv2.imshow('Edges', edges)\ncv2.waitKey(0)\ncv2.destroyAllWindows()"
        },
        {
            "id": 8,
            "question": "¿Qué es un algoritmo genético?",
            "answer": "Los algoritmos genéticos son técnicas de optimización inspiradas en la evolución natural. Utilizan operadores como selección, cruce y mutación para encontrar soluciones óptimas a problemas complejos.",
            "code": "import random\n\nclass GeneticAlgorithm:\n    def __init__(self, population_size, mutation_rate):\n        self.population_size = population_size\n        self.mutation_rate = mutation_rate\n    \n    def create_individual(self, length):\n        return [random.randint(0, 1) for _ in range(length)]\n    \n    def fitness(self, individual):\n        # Función de fitness específica del problema\n        return sum(individual)\n    \n    def mutate(self, individual):\n        for i in range(len(individual)):\n            if random.random() < self.mutation_rate:\n                individual[i] = 1 - individual[i]"
        },
        {
            "id": 9,
            "question": "¿Qué es el aprendizaje por refuerzo?",
            "answer": "El aprendizaje por refuerzo es un tipo de machine learning donde un agente aprende a tomar decisiones mediante interacciones con su entorno, recibiendo recompensas o castigos por sus acciones.",
            "code": "import numpy as np\n\nclass QLearningAgent:\n    def __init__(self, states, actions, learning_rate=0.1, discount=0.9):\n        self.q_table = np.zeros((states, actions))\n        self.lr = learning_rate\n        self.gamma = discount\n    \n    def update(self, state, action, reward, next_state):\n        old_value = self.q_table[state, action]\n        next_max = np.max(self.q_table[next_state])\n        \n        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)\n        self.q_table[state, action] = new_value"
        },
        {
            "id": 10,
            "question": "¿Qué es la automatización inteligente?",
            "answer": "La automatización inteligente combina IA, machine learning y automatización robótica de procesos (RPA) para automatizar tareas complejas que requieren toma de decisiones y adaptabilidad.",
            "code": "from selenium import webdriver\nfrom selenium.webdriver.common.by import By\nimport time\n\nclass IntelligentAutomation:\n    def __init__(self):\n        self.driver = webdriver.Chrome()\n    \n    def automate_task(self, url, actions):\n        self.driver.get(url)\n        \n        for action in actions:\n            if action['type'] == 'click':\n                element = self.driver.find_element(By.ID, action['target'])\n                element.click()\n            elif action['type'] == 'input':\n                element = self.driver.find_element(By.ID, action['target'])\n                element.send_keys(action['value'])\n            \n            time.sleep(1)  # Wait between actions"
        }
    ]
    return questions

@api_router.get("/content/escucha")
async def get_escucha_content(lang: str = "es"):
    videos = [
        {
            "id": 1,
            "title": "Introducción a la IA",
            "url": "https://www.youtube.com/shorts/l5xA4iNjyq4",
            "description": "Video corto sobre los conceptos básicos de inteligencia artificial"
        },
        {
            "id": 2,
            "title": "Desarrollo con IA",
            "url": "https://www.youtube.com/watch?v=MGwTIjM-t1I",
            "description": "Aprende cómo la IA está transformando el desarrollo de software"
        },
        {
            "id": 3,
            "title": "El futuro de la programación",
            "url": "https://www.youtube.com/watch?v=1Wg_RJ59_NU&t=146s",
            "description": "Descubre hacia dónde se dirige la programación en la era de la IA"
        },
        {
            "id": 4,
            "title": "¿Por qué es importante estudiar en Academlo?",
            "url": "https://www.youtube.com/watch?v=1OYCVzsMde4",
            "description": "Conoce las ventajas de estudiar en Academlo y cómo puede transformar tu carrera"
        },
        {
            "id": 5,
            "title": "Tutorial de Emergent",
            "url": "https://www.youtube.com/watch?v=joOJZ9ZJEFc",
            "description": "Aprende a usar Emergent para crear proyectos increíbles con IA"
        }
    ]
    return videos

@api_router.get("/content/prompt")
async def get_prompt_content(lang: str = "es"):
    if lang == "en":
        examples = [
            {
                "id": 1,
                "title": "Prompt for Code Generation",
                "prompt": "Act as an expert Python developer. Create a function that receives a list of numbers and returns the average, median and mode. Include error handling and complete documentation.",
                "description": "This prompt is ideal for generating functional and well-documented code"
            },
            {
                "id": 2,
                "title": "Prompt for Data Analysis",
                "prompt": "You are an expert data scientist. Analyze the following sales dataset and provide key insights, important trends and strategic recommendations. Include suggested visualizations and relevant metrics.",
                "description": "Perfect for getting deep business data analysis"
            },
            {
                "id": 3,
                "title": "Prompt for Software Architecture",
                "prompt": "As a senior software architect, design a scalable architecture for an e-commerce application that handles 100k concurrent users. Include design patterns, recommended technologies, component diagram and security considerations.",
                "description": "Ideal for getting robust and professional architectural designs"
            },
            {
                "id": 4,
                "title": "Prompt for Complete Web Development",
                "prompt": "Act as a full-stack web development expert. Create a complete web application using React and Node.js that includes: JWT authentication system, responsive dashboard, user CRUD, MongoDB database integration, documented REST API, and cloud deployment. Provide complete code, file structure and installation guide.",
                "description": "Perfect for generating complete and modern web applications"
            }
        ]
    else:
        examples = [
            {
                "id": 1,
                "title": "Prompt para Generación de Código",
                "prompt": "Actúa como un experto desarrollador Python. Crea una función que reciba una lista de números y devuelva el promedio, la mediana y la moda. Incluye manejo de errores y documentación completa.",
                "description": "Este prompt es ideal para generar código funcional y bien documentado"
            },
            {
                "id": 2,
                "title": "Prompt para Análisis de Datos",
                "prompt": "Eres un científico de datos experto. Analiza el siguiente dataset de ventas y proporciona insights clave, tendencias importantes y recomendaciones estratégicas. Incluye visualizaciones sugeridas y métricas relevantes.",
                "description": "Perfecto para obtener análisis profundos de datos empresariales"
            },
            {
                "id": 3,
                "title": "Prompt para Arquitectura de Software",
                "prompt": "Como arquitecto de software senior, diseña una arquitectura escalable para una aplicación de e-commerce que maneje 100k usuarios concurrentes. Incluye patrones de diseño, tecnologías recomendadas, diagrama de componentes y consideraciones de seguridad.",
                "description": "Ideal para obtener diseños arquitectónicos robustos y profesionales"
            },
            {
                "id": 4,
                "title": "Prompt para Desarrollo Web Completo",
                "prompt": "Actúa como un experto en desarrollo web full-stack. Crea una aplicación web completa usando React y Node.js que incluya: sistema de autenticación JWT, dashboard responsivo, CRUD de usuarios, integración con base de datos MongoDB, API REST documentada, y deployment en la nube. Proporciona el código completo, estructura de archivos y guía de instalación.",
                "description": "Perfecto para generar aplicaciones web completas y modernas"
            }
        ]
    return examples

# Certificate Generation
@api_router.get("/user/certificate")
async def generate_certificate(current_user: User = Depends(get_current_user)):
    progress = await db.user_progress.find_one({"user_id": current_user.id})
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    
    progress_obj = UserProgress(**progress)
    total_progress = (progress_obj.teorico_progress + progress_obj.escucha_progress + 
                     progress_obj.prompt_progress + progress_obj.proyecto_progress) / 4
    
    if total_progress < 100:
        raise HTTPException(status_code=400, detail="Course not completed")
    
    # Generate PDF certificate (simplified version)
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from datetime import datetime
    import io
    
    buffer = io.BytesIO()
    
    # Create the PDF
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Professional Certificate Design with proper centering
    from reportlab.lib.colors import HexColor
    
    # Background gradient effect (simple rectangles with transparency)
    p.setFillColorRGB(0.02, 0.05, 0.1)  # Dark blue background
    p.rect(0, 0, width, height, fill=1)
    
    # Decorative border
    p.setStrokeColor(HexColor('#00d4ff'))
    p.setLineWidth(4)
    p.rect(30, 30, width-60, height-60, fill=0)
    
    # Inner decorative elements
    p.setStrokeColor(HexColor('#0080ff'))
    p.setLineWidth(1)
    for i in range(5):
        p.circle(width - 100, height - 100 - i*20, 50 + i*10, fill=0)
    
    # Header Section
    p.setFillColor(HexColor('#00d4ff'))
    p.setFont("Helvetica-Bold", 48)
    academy_text = "ACADEMY"
    academy_width = p.stringWidth(academy_text, "Helvetica-Bold", 48)
    p.drawString((width - academy_width)/2, height-100, academy_text)
    
    # Subtitle
    p.setFillColorRGB(0.8, 0.8, 0.8)
    p.setFont("Helvetica", 14)
    subtitle = "CERTIFICADO DE COMPLETACIÓN PROFESIONAL"
    subtitle_width = p.stringWidth(subtitle, "Helvetica", 14)
    p.drawString((width - subtitle_width)/2, height-130, subtitle)
    
    # Decorative line under header
    p.setStrokeColor(HexColor('#00d4ff'))
    p.setLineWidth(3)
    p.line(100, height-150, width-100, height-150)
    
    # Main congratulations
    p.setFillColor(HexColor('#00d4ff'))
    p.setFont("Helvetica-Bold", 28)
    congrats = "¡FELICIDADES PROGRAMADOR DEL FUTURO!"
    congrats_width = p.stringWidth(congrats, "Helvetica-Bold", 28)
    p.drawString((width - congrats_width)/2, height-200, congrats)
    
    # Certificate text
    p.setFillColorRGB(0.9, 0.9, 0.9)
    p.setFont("Helvetica", 18)
    cert_text = "Por medio del presente certificamos que"
    cert_width = p.stringWidth(cert_text, "Helvetica", 18)
    p.drawString((width - cert_width)/2, height-250, cert_text)
    
    # Student name with decorative box
    p.setFillColor(HexColor('#00d4ff'))
    p.setFont("Helvetica-Bold", 32)
    name_width = p.stringWidth(current_user.name, "Helvetica-Bold", 32)
    name_x = (width - name_width)/2
    
    # Name background box
    p.setFillColorRGB(0.1, 0.2, 0.3)
    p.roundRect(name_x - 20, height-300, name_width + 40, 45, 5, fill=1)
    
    p.setFillColor(HexColor('#00d4ff'))
    p.drawString(name_x, height-290, current_user.name)
    
    # Achievement description
    p.setFillColorRGB(0.9, 0.9, 0.9)
    p.setFont("Helvetica", 16)
    achievement1 = "ha completado exitosamente el programa de certificación"
    achievement1_width = p.stringWidth(achievement1, "Helvetica", 16)
    p.drawString((width - achievement1_width)/2, height-340, achievement1)
    
    # Course title with emphasis
    p.setFillColor(HexColor('#0080ff'))
    p.setFont("Helvetica-Bold", 22)
    course_title = "DEEP AGENTS & PROGRAMACIÓN CON IA"
    course_width = p.stringWidth(course_title, "Helvetica-Bold", 22)
    p.drawString((width - course_width)/2, height-380, course_title)
    
    # Additional achievement details
    p.setFillColorRGB(0.8, 0.8, 0.8)
    p.setFont("Helvetica", 14)
    details = "Dominando tecnologías emergentes, inteligencia artificial y desarrollo futuro"
    details_width = p.stringWidth(details, "Helvetica", 14)
    p.drawString((width - details_width)/2, height-410, details)
    
    # Date with style
    p.setFillColorRGB(0.7, 0.7, 0.7)
    p.setFont("Helvetica", 14)
    date_str = f"Completado el {datetime.now().strftime('%d de %B de %Y')}"
    date_width = p.stringWidth(date_str, "Helvetica", 14)
    p.drawString((width - date_width)/2, height-450, date_str)
    
    # Logos section with professional layout
    logo_y = 180
    
    # Logo backgrounds
    p.setFillColorRGB(0.15, 0.25, 0.35)
    p.roundRect(80, logo_y-15, 120, 40, 8, fill=1)
    p.roundRect(220, logo_y-15, 120, 40, 8, fill=1)
    p.roundRect(360, logo_y-15, 120, 40, 8, fill=1)
    
    # Logo texts
    p.setFillColor(HexColor('#00d4ff'))
    p.setFont("Helvetica-Bold", 16)
    
    # ACADEMY logo
    academy_logo = "ACADEMY"
    academy_logo_width = p.stringWidth(academy_logo, "Helvetica-Bold", 16)
    p.drawString(140 - academy_logo_width/2, logo_y, academy_logo)
    
    # ACADEMLO logo  
    academlo_logo = "ACADEMLO"
    academlo_logo_width = p.stringWidth(academlo_logo, "Helvetica-Bold", 16)
    p.drawString(280 - academlo_logo_width/2, logo_y, academlo_logo)
    
    # EMERGENT logo
    emergent_logo = "EMERGENT"
    emergent_logo_width = p.stringWidth(emergent_logo, "Helvetica-Bold", 16)
    p.drawString(420 - emergent_logo_width/2, logo_y, emergent_logo)
    
    # Partnership text
    p.setFillColorRGB(0.6, 0.6, 0.6)
    p.setFont("Helvetica", 12)
    partnership = "En asociación estratégica con"
    partnership_width = p.stringWidth(partnership, "Helvetica", 12)
    p.drawString((width - partnership_width)/2, logo_y + 30, partnership)
    
    # Decorative elements around logos
    p.setStrokeColor(HexColor('#00d4ff'))
    p.setLineWidth(2)
    p.line(50, logo_y-30, width-50, logo_y-30)
    
    # Certificate ID and security elements
    p.setFillColorRGB(0.5, 0.5, 0.5)
    p.setFont("Helvetica", 10)
    cert_id = f"ID de Certificado: {current_user.id[:8].upper()}"
    cert_width = p.stringWidth(cert_id, "Helvetica", 10)
    p.drawString((width - cert_width)/2, 120, cert_id)
    
    # Footer with impact
    p.setFillColor(HexColor('#0080ff'))
    p.setFont("Helvetica-Bold", 12)
    footer_main = "QUANTUM INTELLIGENCE • DIGITAL AUTONOMY • AUGMENTED REALITY"
    footer_width = p.stringWidth(footer_main, "Helvetica-Bold", 12)
    p.drawString((width - footer_width)/2, 90, footer_main)
    
    # Final verification text
    p.setFillColorRGB(0.4, 0.4, 0.4)
    p.setFont("Helvetica", 9)
    verification = "Certificado verificable en academy.emergent.sh"
    verification_width = p.stringWidth(verification, "Helvetica", 9)
    p.drawString((width - verification_width)/2, 70, verification)
    
    # Decorative corner elements
    p.setFillColor(HexColor('#00d4ff'))
    p.circle(80, 80, 15, fill=1)
    p.circle(width-80, 80, 15, fill=1)
    p.circle(80, height-80, 15, fill=1)
    p.circle(width-80, height-80, 15, fill=1)
    
    p.save()
    
    # Update certificate generated flag
    await db.user_progress.update_one(
        {"user_id": current_user.id},
        {"$set": {"certificate_generated": True}}
    )
    
    buffer.seek(0)
    
    from fastapi.responses import Response
    
    # Return PDF as bytes
    pdf_bytes = buffer.getvalue()
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=certificado_{current_user.name.replace(' ', '_')}_academy.pdf"
        }
    )

# Enhanced Chat Route - Advanced AI-focused chatbot
@api_router.post("/chat")
async def chat(message: dict):
    user_message = message.get("message", "").lower()
    
    # Detect language (simple detection)
    is_english = any(word in user_message for word in ["what", "how", "why", "where", "when", "which", "who", "can", "do", "does", "is", "are", "will", "would", "should"])
    
    # Advanced AI and ACADEMY knowledge base
    if "que es" in user_message or "qué es" in user_message or "what is" in user_message or "what's" in user_message:
        if "academy" in user_message:
            response = "ACADEMY es la plataforma educativa más avanzada para formar programadores del futuro. Combinamos Deep Agents, IA generativa y tecnologías emergentes. ¡Tu futuro en la programación inteligente comienza aquí!" if not is_english else "ACADEMY is the most advanced educational platform to train future programmers. We combine Deep Agents, generative AI and emerging technologies. Your future in intelligent programming starts here!"
        elif "deep agent" in user_message or "agent" in user_message:
            response = "Un Deep Agent es un sistema de IA revolucionario que combina redes neuronales profundas con agentes autónomos. En ACADEMY te enseñamos a crear estos sistemas que son el futuro de la programación inteligente." if not is_english else "A Deep Agent is a revolutionary AI system that combines deep neural networks with autonomous agents. At ACADEMY we teach you to create these systems that are the future of intelligent programming."
        elif "chatbot" in user_message or "bot" in user_message:
            response = "Un chatbot es un programa de IA que simula conversaciones humanas usando procesamiento de lenguaje natural. En ACADEMY aprendes a crear chatbots avanzados con Deep Agents para el futuro de la interacción digital." if not is_english else "A chatbot is an AI program that simulates human conversations using natural language processing. At ACADEMY you learn to create advanced chatbots with Deep Agents for the future of digital interaction."
        elif "machine learning" in user_message or "ml" in user_message or "aprendizaje" in user_message:
            response = "Machine Learning es el corazón de la IA moderna. Permite a las máquinas aprender sin programación explícita. En ACADEMY te enseñamos ML avanzado aplicado a Deep Agents y desarrollo futuro." if not is_english else "Machine Learning is the heart of modern AI. It allows machines to learn without explicit programming. At ACADEMY we teach you advanced ML applied to Deep Agents and future development."
        elif "inteligencia artificial" in user_message or "artificial intelligence" in user_message or "ai" in user_message:
            response = "La Inteligencia Artificial es la tecnología que está transformando el mundo. En ACADEMY no solo aprendes IA, sino que te conviertes en un arquitecto del futuro digital con Deep Agents y programación inteligente." if not is_english else "Artificial Intelligence is the technology that is transforming the world. At ACADEMY you don't just learn AI, you become an architect of the digital future with Deep Agents and intelligent programming."
        elif "neural" in user_message or "neuronal" in user_message:
            response = "Las redes neuronales son la base de la IA moderna, inspiradas en el cerebro humano. En ACADEMY aprendes a implementar redes neuronales avanzadas para crear Deep Agents revolucionarios." if not is_english else "Neural networks are the foundation of modern AI, inspired by the human brain. At ACADEMY you learn to implement advanced neural networks to create revolutionary Deep Agents."
        elif "prompt" in user_message:
            response = "Un prompt es una instrucción para IA que determina la calidad de la respuesta. En ACADEMY te enseñamos prompt engineering avanzado para maximizar el potencial de los Deep Agents." if not is_english else "A prompt is an instruction for AI that determines the quality of the response. At ACADEMY we teach you advanced prompt engineering to maximize the potential of Deep Agents."
        else:
            response = "¿Qué tema específico de IA y programación futuro te interesa? En ACADEMY cubrimos desde Deep Agents hasta desarrollo con IA. ¡Pregúntame sobre cualquier concepto!" if not is_english else "What specific topic of AI and future programming interests you? At ACADEMY we cover everything from Deep Agents to AI development. Ask me about any concept!"
    
    elif "como" in user_message or "cómo" in user_message or "how" in user_message:
        if "funciona" in user_message or "works" in user_message:
            if "academy" in user_message:
                response = "ACADEMY funciona con un sistema revolucionario de 4 módulos: Teórico (fundamentos IA), Escucha (videos expertos), Prompt (ingeniería avanzada) y Proyecto (Deep Agents reales). ¡25% por módulo hacia tu futuro!" if not is_english else "ACADEMY works with a revolutionary 4-module system: Theoretical (AI fundamentals), Listen (expert videos), Prompt (advanced engineering) and Project (real Deep Agents). 25% per module towards your future!"
            elif "ia" in user_message or "ai" in user_message:
                response = "La IA funciona procesando datos, reconociendo patrones y tomando decisiones. En ACADEMY te enseñamos cómo crear sistemas de IA que evolucionan y aprenden. ¡El futuro de la programación!" if not is_english else "AI works by processing data, recognizing patterns and making decisions. At ACADEMY we teach you how to create AI systems that evolve and learn. The future of programming!"
        elif "empezar" in user_message or "comenzar" in user_message or "start" in user_message:
            response = "¡Empezar en ACADEMY es fácil! 1) Regístrate, 2) Comienza con el módulo Teórico, 3) Avanza por los 4 módulos, 4) Obtén tu certificado como Programador del Futuro. ¡Tu viaje hacia la IA comienza ahora!" if not is_english else "Starting at ACADEMY is easy! 1) Register, 2) Start with the Theoretical module, 3) Advance through the 4 modules, 4) Get your certificate as Future Programmer. Your AI journey starts now!"
        elif "crear" in user_message or "create" in user_message:
            response = "En ACADEMY aprendes a crear Deep Agents, chatbots inteligentes, sistemas de IA y proyectos revolucionarios. Usamos Emergent como plataforma de desarrollo para proyectos reales." if not is_english else "At ACADEMY you learn to create Deep Agents, intelligent chatbots, AI systems and revolutionary projects. We use Emergent as a development platform for real projects."
    
    elif "por que" in user_message or "por qué" in user_message or "why" in user_message:
        if "academy" in user_message:
            response = "ACADEMY es diferente porque combinamos teoría avanzada, práctica real y tecnología emergente. No solo aprendes programación, te conviertes en un arquitecto del futuro digital con Deep Agents." if not is_english else "ACADEMY is different because we combine advanced theory, real practice and emerging technology. You don't just learn programming, you become an architect of the digital future with Deep Agents."
        elif "importante" in user_message or "important" in user_message:
            response = "La IA es importante porque está redefiniendo todo: trabajo, comunicación, creatividad. En ACADEMY te preparamos para liderar esta revolución tecnológica como Programador del Futuro." if not is_english else "AI is important because it's redefining everything: work, communication, creativity. At ACADEMY we prepare you to lead this technological revolution as a Future Programmer."
    
    elif "donde" in user_message or "dónde" in user_message or "where" in user_message:
        response = "ACADEMY está disponible online 24/7. Accede desde cualquier lugar del mundo. También tenemos integración con Emergent y partnerships con Academlo para una experiencia completa." if not is_english else "ACADEMY is available online 24/7. Access from anywhere in the world. We also have integration with Emergent and partnerships with Academlo for a complete experience."
    
    elif "cuando" in user_message or "cuándo" in user_message or "when" in user_message:
        response = "¡El momento perfecto para aprender IA es AHORA! La revolución de los Deep Agents ya comenzó. En ACADEMY puedes empezar inmediatamente y completar a tu ritmo." if not is_english else "The perfect time to learn AI is NOW! The Deep Agents revolution has already begun. At ACADEMY you can start immediately and complete at your own pace."
    
    elif "modulo" in user_message or "módulo" in user_message or "module" in user_message:
        response = "Los 4 módulos de ACADEMY: 1) TEÓRICO: Fundamentos de Deep Agents y IA, 2) ESCUCHA: Videos de expertos y casos reales, 3) PROMPT: Ingeniería avanzada de prompts, 4) PROYECTO: Desarrollo real con Emergent. ¡25% cada uno!" if not is_english else "ACADEMY's 4 modules: 1) THEORETICAL: Deep Agents and AI fundamentals, 2) LISTEN: Expert videos and real cases, 3) PROMPT: Advanced prompt engineering, 4) PROJECT: Real development with Emergent. 25% each!"
    
    elif "certificado" in user_message or "certificate" in user_message:
        response = "El certificado de ACADEMY te acredita como 'PROGRAMADOR DEL FUTURO' 🚀. Incluye logos de Academy, Academlo y Emergent. Es tu pasaporte al mundo de la IA y Deep Agents profesional." if not is_english else "ACADEMY's certificate accredits you as 'FUTURE PROGRAMMER' 🚀. Includes logos from Academy, Academlo and Emergent. It's your passport to the world of professional AI and Deep Agents."
    
    elif "emergent" in user_message:
        response = "Emergent es nuestra plataforma partner para crear proyectos reales de IA. En el módulo 4 de ACADEMY, usarás Emergent para desarrollar tu Deep Agent final. ¡Teoría + Práctica = Futuro!" if not is_english else "Emergent is our partner platform for creating real AI projects. In module 4 of ACADEMY, you'll use Emergent to develop your final Deep Agent. Theory + Practice = Future!"
    
    elif "academlo" in user_message:
        response = "Academlo es nuestro partner educativo que complementa tu formación. Mientras ACADEMY te especializa en Deep Agents e IA, Academlo ofrece formación completa en desarrollo. ¡La combinación perfecta!" if not is_english else "Academlo is our educational partner that complements your training. While ACADEMY specializes you in Deep Agents and AI, Academlo offers complete development training. The perfect combination!"
    
    elif "futuro" in user_message or "future" in user_message:
        response = "El futuro de la programación son los Deep Agents y la IA generativa. En ACADEMY te preparamos para ese futuro HOY. No seas espectador, ¡conviértete en protagonista de la revolución tecnológica!" if not is_english else "The future of programming is Deep Agents and generative AI. At ACADEMY we prepare you for that future TODAY. Don't be a spectator, become the protagonist of the technological revolution!"
    
    elif "ayuda" in user_message or "support" in user_message or "whatsapp" in user_message or "help" in user_message:
        response = "Para cualquier duda, te dejo nuestro número de WhatsApp directo: +528136037100 📱. También puedo resolver tus preguntas sobre IA, Deep Agents, programación y todo lo relacionado con ACADEMY 24/7." if not is_english else "For any questions, here's our direct WhatsApp number: +528136037100 📱. I can also answer your questions about AI, Deep Agents, programming and everything related to ACADEMY 24/7."
    
    elif "precio" in user_message or "cost" in user_message or "price" in user_message:
        response = "¡La inversión en tu futuro como Programador de IA es invaluable! En ACADEMY ofrecemos acceso a conocimiento de vanguardia en Deep Agents. Contacta al +528136037100 para info detallada." if not is_english else "The investment in your future as an AI Programmer is invaluable! At ACADEMY we offer access to cutting-edge knowledge in Deep Agents. Contact +528136037100 for detailed info."
    
    # Error tolerance - handle misspellings and variations
    elif any(word in user_message for word in ["acedemy", "academi", "akademy", "academie"]):
        response = "¿Te refieres a ACADEMY? 😊 Somos la plataforma líder en formación de Programadores del Futuro con Deep Agents e IA. ¡Pregúntame lo que quieras sobre nosotros!" if not is_english else "Do you mean ACADEMY? 😊 We are the leading platform for training Future Programmers with Deep Agents and AI. Ask me anything about us!"
    
    # Default response that always leads to ACADEMY
    else:
        default_responses = [
            "Interesante pregunta. En ACADEMY abordamos todos estos temas desde la perspectiva de Deep Agents y el futuro de la IA. ¿Te gustaría saber cómo aplicamos esto en nuestros módulos?",
            "¡Excelente curiosidad! Este tipo de pensamiento es exactamente lo que desarrollamos en ACADEMY. Como futuros programadores de IA, exploramos estas ideas a fondo.",
            "Esa es una pregunta que todo Programador del Futuro debe hacerse. En ACADEMY te damos las herramientas para entender y crear soluciones con Deep Agents e IA.",
            "¡Me encanta tu curiosidad! En ACADEMY formamos pensadores como tú para liderar la revolución de la IA y Deep Agents. ¿Quieres saber más sobre nuestros módulos?"
        ]
        
        default_responses_en = [
            "Interesting question. At ACADEMY we address all these topics from the perspective of Deep Agents and the future of AI. Would you like to know how we apply this in our modules?",
            "Excellent curiosity! This type of thinking is exactly what we develop at ACADEMY. As future AI programmers, we explore these ideas in depth.",
            "That's a question every Future Programmer should ask. At ACADEMY we give you the tools to understand and create solutions with Deep Agents and AI.",
            "I love your curiosity! At ACADEMY we train thinkers like you to lead the AI and Deep Agents revolution. Do you want to know more about our modules?"
        ]
        
        import random
        response = random.choice(default_responses_en if is_english else default_responses)
        response += " Para más info: WhatsApp +528136037100 📱"
    
    # Save chat message
    chat_msg = ChatMessage(message=user_message, response=response)
    await db.chat_history.insert_one(chat_msg.dict())
    
    return {"response": response}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
