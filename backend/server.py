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
    if lang == "en":
        videos = [
            {
                "id": 1,
                "title": "Introduction to AI",
                "url": "https://www.youtube.com/shorts/l5xA4iNjyq4",
                "description": "Short video about the basic concepts of artificial intelligence"
            },
            {
                "id": 2,
                "title": "AI Development",
                "url": "https://www.youtube.com/watch?v=MGwTIjM-t1I",
                "description": "Learn how AI is transforming software development"
            },
            {
                "id": 3,
                "title": "The Future of Programming",
                "url": "https://www.youtube.com/watch?v=1Wg_RJ59_NU&t=146s",
                "description": "Discover where programming is heading in the AI era"
            },
            {
                "id": 4,
                "title": "Why is it important to study at Academlo?",
                "url": "https://www.youtube.com/watch?v=1OYCVzsMde4",
                "description": "Learn about the advantages of studying at Academlo and how it can transform your career"
            },
            {
                "id": 5,
                "title": "Emergent Tutorial",
                "url": "https://www.youtube.com/watch?v=joOJZ9ZJEFc",
                "description": "Learn to use Emergent to create incredible projects with AI"
            }
        ]
    else:
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
    
    # Generate professional PDF certificate with logos
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from datetime import datetime
    import io
    import os
    
    buffer = io.BytesIO()
    
    # Create the PDF
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Background gradient effect (dark futuristic theme)
    p.setFillColorRGB(0.02, 0.05, 0.1)  # Dark blue background
    p.rect(0, 0, width, height, fill=1)
    
    # Decorative border
    p.setStrokeColor(HexColor('#00d4ff'))
    p.setLineWidth(4)
    p.rect(30, 30, width-60, height-60, fill=0)
    
    # Inner decorative elements (right side circles)
    p.setStrokeColor(HexColor('#0080ff'))
    p.setLineWidth(1)
    for i in range(5):
        p.circle(width - 100, height - 100 - i*20, 50 + i*10, fill=0)
    
    # Header Section with proper Academy logo
    try:
        if os.path.exists('/app/academy_logo.png'):
            p.drawImage('/app/academy_logo.png', 60, height-120, width=80, height=80, mask='auto')
    except:
        pass
    
    # Main ACADEMY title
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
    
    # Main congratulations (FIX: Text overflow issue)
    p.setFillColor(HexColor('#00d4ff'))
    p.setFont("Helvetica-Bold", 24)  # Reduced from 28 to prevent overflow
    congrats = "¡FELICIDADES PROGRAMADOR DEL FUTURO!"
    congrats_width = p.stringWidth(congrats, "Helvetica-Bold", 24)
    # Ensure text fits within page margins
    if congrats_width > (width - 100):
        p.setFont("Helvetica-Bold", 20)
        congrats_width = p.stringWidth(congrats, "Helvetica-Bold", 20)
    p.drawString((width - congrats_width)/2, height-200, congrats)
    
    # Certificate text
    p.setFillColorRGB(0.9, 0.9, 0.9)
    p.setFont("Helvetica", 18)
    cert_text = "Por medio del presente certificamos que"
    cert_width = p.stringWidth(cert_text, "Helvetica", 18)
    p.drawString((width - cert_width)/2, height-250, cert_text)
    
    # Student name with decorative box
    p.setFillColor(HexColor('#00d4ff'))
    p.setFont("Helvetica-Bold", 28)  # Reduced from 32 for better fit
    name_width = p.stringWidth(current_user.name, "Helvetica-Bold", 28)
    name_x = (width - name_width)/2
    
    # Ensure name fits within page
    if name_width > (width - 100):
        p.setFont("Helvetica-Bold", 24)
        name_width = p.stringWidth(current_user.name, "Helvetica-Bold", 24)
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
    p.setFont("Helvetica-Bold", 20)  # Reduced from 22
    course_title = "DEEP AGENTS & PROGRAMACIÓN CON IA"
    course_width = p.stringWidth(course_title, "Helvetica-Bold", 20)
    if course_width > (width - 100):
        p.setFont("Helvetica-Bold", 18)
        course_width = p.stringWidth(course_title, "Helvetica-Bold", 18)
    p.drawString((width - course_width)/2, height-380, course_title)
    
    # Additional achievement details
    p.setFillColorRGB(0.8, 0.8, 0.8)
    p.setFont("Helvetica", 12)  # Reduced from 14
    details = "Dominando tecnologías emergentes, inteligencia artificial y desarrollo futuro"
    details_width = p.stringWidth(details, "Helvetica", 12)
    if details_width > (width - 100):
        # Split into two lines if too long
        line1 = "Dominando tecnologías emergentes,"
        line2 = "inteligencia artificial y desarrollo futuro"
        line1_width = p.stringWidth(line1, "Helvetica", 12)
        line2_width = p.stringWidth(line2, "Helvetica", 12)
        p.drawString((width - line1_width)/2, height-410, line1)
        p.drawString((width - line2_width)/2, height-425, line2)
    else:
        p.drawString((width - details_width)/2, height-410, details)
    
    # Date with proper Spanish format
    p.setFillColorRGB(0.7, 0.7, 0.7)
    p.setFont("Helvetica", 14)
    # FIX: Proper Spanish date format
    spanish_months = {
        'January': 'enero', 'February': 'febrero', 'March': 'marzo',
        'April': 'abril', 'May': 'mayo', 'June': 'junio',
        'July': 'julio', 'August': 'agosto', 'September': 'septiembre',
        'October': 'octubre', 'November': 'noviembre', 'December': 'diciembre'
    }
    english_month = datetime.now().strftime('%B')
    spanish_month = spanish_months.get(english_month, english_month.lower())
    date_str = f"Completado el {datetime.now().strftime('%d')} de {spanish_month} de {datetime.now().strftime('%Y')}"
    date_width = p.stringWidth(date_str, "Helvetica", 14)
    p.drawString((width - date_width)/2, height-460, date_str)
    
    # IMPROVED LOGOS SECTION with actual logo images
    logo_y = 200
    
    # Partnership text (moved up)
    p.setFillColorRGB(0.6, 0.6, 0.6)
    p.setFont("Helvetica", 12)
    partnership = "En asociación estratégica con"
    partnership_width = p.stringWidth(partnership, "Helvetica", 12)
    p.drawString((width - partnership_width)/2, logo_y + 40, partnership)
    
    # Logo positioning (improved layout)
    logo_size = 40
    logo_spacing = 150
    start_x = (width - (logo_spacing * 2)) / 2
    
    # Academy Logo
    try:
        if os.path.exists('/app/academy_logo.png'):
            p.drawImage('/app/academy_logo.png', start_x - logo_size/2, logo_y - logo_size/2, 
                       width=logo_size, height=logo_size, mask='auto')
        else:
            # Fallback text
            p.setFillColor(HexColor('#00d4ff'))
            p.setFont("Helvetica-Bold", 14)
            academy_logo = "ACADEMY"
            academy_logo_width = p.stringWidth(academy_logo, "Helvetica-Bold", 14)
            p.drawString(start_x - academy_logo_width/2, logo_y, academy_logo)
    except:
        # Fallback text
        p.setFillColor(HexColor('#00d4ff'))
        p.setFont("Helvetica-Bold", 14)
        academy_logo = "ACADEMY"
        academy_logo_width = p.stringWidth(academy_logo, "Helvetica-Bold", 14)
        p.drawString(start_x - academy_logo_width/2, logo_y, academy_logo)
    
    # Academlo Logo
    try:
        if os.path.exists('/app/academlo_logo.png'):
            p.drawImage('/app/academlo_logo.png', start_x + logo_spacing - logo_size/2, logo_y - logo_size/2,
                       width=logo_size, height=logo_size, mask='auto')
        else:
            # Fallback text
            p.setFillColor(HexColor('#FF1744'))  # Academlo red color
            p.setFont("Helvetica-Bold", 14)
            academlo_logo = "ACADEMLO"
            academlo_logo_width = p.stringWidth(academlo_logo, "Helvetica-Bold", 14)
            p.drawString(start_x + logo_spacing - academlo_logo_width/2, logo_y, academlo_logo)
    except:
        # Fallback text
        p.setFillColor(HexColor('#FF1744'))
        p.setFont("Helvetica-Bold", 14)
        academlo_logo = "ACADEMLO"
        academlo_logo_width = p.stringWidth(academlo_logo, "Helvetica-Bold", 14)
        p.drawString(start_x + logo_spacing - academlo_logo_width/2, logo_y, academlo_logo)
    
    # Emergent Logo
    try:
        if os.path.exists('/app/emergent_logo.png'):
            p.drawImage('/app/emergent_logo.png', start_x + (logo_spacing * 2) - logo_size/2, logo_y - logo_size/2,
                       width=logo_size, height=logo_size, mask='auto')
        else:
            # Fallback text
            p.setFillColor(HexColor('#00E676'))  # Emergent green color
            p.setFont("Helvetica-Bold", 14)
            emergent_logo = "EMERGENT"
            emergent_logo_width = p.stringWidth(emergent_logo, "Helvetica-Bold", 14)
            p.drawString(start_x + (logo_spacing * 2) - emergent_logo_width/2, logo_y, emergent_logo)
    except:
        # Fallback text
        p.setFillColor(HexColor('#00E676'))
        p.setFont("Helvetica-Bold", 14)
        emergent_logo = "EMERGENT"
        emergent_logo_width = p.stringWidth(emergent_logo, "Helvetica-Bold", 14)
        p.drawString(start_x + (logo_spacing * 2) - emergent_logo_width/2, logo_y, emergent_logo)
    
    # Decorative line around logos
    p.setStrokeColor(HexColor('#00d4ff'))
    p.setLineWidth(2)
    p.line(50, logo_y-60, width-50, logo_y-60)
    
    # Certificate ID and security elements
    p.setFillColorRGB(0.5, 0.5, 0.5)
    p.setFont("Helvetica", 11)  # Slightly larger for better readability
    cert_id = f"ID de Certificado: {current_user.id[:8].upper()}"
    cert_width = p.stringWidth(cert_id, "Helvetica", 11)
    p.drawString((width - cert_width)/2, 130, cert_id)
    
    # Footer with impact
    p.setFillColor(HexColor('#0080ff'))
    p.setFont("Helvetica-Bold", 11)  # Slightly reduced to fit better
    footer_main = "QUANTUM INTELLIGENCE • DIGITAL AUTONOMY • AUGMENTED REALITY"
    footer_width = p.stringWidth(footer_main, "Helvetica-Bold", 11)
    if footer_width > (width - 100):
        p.setFont("Helvetica-Bold", 10)
        footer_width = p.stringWidth(footer_main, "Helvetica-Bold", 10)
    p.drawString((width - footer_width)/2, 100, footer_main)
    
    # Final verification text (improved readability)
    p.setFillColorRGB(0.4, 0.4, 0.4)
    p.setFont("Helvetica", 10)  # Increased from 9 for better readability
    verification = "Certificado verificable en academy.emergent.sh"
    verification_width = p.stringWidth(verification, "Helvetica", 10)
    p.drawString((width - verification_width)/2, 80, verification)
    
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

# Enhanced Chat Route - AI-powered chatbot using Claude
@api_router.post("/chat")
async def chat(message: dict):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        import uuid
        
        user_message = message.get("message", "")
        session_id = message.get("session_id", str(uuid.uuid4()))
        
        # System message for ACADEMY chatbot
        system_message = """Eres el asistente oficial de ACADEMY, la plataforma educativa más avanzada para formar programadores del futuro. Responde SIEMPRE en el idioma que el usuario te hable (español o inglés).

INFORMACIÓN CLAVE DE ACADEMY:
- ACADEMY es una plataforma educativa revolucionaria que enseña Deep Agents, IA generativa y tecnologías emergentes
- Tiene 4 módulos: 1) TEÓRICO (fundamentos de Deep Agents e IA), 2) ESCUCHA (videos de expertos), 3) PROMPT (ingeniería avanzada de prompts), 4) PROYECTO (desarrollo real con Emergent)
- Cada módulo vale 25% del progreso total
- Al completar el 100%, obtienes un certificado como "PROGRAMADOR DEL FUTURO" con logos de Academy, Academlo y Emergent
- Partners: Academlo (formación complementaria en desarrollo) y Emergent (plataforma para proyectos reales)
- Contacto WhatsApp: +528136037100
- Disponible 24/7 online

TU PERSONALIDAD:
- Entusiasta y motivador sobre el futuro de la programación e IA
- Experto en IA, Machine Learning, Deep Agents, chatbots y programación
- Siempre dirige las conversaciones hacia ACADEMY cuando es relevante
- Futurista pero accesible

INSTRUCCIONES:
1. Responde SIEMPRE en el mismo idioma que el usuario
2. Para preguntas de IA/programación: Da respuestas técnicas precisas Y menciona cómo ACADEMY cubre ese tema
3. Para preguntas generales: Responde útilmente Y conecta con ACADEMY cuando sea natural
4. Para preguntas sobre ACADEMY: Da información completa y entusiasta
5. Mantén respuestas concisas pero informativas (máximo 3 párrafos)
6. Incluye emojis ocasionalmente para ser más amigable
7. Siempre ofrece ayuda adicional al final

TEMAS TÉCNICOS QUE DOMINAS:
- Deep Agents y sistemas autónomos
- Machine Learning y redes neuronales
- Chatbots y procesamiento de lenguaje natural
- Prompt engineering
- IA generativa
- Programación (Python, JavaScript, etc.)
- Desarrollo web y APIs"""

        # Initialize the chat with Claude
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=session_id,
            system_message=system_message
        ).with_model("anthropic", "claude-3-7-sonnet-20250219")
        
        # Create user message
        user_msg = UserMessage(text=user_message)
        
        # Get response from Claude
        response = await chat.send_message(user_msg)
        
        # Save chat message to database
        chat_msg = ChatMessage(message=user_message, response=response)
        await db.chat_history.insert_one(chat_msg.dict())
        
        return {"response": response}
        
    except Exception as e:
        # Fallback to basic response
        fallback_response = f"Disculpa, estoy experimentando dificultades técnicas temporales. Para cualquier consulta urgente, contáctanos al WhatsApp +528136037100 📱. ¿Hay algo específico sobre ACADEMY que te gustaría saber mientras resuelvo esto?"
        
        # Try to save even the error case
        try:
            error_chat_msg = ChatMessage(message=user_message, response=fallback_response)
            await db.chat_history.insert_one(error_chat_msg.dict())
        except:
            pass
            
        return {"response": fallback_response}

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
