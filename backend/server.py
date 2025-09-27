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
            # Video 1 = 33%, Video 2 = 33%, Video 3 = 34% = 100%
            video_percentages = {1: 33, 2: 33, 3: 34}
            progress["escucha_progress"] = sum(video_percentages[vid_id] for vid_id in progress["escucha_completed"] if vid_id in video_percentages)
    
    elif update.module == "prompt":
        progress["prompt_completed"] = True
        progress["prompt_progress"] = 100
    
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
async def get_escucha_content():
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
        }
    ]
    return videos

@api_router.get("/content/prompt")
async def get_prompt_content():
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
    
    # Certificate design
    p.setFillColorRGB(0, 0.8, 1)  # Cyan color
    p.setFont("Helvetica-Bold", 36)
    p.drawCentredText(width/2, height-100, "ACADEMY CERTIFICATE")
    
    p.setFillColorRGB(0, 0, 0)
    p.setFont("Helvetica", 24)
    p.drawCentredText(width/2, height-200, "¡Felicidades Programador del Futuro!")
    
    p.setFont("Helvetica", 18)
    p.drawCentredText(width/2, height-280, f"Certificamos que {current_user.name}")
    p.drawCentredText(width/2, height-320, "ha completado exitosamente el curso de")
    
    p.setFillColorRGB(0, 0.8, 1)
    p.setFont("Helvetica-Bold", 20)
    p.drawCentredText(width/2, height-360, "Deep Agents & Programación con IA")
    
    p.setFillColorRGB(0, 0, 0)
    p.setFont("Helvetica", 14)
    p.drawCentredText(width/2, height-420, f"Fecha de completación: {datetime.now().strftime('%d de %B, %Y')}")
    
    # Logos section
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, 100, "ACADEMY")
    p.drawString(200, 100, "ACADEMLO")
    p.drawString(350, 100, "EMERGENT")
    
    # Footer
    p.setFont("Helvetica", 10)
    p.drawCentredText(width/2, 50, "Quantum Intelligence • Digital Autonomy • Augmented Reality")
    
    p.save()
    
    # Update certificate generated flag
    await db.user_progress.update_one(
        {"user_id": current_user.id},
        {"$set": {"certificate_generated": True}}
    )
    
    buffer.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.BytesIO(buffer.getvalue()),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=academy_certificate.pdf"}
    )

# Chat Route (Simple chatbot without external API)
@api_router.post("/chat")
async def chat(message: dict):
    user_message = message.get("message", "").lower()
    
    # Simple rule-based responses about ACADEMY
    if "que es" in user_message or "qué es" in user_message:
        if "academy" in user_message:
            response = "ACADEMY es una plataforma educativa futurista diseñada para formar a los programadores del futuro. Ofrecemos cursos sobre Deep Agents, IA, programación y desarrollo de proyectos con tecnología emergente."
        elif "deep agent" in user_message:
            response = "Un Deep Agent es un sistema de IA avanzado que combina redes neuronales profundas con agentes inteligentes para resolver problemas complejos de manera autónoma."
        elif "modulo" in user_message or "módulo" in user_message:
            response = "Nuestros módulos incluyen: 1) Teórico (conceptos fundamentales), 2) Escucha (videos educativos), 3) Prompt (práctica con IA), y 4) Proyecto (desarrollo práctico con Emergent)."
        else:
            response = "¿Sobre qué tema específico de ACADEMY te gustaría saber más? Puedo ayudarte con información sobre nuestros módulos, certificaciones o el proceso de aprendizaje."
    elif "como" in user_message or "cómo" in user_message:
        if "funciona" in user_message:
            response = "ACADEMY funciona mediante un sistema de progreso por módulos. Completas cada módulo (25% cada uno) hasta alcanzar el 100% y obtener tu certificado como Programador del Futuro."
        elif "empezar" in user_message or "comenzar" in user_message:
            response = "Para empezar, regístrate en la plataforma, completa tu perfil y comienza con el módulo Teórico. Cada módulo te acerca más a convertirte en un experto en desarrollo con IA."
        else:
            response = "Puedo ayudarte con información sobre cómo usar ACADEMY, cómo completar los módulos, o cómo obtener tu certificación."
    elif "certificado" in user_message:
        response = "Al completar los 4 módulos (100% de progreso), obtienes un certificado PDF profesional con los logos de Academy, Academlo y Emergent que te acredita como 'Programador del Futuro'."
    elif "progreso" in user_message:
        response = "Tu progreso se divide en 4 módulos de 25% cada uno: Teórico, Escucha, Prompt y Proyecto. Puedes ver tu avance en tiempo real en tu dashboard personal."
    elif "ayuda" in user_message or "support" in user_message:
        response = "Estoy aquí para ayudarte 24/7 con cualquier duda sobre ACADEMY. También puedes contactar ayuda urgente al WhatsApp +528136037100 o visitar nuestras redes sociales."
    elif "emergent" in user_message:
        response = "Emergent es nuestra plataforma partner donde podrás crear proyectos reales de IA. En el módulo final, usarás Emergent para desarrollar tu proyecto y completar tu certificación."
    else:
        response = "¡Hola! Soy el asistente de ACADEMY 🚀 Estoy aquí para resolver todas tus dudas sobre nuestros cursos de programación e IA. ¿En qué puedo ayudarte hoy?"
    
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
