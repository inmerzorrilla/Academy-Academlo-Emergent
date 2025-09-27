#!/usr/bin/env python3
"""
Script to create an admin user for ACADEMY
"""
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
db_name = os.environ['DB_NAME']

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

async def create_admin_user():
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    
    # Admin user data
    admin_data = {
        "id": str(uuid.uuid4()),
        "name": "Admin ACADEMY",
        "email": "admin@academy.com",
        "phone": "+528136037100",
        "password": hash_password("admin123"),
        "is_admin": True,
        "created_at": datetime.now(timezone.utc),
        "last_login": None
    }
    
    # Check if admin already exists
    existing_admin = await db.users.find_one({"email": "admin@academy.com"})
    if existing_admin:
        print("Admin user already exists!")
        return
    
    # Insert admin user
    await db.users.insert_one(admin_data)
    
    # Create initial progress for admin
    progress_data = {
        "id": str(uuid.uuid4()),
        "user_id": admin_data["id"],
        "teorico_progress": 0,
        "escucha_progress": 0,
        "prompt_progress": 0,
        "proyecto_progress": 0,
        "teorico_completed": [],
        "escucha_completed": [],
        "prompt_completed": False,
        "proyecto_url": None,
        "certificate_generated": False,
        "updated_at": datetime.now(timezone.utc)
    }
    
    await db.user_progress.insert_one(progress_data)
    
    print("✅ Admin user created successfully!")
    print("Email: admin@academy.com")
    print("Password: admin123")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(create_admin_user())