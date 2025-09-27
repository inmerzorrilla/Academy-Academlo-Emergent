#!/usr/bin/env python3
"""
Script to create the super admin user Inmer Zorrilla for ACADEMY
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

async def create_super_admin():
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    
    # Super admin user data - Inmer Zorrilla
    super_admin_data = {
        "id": str(uuid.uuid4()),
        "name": "Inmer Zorrilla",
        "email": "elchamoinmer@gmail.com",
        "phone": "+528136037100",
        "password": hash_password("admin123"),
        "is_admin": True,
        "is_super_admin": True,  # Special flag for super admin
        "created_at": datetime.now(timezone.utc),
        "last_login": None
    }
    
    # Check if super admin already exists
    existing_admin = await db.users.find_one({"email": "elchamoinmer@gmail.com"})
    if existing_admin:
        print("Super admin user already exists!")
        # Update to ensure super admin flag
        await db.users.update_one(
            {"email": "elchamoinmer@gmail.com"},
            {"$set": {"is_super_admin": True, "is_admin": True}}
        )
        print("✅ Updated super admin privileges!")
        client.close()
        return
    
    # Insert super admin user
    await db.users.insert_one(super_admin_data)
    
    # Create initial progress for super admin
    progress_data = {
        "id": str(uuid.uuid4()),
        "user_id": super_admin_data["id"],
        "teorico_progress": 0,
        "escucha_progress": 0,
        "prompt_progress": 0,
        "proyecto_progress": 0,
        "teorico_completed": [],
        "escucha_completed": [],
        "prompt_completed": False,
        "prompt_tips_completed": False,
        "prompt_examples_completed": [],
        "prompt_practice_completed": False,
        "proyecto_url": None,
        "certificate_generated": False,
        "updated_at": datetime.now(timezone.utc)
    }
    
    await db.user_progress.insert_one(progress_data)
    
    print("✅ Super admin user created successfully!")
    print("Name: Inmer Zorrilla")
    print("Email: elchamoinmer@gmail.com")
    print("Password: admin123")
    print("Super Admin: YES")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(create_super_admin())