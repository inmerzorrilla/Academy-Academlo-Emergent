#!/usr/bin/env python3
"""
Script to complete admin progress for testing certificate
"""
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
db_name = os.environ['DB_NAME']

async def complete_admin_progress():
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    
    # Find admin user
    admin_user = await db.users.find_one({"email": "admin@academy.com"})
    if not admin_user:
        print("Admin user not found!")
        return
    
    # Update progress to 100% complete
    progress_update = {
        "teorico_progress": 100,
        "escucha_progress": 100,
        "prompt_progress": 100,
        "proyecto_progress": 100,
        "teorico_completed": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "escucha_completed": [1, 2, 3, 4, 5],
        "prompt_completed": True,
        "prompt_tips_completed": True,
        "prompt_examples_completed": [1, 2, 3, 4],
        "prompt_practice_completed": True,
        "proyecto_url": "https://test-project.emergent.sh/",
        "certificate_generated": False,
        "updated_at": datetime.now(timezone.utc)
    }
    
    result = await db.user_progress.update_one(
        {"user_id": admin_user["id"]},
        {"$set": progress_update},
        upsert=True
    )
    
    print("✅ Admin progress completed!")
    print("- Teorico: 100%")
    print("- Escucha: 100%")
    print("- Prompt: 100%")
    print("- Proyecto: 100%")
    print("- Total: 100% - Ready for certificate!")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(complete_admin_progress())