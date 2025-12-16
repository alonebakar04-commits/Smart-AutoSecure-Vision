# config.py
import os
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017"  # fallback
)

DATABASE_NAME = "autosecure_db"
COLLECTION_NAME = "known_persons"