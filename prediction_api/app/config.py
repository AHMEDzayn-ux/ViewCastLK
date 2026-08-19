import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
DASHBOARD_ORIGIN_RAW = os.getenv("DASHBOARD_ORIGIN", "http://localhost:3000").strip()
ALLOWED_ORIGINS = [
    origin.strip() for origin in DASHBOARD_ORIGIN_RAW.split(",") if origin.strip()
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000"]
DASHBOARD_ORIGIN = ALLOWED_ORIGINS[0]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
