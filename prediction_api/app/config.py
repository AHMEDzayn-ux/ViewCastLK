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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

# Keep the primary model separately for backwards-compatible deployments. The
# fallback list is ordered for the free-tier quotas currently available to this
# project: high-throughput Flash-Lite models first, followed by capable Flash
# models with smaller daily allowances. A semicolon separator is Cloud Run
# friendly because commas delimit separate --set-env-vars entries in gcloud.
_DEFAULT_GEMINI_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite;"
    "gemini-3.7-flash;"
    "gemini-3.6-flash;"
    "gemini-3.5-flash;"
    "gemini-3-flash-preview;"
    "gemini-2.5-flash;"
    "gemini-2.5-flash-lite"
)
_gemini_fallback_models_raw = os.getenv(
    "GEMINI_FALLBACK_MODELS", _DEFAULT_GEMINI_FALLBACK_MODELS
).strip()
GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in _gemini_fallback_models_raw.replace(",", ";").split(";")
    if model.strip()
)
