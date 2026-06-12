from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATABASE_URL = f"sqlite+aiosqlite:///{DATA_DIR / 'novel_voice_cast.db'}"

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

UPLOAD_DIR = DATA_DIR / "uploads"
PRESET_DIR = DATA_DIR / "presets"
OUTPUT_DIR = DATA_DIR / "output"
