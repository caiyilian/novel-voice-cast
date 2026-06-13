from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, UPLOAD_DIR, PRESET_DIR
from app.database import engine
from app.models import Base
from app.api.project import router as project_router
from app.api.character import router as character_router
from app.api.audio import router as audio_router
from app.api.presets import router as presets_router
from app.api.emotions import router as emotions_router
from app.api.preview import router as preview_router
from app.api.ws import router as ws_router
from app.api.synthesis import router as synthesis_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Novel Voice Cast", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project_router)
app.include_router(character_router)
app.include_router(audio_router)
app.include_router(presets_router)
app.include_router(emotions_router)
app.include_router(preview_router)
app.include_router(ws_router)
app.include_router(synthesis_router)

app.mount("/audio", StaticFiles(directory=str(UPLOAD_DIR)), name="audio")
app.mount("/presets", StaticFiles(directory=str(PRESET_DIR)), name="presets")


@app.get("/health")
async def health():
    return {"status": "ok"}
