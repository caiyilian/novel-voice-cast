import hashlib
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import UPLOAD_DIR
from app.models import Project, Character, AudioFile
from app.tts.manager import TTSProviderManager
from app.tts.preset import EdgeTTSProvider

router = APIRouter(prefix="/api/project", tags=["preview"])

# Global TTS manager (initialized on first use)
_tts_manager: Optional[TTSProviderManager] = None


def _get_tts_manager() -> TTSProviderManager:
    """Get or initialize the TTS manager."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSProviderManager()
        # register edge-tts as fallback
        _tts_manager.register(EdgeTTSProvider())
    return _tts_manager


# Simple in-memory cache
_preview_cache = {}
CACHE_MAX_SIZE = 100


def _get_cache_key(text: str, voice_id: str) -> str:
    """Generate cache key for preview."""
    return hashlib.md5(f"{text}:{voice_id}".encode()).hexdigest()


class PreviewRequest(BaseModel):
    text: str
    character_name: str


@router.post("/{project_id}/preview")
async def preview_audio(
    project_id: int,
    body: PreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Preview audio for a character with given text.

    Returns audio bytes (MP3) that can be played directly in the frontend.
    """
    # verify project exists
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # verify character exists
    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == body.character_name)
    )
    character = result.scalar_one_or_none()
    if not character:
        raise HTTPException(404, f"Character '{body.character_name}' not found")

    # get character's audio assignment
    audio_result = await db.execute(
        select(AudioFile).where(AudioFile.character_id == character.id).limit(1)
    )
    audio = audio_result.scalar_one_or_none()

    # determine voice_id
    if audio:
        # character has uploaded audio - for now use default male voice
        # TODO: implement VoxCPM for cloned voices
        voice_id = "zh-CN-YunxiNeural"
    else:
        # no audio assigned - use gender-based default
        if character.gender == "female":
            voice_id = "zh-CN-XiaoxiaoNeural"
        else:
            voice_id = "zh-CN-YunxiNeural"

    # check cache
    cache_key = _get_cache_key(body.text, voice_id)
    if cache_key in _preview_cache:
        return Response(
            content=_preview_cache[cache_key],
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"inline; filename=preview_{body.character_name}.mp3"},
        )

    # synthesize
    manager = _get_tts_manager()
    try:
        audio_bytes = await manager.synthesize(body.text, voice_id)
    except Exception as e:
        raise HTTPException(500, f"TTS synthesis failed: {str(e)}")

    # cache result
    if len(_preview_cache) >= CACHE_MAX_SIZE:
        # remove oldest entry
        _preview_cache.pop(next(iter(_preview_cache)))
    _preview_cache[cache_key] = audio_bytes

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename=preview_{body.character_name}.mp3"},
    )
