import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import PRESET_DIR
from app.models import Project, Character, AudioFile

router = APIRouter(prefix="/api/project/{project_id}", tags=["presets"])


class PresetInfo(BaseModel):
    name: str
    filename: str
    file_size: int


class PresetList(BaseModel):
    project_id: int
    presets: List[PresetInfo]
    total: int


class CharacterAudioStatus(BaseModel):
    character_name: str
    gender: str
    dialogue_count: int
    audio_assigned: bool
    audio_source: str  # "upload", "preset_male", "preset_female", "none"
    audio_id: Optional[int] = None


class AudioStatusOverview(BaseModel):
    project_id: int
    characters: List[CharacterAudioStatus]
    total_characters: int
    assigned_count: int


@router.get("/presets", response_model=PresetList)
async def list_presets(project_id: int):
    """List available preset audio files."""
    # verify project exists
    db_session = None
    async for db in get_db():
        db_session = db
        break
    if db_session:
        project = await db_session.get(Project, project_id)
        if not project:
            raise HTTPException(404, "Project not found")

    # scan preset directory
    presets = []
    if PRESET_DIR.exists():
        for f in PRESET_DIR.glob("*.wav"):
            presets.append(PresetInfo(
                name=f.stem,
                filename=f.name,
                file_size=f.stat().st_size,
            ))

    return PresetList(
        project_id=project_id,
        presets=presets,
        total=len(presets),
    )


@router.get("/presets/{preset_name}/download")
async def download_preset(project_id: int, preset_name: str):
    """Download a preset audio file."""
    # verify project exists
    db_session = None
    async for db in get_db():
        db_session = db
        break
    if db_session:
        project = await db_session.get(Project, project_id)
        if not project:
            raise HTTPException(404, "Project not found")

    # find preset file
    preset_path = PRESET_DIR / f"{preset_name}.wav"
    if not preset_path.exists():
        raise HTTPException(404, f"Preset '{preset_name}' not found")

    return FileResponse(
        path=str(preset_path),
        media_type="audio/wav",
        filename=f"{preset_name}.wav",
    )


@router.get("/audio-status", response_model=AudioStatusOverview)
async def get_audio_status(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get audio assignment status for all characters."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # get all characters
    result = await db.execute(
        select(Character).where(Character.project_id == project_id)
    )
    characters = result.scalars().all()

    statuses = []
    assigned_count = 0

    for char in characters:
        # check if character has assigned audio
        audio_result = await db.execute(
            select(AudioFile).where(AudioFile.character_id == char.id).limit(1)
        )
        audio = audio_result.scalar_one_or_none()

        if audio:
            audio_source = "upload"
            audio_id = audio.id
            assigned_count += 1
        else:
            audio_source = "none"
            audio_id = None

        statuses.append(CharacterAudioStatus(
            character_name=char.name,
            gender=char.gender,
            dialogue_count=char.dialogue_count,
            audio_assigned=audio_source != "none",
            audio_source=audio_source,
            audio_id=audio_id,
        ))

    return AudioStatusOverview(
        project_id=project_id,
        characters=statuses,
        total_characters=len(characters),
        assigned_count=assigned_count,
    )
