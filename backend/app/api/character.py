from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Project, Character, Dialogue, AudioFile
from app.schemas import CharacterInfo, CharacterList, CharacterUpdate, DialogueInfo, DialogueList
from app.core.gender_identifier import identify_all_genders
from app.core.ollama_client import OllamaClient, OllamaConfig

router = APIRouter(prefix="/api/project/{project_id}/characters", tags=["character"])


async def _verify_project(project_id: int, db: AsyncSession):
    """Verify project exists, raise 404 if not."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@router.get("", response_model=CharacterList)
async def list_characters(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    await _verify_project(project_id, db)

    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id)
        .order_by(Character.dialogue_count.desc())
    )
    characters = result.scalars().all()

    return CharacterList(
        project_id=project_id,
        characters=[CharacterInfo.model_validate(c) for c in characters],
        total_characters=len(characters),
    )


@router.get("/{character_name}", response_model=CharacterInfo)
async def get_character(
    project_id: int,
    character_name: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == character_name)
    )
    character = result.scalar_one_or_none()

    if not character:
        raise HTTPException(404, f"Character '{character_name}' not found")

    return CharacterInfo.model_validate(character)


@router.patch("/{character_name}", response_model=CharacterInfo)
async def update_character(
    project_id: int,
    character_name: str,
    body: CharacterUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == character_name)
    )
    character = result.scalar_one_or_none()

    if not character:
        raise HTTPException(404, f"Character '{character_name}' not found")

    character.gender = body.gender
    await db.commit()
    await db.refresh(character)

    return CharacterInfo.model_validate(character)


@router.get("/{character_name}/dialogues", response_model=DialogueList)
async def list_character_dialogues(
    project_id: int,
    character_name: str,
    limit: Optional[int] = Query(None, ge=1),
    db: AsyncSession = Depends(get_db),
):
    # verify character exists
    char_result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == character_name)
    )
    if not char_result.scalar_one_or_none():
        raise HTTPException(404, f"Character '{character_name}' not found")

    # get dialogues
    stmt = (
        select(Dialogue)
        .where(Dialogue.project_id == project_id, Dialogue.speaker == character_name)
        .order_by(Dialogue.order)
    )
    if limit:
        stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    dialogues = result.scalars().all()

    return DialogueList(
        project_id=project_id,
        character_name=character_name,
        dialogues=[DialogueInfo.model_validate(d) for d in dialogues],
        total_dialogues=len(dialogues),
    )


class GenderResult(BaseModel):
    character_name: str
    gender: str
    confidence: float
    evidence: str


class GenderIdentificationResult(BaseModel):
    project_id: int
    results: List[GenderResult]
    total_characters: int
    identified: int


@router.post("/identify-genders", response_model=GenderIdentificationResult)
async def identify_genders(
    project_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Identify gender for all characters in the project.

    Requires uploading the novel text file for context analysis.
    """
    project = await _verify_project(project_id, db)

    # read novel text
    content = await file.read()
    text = content.decode("utf-8")

    # get all characters
    result = await db.execute(
        select(Character).where(Character.project_id == project_id)
    )
    characters = result.scalars().all()

    if not characters:
        raise HTTPException(400, "No characters found. Upload a novel first.")

    # identify genders
    client = OllamaClient()
    character_names = [c.name for c in characters]
    gender_results = identify_all_genders(character_names, text, client, max_tool_steps=8)

    # update database
    identified = 0
    for gr in gender_results:
        for char in characters:
            if char.name == gr["character_name"]:
                char.gender = gr["gender"]
                if gr["confidence"] >= 0.5:
                    identified += 1
                break

    await db.commit()

    return GenderIdentificationResult(
        project_id=project_id,
        results=[GenderResult(**gr) for gr in gender_results],
        total_characters=len(characters),
        identified=identified,
    )


# ─── Audio Assignment ──────────────────────────────────────────────

class AudioAssignRequest(BaseModel):
    source: str  # "upload" or "preset_male" or "preset_female"
    audio_id: Optional[int] = None  # required if source == "upload"


class AudioAssignResponse(BaseModel):
    character_name: str
    voice_source: str
    voice_audio_id: Optional[int] = None


@router.post("/{character_name}/assign", response_model=AudioAssignResponse)
async def assign_audio(
    project_id: int,
    character_name: str,
    body: AudioAssignRequest,
    db: AsyncSession = Depends(get_db),
):
    """Assign audio source to a character."""
    # verify character exists
    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == character_name)
    )
    character = result.scalar_one_or_none()
    if not character:
        raise HTTPException(404, f"Character '{character_name}' not found")

    # handle different source types
    if body.source == "upload":
        if not body.audio_id:
            raise HTTPException(400, "audio_id required for upload source")
        # verify audio exists and belongs to this project
        audio = await db.get(AudioFile, body.audio_id)
        if not audio:
            raise HTTPException(404, "Audio file not found")
        # update audio's character_id
        audio.character_id = character.id
        audio.source = "upload"
        await db.commit()
        return AudioAssignResponse(
            character_name=character_name,
            voice_source="upload",
            voice_audio_id=audio.id,
        )

    elif body.source in ("preset_male", "preset_female"):
        # for preset, just record the source type
        # (actual preset audio will be handled during synthesis)
        character._voice_source = body.source
        await db.commit()
        return AudioAssignResponse(
            character_name=character_name,
            voice_source=body.source,
            voice_audio_id=None,
        )

    else:
        raise HTTPException(400, f"Invalid source: {body.source}")


@router.delete("/{character_name}/assign")
async def unassign_audio(
    project_id: int,
    character_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Remove audio assignment from a character."""
    # verify character exists
    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == character_name)
    )
    character = result.scalar_one_or_none()
    if not character:
        raise HTTPException(404, f"Character '{character_name}' not found")

    # find and unassign any audio files linked to this character
    audio_result = await db.execute(
        select(AudioFile).where(AudioFile.character_id == character.id)
    )
    audio_files = audio_result.scalars().all()

    for audio in audio_files:
        audio.character_id = None

    # clear preset source if any
    character._voice_source = None

    await db.commit()

    return {"status": "unassigned", "character_name": character_name}
