import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import UPLOAD_DIR
from app.models import Project, Character, AudioFile, AudioSource
from app.schemas import AudioUploadResult, AudioInfo

router = APIRouter(prefix="/api", tags=["audio"])

ALLOWED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


async def _verify_project(project_id: int, db: AsyncSession):
    """Verify project exists, raise 404 if not."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


async def _verify_character(project_id: int, character_name: str, db: AsyncSession):
    """Verify character exists, raise 404 if not."""
    result = await db.execute(
        select(Character)
        .where(Character.project_id == project_id, Character.name == character_name)
    )
    character = result.scalar_one_or_none()
    if not character:
        raise HTTPException(404, f"Character '{character_name}' not found")
    return character


def _validate_file(file: UploadFile) -> None:
    """Validate file format and size."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}"
        )


async def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to 24kHz 16bit wav using pydub."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise HTTPException(500, f"Audio conversion failed: {str(e)}")


@router.post("/project/{project_id}/audio/upload", response_model=AudioUploadResult)
async def upload_audio(
    project_id: int,
    character_name: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload audio file for a character."""
    project = await _verify_project(project_id, db)
    character = await _verify_character(project_id, character_name, db)

    # validate file
    _validate_file(file)

    # read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")

    # create upload directory
    project_dir = UPLOAD_DIR / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    # save original file
    ext = Path(file.filename).suffix.lower()
    file_id = str(uuid.uuid4())
    original_path = project_dir / f"{file_id}_original{ext}"
    with open(original_path, "wb") as f:
        f.write(content)

    # convert to wav
    wav_path = project_dir / f"{file_id}.wav"
    await _convert_to_wav(str(original_path), str(wav_path))

    # delete original file
    original_path.unlink(missing_ok=True)

    # save to database
    audio = AudioFile(
        file_path=str(wav_path.relative_to(UPLOAD_DIR)),
        source=AudioSource.upload,
        character_id=character.id,
    )
    db.add(audio)
    await db.commit()
    await db.refresh(audio)

    return AudioUploadResult(
        id=audio.id,
        character_name=character_name,
        file_path=audio.file_path,
        source=audio.source.value,
    )


@router.get("/audio/{audio_id}", response_model=AudioInfo)
async def get_audio_info(
    audio_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get audio file info."""
    audio = await db.get(AudioFile, audio_id)
    if not audio:
        raise HTTPException(404, "Audio file not found")

    return AudioInfo(
        id=audio.id,
        file_path=audio.file_path,
        source=audio.source.value,
        character_id=audio.character_id,
        created_at=audio.created_at,
    )


@router.get("/audio/{audio_id}/download")
async def download_audio(
    audio_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Download audio file."""
    audio = await db.get(AudioFile, audio_id)
    if not audio:
        raise HTTPException(404, "Audio file not found")

    file_path = UPLOAD_DIR / audio.file_path
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found on disk")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=f"audio_{audio_id}.wav",
    )


@router.delete("/project/{project_id}/audio/{audio_id}")
async def delete_audio(
    project_id: int,
    audio_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete audio file."""
    await _verify_project(project_id, db)

    audio = await db.get(AudioFile, audio_id)
    if not audio:
        raise HTTPException(404, "Audio file not found")

    # delete file from disk
    file_path = UPLOAD_DIR / audio.file_path
    if file_path.exists():
        file_path.unlink()

    # delete from database
    await db.delete(audio)
    await db.commit()

    return {"status": "deleted", "id": audio_id}
