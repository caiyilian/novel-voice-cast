from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Project, Character, Dialogue
from app.schemas import CharacterInfo, CharacterList, CharacterUpdate, DialogueInfo, DialogueList

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
