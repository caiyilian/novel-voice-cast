from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Project, Character, Dialogue, ProjectStatus
from app.schemas import ProjectCreate, ProjectResponse, ProjectOverview, UploadResult
from app.core.parser import parse

router = APIRouter(prefix="/api/project", tags=["project"])


@router.post("", response_model=ProjectResponse)
async def create_project(body: ProjectCreate, db: AsyncSession = Depends(get_db)):
    project = Project(name=body.name, status=ProjectStatus.created)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("/{project_id}", response_model=ProjectOverview)
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    char_count = await db.scalar(
        select(func.count()).select_from(Character).where(Character.project_id == project_id)
    )
    dia_count = await db.scalar(
        select(func.count()).select_from(Dialogue).where(Dialogue.project_id == project_id)
    )
    chapter_rows = await db.execute(
        select(Dialogue.chapter).where(Dialogue.project_id == project_id, Dialogue.chapter != "").distinct()
    )
    chapters = len(chapter_rows.all())

    return ProjectOverview(
        id=project.id,
        name=project.name,
        status=project.status.value,
        character_count=char_count or 0,
        dialogue_count=dia_count or 0,
        chapter_count=chapters,
    )


@router.post("/{project_id}/upload", response_model=UploadResult)
async def upload_novel(
    project_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    content = await file.read()
    text = content.decode("utf-8")

    dialogues, character_names = parse(text)
    if not dialogues:
        raise HTTPException(400, "No dialogues found in the uploaded file")

    # write characters
    char_map = {}
    for name in character_names:
        c = Character(
            name=name,
            project_id=project_id,
            dialogue_count=sum(1 for d in dialogues if d.get("speaker") == name),
        )
        db.add(c)
        await db.flush()
        char_map[name] = c

    # write dialogues
    chapters = set()
    for order, d in enumerate(dialogues, start=1):
        chapter = d.get("chapter", "")
        if chapter:
            chapters.add(chapter)
        dia = Dialogue(
            chapter=chapter,
            text=d["text"],
            speaker=d.get("speaker", ""),
            order=order,
            project_id=project_id,
        )
        db.add(dia)

    project.status = ProjectStatus.parsed
    await db.commit()

    return UploadResult(
        project_id=project_id,
        characters=len(character_names),
        dialogues=len(dialogues),
        chapters=len(chapters),
    )
