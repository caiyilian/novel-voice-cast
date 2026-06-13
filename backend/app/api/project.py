from typing import List, Tuple, Optional

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Project, Character, Dialogue, ProjectStatus
from app.schemas import (
    ProjectCreate, ProjectResponse, ProjectOverview,
    UploadResult, ChapterInfo, ChapterList,
)
from app.core.parser import parse, extract_chapters
from app.core.chapter_extractor import llm_extract_chapters
from app.core.ollama_client import OllamaClient, OllamaConfig

router = APIRouter(prefix="/api/project", tags=["project"])


def _try_get_ollama_client():
    """Try to create Ollama client, return None if unavailable."""
    try:
        config = OllamaConfig.from_ip_config()
        if not config.base_url:
            return None
        client = OllamaClient(config)
        status = client.check_connection()
        return client if status.ok else None
    except Exception:
        return None


def _extract_chapters_with_llm(text: str, regex_chapters: list) -> Tuple[list, str]:
    """Extract chapters using LLM with regex fallback.

    Returns (chapters, method) where method is "llm" or "regex".
    """
    client = _try_get_ollama_client()
    if client:
        try:
            llm_chapters = llm_extract_chapters(
                text=text,
                client=client,
                max_tool_steps=12,
                regex_candidates=regex_chapters,
                max_retries=2,
            )
            if llm_chapters:
                return llm_chapters, "llm"
        except Exception:
            pass
    return regex_chapters, "regex"


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


@router.get("/{project_id}/chapters", response_model=ChapterList)
async def get_chapters(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # get all dialogues grouped by chapter
    result = await db.execute(
        select(Dialogue.chapter, func.count(Dialogue.id))
        .where(Dialogue.project_id == project_id, Dialogue.chapter != "")
        .group_by(Dialogue.chapter)
        .order_by(Dialogue.chapter)
    )

    chapters = []
    for chapter_title, count in result.all():
        chapters.append(ChapterInfo(
            title=chapter_title,
            line_number=0,  # not stored in DB
            dialogue_count=count,
        ))

    return ChapterList(
        project_id=project_id,
        chapters=chapters,
        total_chapters=len(chapters),
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

    # parse dialogues
    dialogues, character_names = parse(text)
    if not dialogues:
        raise HTTPException(400, "No dialogues found in the uploaded file")

    # extract chapters (LLM with regex fallback)
    regex_chapters = extract_chapters(text)
    llm_chapters, chapter_method = _extract_chapters_with_llm(text, regex_chapters)

    # build chapter lookup: line_number -> title
    chapter_map = {}
    for ch in llm_chapters:
        chapter_map[ch["line_number"]] = ch["title"]

    # assign chapters to dialogues based on line proximity
    def find_chapter(dialogue_order: int, total_dialogues: int) -> str:
        """Find chapter for a dialogue based on its position."""
        if not chapter_map:
            return ""
        # estimate line number from dialogue order
        estimated_line = int(dialogue_order * len(text.splitlines()) / total_dialogues)
        # find closest chapter
        best_chapter = ""
        best_distance = float("inf")
        for ln, title in chapter_map.items():
            distance = abs(ln - estimated_line)
            if distance < best_distance:
                best_distance = distance
                best_chapter = title
        return best_chapter

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

    # write dialogues with chapter info
    chapters_found = set()
    for order, d in enumerate(dialogues, start=1):
        chapter = find_chapter(order, len(dialogues))
        if chapter:
            chapters_found.add(chapter)
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
        chapters=len(chapters_found),
        chapter_method=chapter_method,
    )
