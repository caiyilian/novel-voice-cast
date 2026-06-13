from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Project, Character, Dialogue
from app.core.emotion_labeler import label_emotion
from app.core.ollama_client import OllamaClient

router = APIRouter(prefix="/api/project/{project_id}/emotions", tags=["emotions"])


class EmotionResult(BaseModel):
    dialogue_index: int
    dialogue_text: str
    speaker: str
    emotion: str
    tone: str
    confidence: float
    evidence: str


class EmotionAnalysisResult(BaseModel):
    project_id: int
    results: List[EmotionResult]
    total_dialogues: int
    analyzed: int


class EmotionStatus(BaseModel):
    project_id: int
    total_dialogues: int
    labeled_dialogues: int
    unlabeled_dialogues: int
    average_confidence: float


@router.post("/analyze", response_model=EmotionAnalysisResult)
async def analyze_emotions(
    project_id: int,
    file: UploadFile = File(...),
    limit: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """Analyze emotions for dialogues in the project.

    Requires uploading the novel text file for context analysis.
    Use limit to control how many dialogues to analyze.
    """
    # verify project exists
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # read novel text
    content = await file.read()
    text = content.decode("utf-8")

    # get dialogues
    stmt = select(Dialogue).where(Dialogue.project_id == project_id).order_by(Dialogue.order)
    if limit:
        stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    dialogues = result.scalars().all()

    if not dialogues:
        raise HTTPException(400, "No dialogues found. Upload a novel first.")

    # analyze emotions
    client = OllamaClient()
    results = []
    analyzed = 0

    for i, dia in enumerate(dialogues):
        try:
            emotion_result = label_emotion(
                dialogue_text=dia.text,
                dialogue_line=i + 1,  # approximate line number
                dialogue_index=i,
                text=text,
                client=client,
                max_tool_steps=8,
            )

            # update database
            dia.emotion = emotion_result["emotion"]
            dia.tone = emotion_result["tone"]

            results.append(EmotionResult(
                dialogue_index=i,
                dialogue_text=dia.text[:100],
                speaker=dia.speaker,
                emotion=emotion_result["emotion"],
                tone=emotion_result["tone"],
                confidence=emotion_result["confidence"],
                evidence=emotion_result["evidence"],
            ))

            if emotion_result["confidence"] >= 0.5:
                analyzed += 1

        except Exception as e:
            results.append(EmotionResult(
                dialogue_index=i,
                dialogue_text=dia.text[:100],
                speaker=dia.speaker,
                emotion="calm",
                tone="serious",
                confidence=0.0,
                evidence=f"Error: {str(e)}",
            ))

    await db.commit()

    return EmotionAnalysisResult(
        project_id=project_id,
        results=results,
        total_dialogues=len(dialogues),
        analyzed=analyzed,
    )


@router.get("/status", response_model=EmotionStatus)
async def get_emotion_status(project_id: int, db: AsyncSession = Depends(get_db)):
    """Get emotion labeling status for the project."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # count total dialogues
    total = await db.scalar(
        select(func.count()).select_from(Dialogue).where(Dialogue.project_id == project_id)
    )

    # count labeled dialogues (emotion != "unknown")
    labeled = await db.scalar(
        select(func.count()).select_from(Dialogue)
        .where(Dialogue.project_id == project_id, Dialogue.emotion != "unknown")
    )

    # get average confidence (approximate: count labeled / total)
    avg_confidence = (labeled / total) if total > 0 else 0.0

    return EmotionStatus(
        project_id=project_id,
        total_dialogues=total or 0,
        labeled_dialogues=labeled or 0,
        unlabeled_dialogues=(total or 0) - (labeled or 0),
        average_confidence=avg_confidence,
    )
