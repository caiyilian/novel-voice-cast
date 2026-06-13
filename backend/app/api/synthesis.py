"""
合成 & 输出 API — 启动全卷合成、查询状态、下载音频。

端点：
- POST /api/project/{id}/synthesize — 启动全卷合成
- GET /api/project/{id}/synthesis/status — 查询合成状态
- DELETE /api/project/{id}/synthesis — 取消合成
- GET /api/project/{id}/output — 下载整卷合成音频
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, async_session
from app.config import OUTPUT_DIR
from app.models import Project, Character, ProjectStatus
from app.core.orchestrator import Orchestrator, TaskStatus
from app.core.task_queue import get_task_queue
from app.tts.manager import TTSProviderManager
from app.tts.preset import EdgeTTSProvider

router = APIRouter(prefix="/api/project/{project_id}", tags=["synthesis"])


# ─── Schemas ───────────────────────────────────────────────────────

class SynthesisRequest(BaseModel):
    character_voice_map: dict  # {character_name: voice_id}


class SynthesisStatus(BaseModel):
    project_id: int
    status: str  # "idle", "synthesizing", "done", "failed"
    current: int = 0
    total: int = 0
    message: str = ""


# ─── Global State ──────────────────────────────────────────────────

_synthesis_tasks = {}  # project_id -> task info


def _get_tts_manager() -> TTSProviderManager:
    """Get or initialize TTS provider manager."""
    manager = TTSProviderManager()
    manager.register(EdgeTTSProvider())
    return manager


# ─── Background Task ───────────────────────────────────────────────

async def _run_synthesis(project_id: int, character_voice_map: dict):
    """Background synthesis task."""
    task_info = _synthesis_tasks.get(project_id)
    if not task_info:
        return

    try:
        tts_manager = _get_tts_manager()
        orchestrator = Orchestrator(tts_manager)

        results = await orchestrator.run_all(project_id, character_voice_map)

        # update status
        done_count = sum(1 for r in results if r.status.value == "done")
        failed_count = sum(1 for r in results if r.status.value == "failed")

        _synthesis_tasks[project_id]["status"] = "done"
        _synthesis_tasks[project_id]["current"] = len(results)
        _synthesis_tasks[project_id]["message"] = f"Completed: {done_count} done, {failed_count} failed"

        # update project status
        async with async_session() as db:
            project = await db.get(Project, project_id)
            if project:
                project.status = ProjectStatus.done
                await db.commit()

    except Exception as e:
        _synthesis_tasks[project_id]["status"] = "failed"
        _synthesis_tasks[project_id]["message"] = str(e)

        async with async_session() as db:
            project = await db.get(Project, project_id)
            if project:
                project.status = ProjectStatus.created
                await db.commit()


# ─── Endpoints ─────────────────────────────────────────────────────

@router.post("/synthesize", response_model=SynthesisStatus)
async def start_synthesis(
    project_id: int,
    body: SynthesisRequest,
    db: AsyncSession = Depends(get_db),
):
    """启动全卷合成"""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # check if already synthesizing
    if project_id in _synthesis_tasks:
        task_info = _synthesis_tasks[project_id]
        if task_info["status"] == "running":
            return SynthesisStatus(
                project_id=project_id,
                status="synthesizing",
                current=task_info.get("current", 0),
                total=task_info.get("total", 0),
                message="Already synthesizing",
            )

    # create synthesizer and build tasks
    tts_manager = _get_tts_manager()
    orchestrator = Orchestrator(tts_manager)

    async with async_session() as synth_db:
        tasks = await orchestrator.build_tasks(project_id, body.character_voice_map, synth_db)

    if not tasks:
        raise HTTPException(400, "No tasks to synthesize. Check character_voice_map.")

    # start synthesis in background
    _synthesis_tasks[project_id] = {
        "status": "running",
        "current": 0,
        "total": len(tasks),
        "tasks": tasks,
        "orchestrator": orchestrator,
    }

    # update project status
    project.status = ProjectStatus.synthesizing
    await db.commit()

    # run synthesis in background
    asyncio.create_task(_run_synthesis(project_id, body.character_voice_map))

    return SynthesisStatus(
        project_id=project_id,
        status="synthesizing",
        current=0,
        total=len(tasks),
        message=f"Started synthesis with {len(tasks)} tasks",
    )


@router.get("/synthesis/status", response_model=SynthesisStatus)
async def get_synthesis_status(project_id: int):
    """查询合成状态"""
    if project_id not in _synthesis_tasks:
        return SynthesisStatus(
            project_id=project_id,
            status="idle",
            message="No synthesis in progress",
        )

    task_info = _synthesis_tasks[project_id]
    return SynthesisStatus(
        project_id=project_id,
        status=task_info["status"],
        current=task_info.get("current", 0),
        total=task_info.get("total", 0),
        message=task_info.get("message", ""),
    )


@router.delete("/synthesis")
async def cancel_synthesis(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """取消合成"""
    if project_id not in _synthesis_tasks:
        raise HTTPException(404, "No synthesis in progress")

    task_info = _synthesis_tasks[project_id]
    if task_info["status"] != "running":
        raise HTTPException(400, "Synthesis is not running")

    # mark as cancelled
    _synthesis_tasks[project_id]["status"] = "failed"
    _synthesis_tasks[project_id]["message"] = "Cancelled by user"

    # update project status
    project = await db.get(Project, project_id)
    if project:
        project.status = ProjectStatus.created
        await db.commit()

    return {"status": "cancelled", "project_id": project_id}


@router.get("/output")
async def get_output(project_id: int, db: AsyncSession = Depends(get_db)):
    """下载整卷合成音频"""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # find output file
    output_dir = OUTPUT_DIR / str(project_id)
    if not output_dir.exists():
        raise HTTPException(404, "No output found. Run synthesis first.")

    # find wav files
    wav_files = list(output_dir.glob("*.wav"))
    if not wav_files:
        raise HTTPException(404, "No audio files found")

    # return the first (or main) wav file
    main_file = wav_files[0]
    return FileResponse(
        path=str(main_file),
        media_type="audio/wav",
        filename=f"{project.name}.wav",
    )
