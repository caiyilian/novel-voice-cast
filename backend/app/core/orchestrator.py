"""
合成调度器 — 按角色合并对话，调用 TTS 合成，管理任务状态。

核心逻辑：
1. 读取项目的所有对话（按 order 排序）
2. 合并同角色连续对话（减少模型调用）
3. 逐角色调用 TTS 合成
4. 输出音频文件
"""
import asyncio
import enum
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models import Project, Character, Dialogue, AudioFile, AudioSource, ProjectStatus
from app.tts.manager import TTSProviderManager
from app.config import OUTPUT_DIR


# ─── Task State Machine ────────────────────────────────────────────

class TaskStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class SynthesisTask:
    """单个角色的合成任务"""
    character_name: str
    voice_id: str
    text_segments: List[str]
    emotion: str = "calm"
    tone: str = "serious"
    status: TaskStatus = TaskStatus.pending
    audio_path: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# ─── Dialogue Merging ──────────────────────────────────────────────

def merge_consecutive_dialogues(dialogues: List[Dict]) -> List[Dict]:
    """合并同角色连续对话，减少模型调用次数。

    输入: [{"speaker": "A", "text": "你好"}, {"speaker": "A", "text": "再见"}, {"speaker": "B", "text": "嗯"}]
    输出: [{"speaker": "A", "text": "你好\n再见"}, {"speaker": "B", "text": "嗯"}]
    """
    if not dialogues:
        return []

    merged = []
    current = dialogues[0].copy()

    for d in dialogues[1:]:
        if d["speaker"] == current["speaker"]:
            # 同角色，合并文本（用换行连接）
            current["text"] += "\n" + d["text"]
        else:
            # 不同角色，保存当前，开始新的
            merged.append(current)
            current = d.copy()

    merged.append(current)
    return merged


# ─── Orchestrator ──────────────────────────────────────────────────

class Orchestrator:
    """合成调度器"""

    def __init__(self, tts_manager: TTSProviderManager, output_dir: str = None):
        self.tts_manager = tts_manager
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def build_tasks(
        self,
        project_id: int,
        character_voice_map: Dict[str, str],
        db: AsyncSession,
    ) -> List[SynthesisTask]:
        """构建合成任务列表。

        Args:
            project_id: 项目 ID
            character_voice_map: {角色名: voice_id} 映射
            db: 数据库 session

        Returns:
            合成任务列表
        """
        # 获取所有对话（按 order 排序）
        result = await db.execute(
            select(Dialogue)
            .where(Dialogue.project_id == project_id)
            .order_by(Dialogue.order)
        )
        dialogues = result.scalars().all()

        if not dialogues:
            return []

        # 转换为 dict 格式
        dialogue_dicts = [
            {
                "speaker": d.speaker,
                "text": d.text,
                "emotion": d.emotion or "calm",
                "tone": d.tone or "serious",
                "order": d.order,
            }
            for d in dialogues
        ]

        # 合并同角色连续对话
        merged = merge_consecutive_dialogues(dialogue_dicts)

        # 按角色分组
        character_tasks: Dict[str, Dict] = {}
        for d in merged:
            speaker = d["speaker"]
            if speaker not in character_tasks:
                character_tasks[speaker] = {
                    "text_segments": [],
                    "emotion": d["emotion"],
                    "tone": d["tone"],
                }
            character_tasks[speaker]["text_segments"].append(d["text"])

        # 构建任务列表
        tasks = []
        for speaker, info in character_tasks.items():
            voice_id = character_voice_map.get(speaker, "")
            if not voice_id:
                continue  # 跳过没有分配音频的角色

            task = SynthesisTask(
                character_name=speaker,
                voice_id=voice_id,
                text_segments=info["text_segments"],
                emotion=info["emotion"],
                tone=info["tone"],
            )
            tasks.append(task)

        return tasks

    async def run_task(
        self,
        task: SynthesisTask,
        chapter: str = "",
        project_id: int = 0,
    ) -> str:
        """执行单个合成任务。

        Args:
            task: 合成任务
            chapter: 章节名（用于输出文件命名）
            project_id: 项目 ID

        Returns:
            输出音频文件路径
        """
        task.status = TaskStatus.running
        task.start_time = time.time()

        try:
            # 合并文本段落
            full_text = "\n".join(task.text_segments)

            # 调用 TTS 合成
            audio_bytes = await self.tts_manager.synthesize(
                text=full_text,
                voice_id=task.voice_id,
                emotion=task.emotion,
                tone=task.tone,
            )

            # 保存音频文件
            safe_name = "".join(c for c in task.character_name if c.isalnum() or c in "_-")
            filename = f"{safe_name}.wav"
            output_path = self.output_dir / str(project_id) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            task.audio_path = str(output_path)
            task.status = TaskStatus.done
            task.end_time = time.time()

            return str(output_path)

        except Exception as e:
            task.status = TaskStatus.failed
            task.error = str(e)
            task.end_time = time.time()
            raise

    async def run_all(
        self,
        project_id: int,
        character_voice_map: Dict[str, str],
        progress_callback=None,
    ) -> List[SynthesisTask]:
        """执行所有合成任务。

        Args:
            project_id: 项目 ID
            character_voice_map: {角色名: voice_id} 映射
            progress_callback: 进度回调函数 callback(current, total, task)

        Returns:
            所有任务的结果
        """
        async with async_session() as db:
            tasks = await self.build_tasks(project_id, character_voice_map, db)

        if not tasks:
            return []

        results = []
        for i, task in enumerate(tasks):
            if progress_callback:
                await progress_callback(i, len(tasks), task)

            try:
                await self.run_task(task, project_id=project_id)
            except Exception:
                pass  # 错误已记录在 task.error 中

            results.append(task)

        if progress_callback:
            await progress_callback(len(tasks), len(tasks), None)

        return results

    def get_summary(self, tasks: List[SynthesisTask]) -> Dict[str, Any]:
        """获取任务执行摘要。"""
        total = len(tasks)
        done = sum(1 for t in tasks if t.status == TaskStatus.done)
        failed = sum(1 for t in tasks if t.status == TaskStatus.failed)
        pending = sum(1 for t in tasks if t.status == TaskStatus.pending)

        total_duration = sum(t.duration or 0 for t in tasks if t.duration)

        return {
            "total": total,
            "done": done,
            "failed": failed,
            "pending": pending,
            "total_duration": total_duration,
            "tasks": [
                {
                    "character": t.character_name,
                    "status": t.status.value,
                    "duration": t.duration,
                    "audio_path": t.audio_path,
                    "error": t.error,
                }
                for t in tasks
            ],
        }
