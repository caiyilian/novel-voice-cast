"""Tests for the synthesis orchestrator."""
import sys
sys.path.insert(0, "..")

import asyncio
from app.core.orchestrator import merge_consecutive_dialogues, Orchestrator, SynthesisTask, TaskStatus
from app.tts.manager import TTSProviderManager


# ─── Mock TTS Provider ─────────────────────────────────────────────

class MockProvider:
    """Mock TTS provider for testing."""

    def __init__(self, name="mock"):
        self._name = name

    @property
    def name(self):
        return self._name

    async def synthesize(self, text, voice_id, emotion=None, tone=None, **params):
        return f"audio:{voice_id}:{text[:20]}".encode()

    async def clone_voice(self, audio_path, voice_name):
        pass

    async def get_voices(self):
        return []


# ─── Tests ─────────────────────────────────────────────────────────

def test_merge_same_character():
    """合并同角色连续对话"""
    dialogues = [
        {"speaker": "A", "text": "你好"},
        {"speaker": "A", "text": "再见"},
        {"speaker": "B", "text": "嗯"},
    ]
    merged = merge_consecutive_dialogues(dialogues)
    assert len(merged) == 2
    assert merged[0]["speaker"] == "A"
    assert merged[0]["text"] == "你好\n再见"
    assert merged[1]["speaker"] == "B"
    assert merged[1]["text"] == "嗯"
    print("[PASS] Merge same character")


def test_merge_different_characters():
    """不同角色不合并"""
    dialogues = [
        {"speaker": "A", "text": "你好"},
        {"speaker": "B", "text": "再见"},
        {"speaker": "A", "text": "嗯"},
    ]
    merged = merge_consecutive_dialogues(dialogues)
    assert len(merged) == 3
    assert merged[0]["speaker"] == "A"
    assert merged[1]["speaker"] == "B"
    assert merged[2]["speaker"] == "A"
    print("[PASS] Merge different characters")


def test_merge_empty():
    """空列表"""
    merged = merge_consecutive_dialogues([])
    assert len(merged) == 0
    print("[PASS] Merge empty list")


def test_merge_single():
    """单条对话"""
    merged = merge_consecutive_dialogues([{"speaker": "A", "text": "你好"}])
    assert len(merged) == 1
    assert merged[0]["text"] == "你好"
    print("[PASS] Merge single dialogue")


def test_merge_long_chain():
    """长链合并"""
    dialogues = [{"speaker": "A", "text": f"第{i}句"} for i in range(10)]
    merged = merge_consecutive_dialogues(dialogues)
    assert len(merged) == 1
    assert merged[0]["text"].count("\n") == 9  # 10句话，9个换行
    print("[PASS] Merge long chain")


def test_task_state_machine():
    """任务状态机"""
    task = SynthesisTask(
        character_name="test",
        voice_id="test_voice",
        text_segments=["hello"],
    )
    assert task.status == TaskStatus.pending

    task.status = TaskStatus.running
    task.start_time = 100.0
    assert task.status == TaskStatus.running

    task.status = TaskStatus.done
    task.end_time = 101.0
    assert task.status == TaskStatus.done
    assert abs(task.duration - 1.0) < 0.01

    task.status = TaskStatus.failed
    task.error = "test error"
    assert task.status == TaskStatus.failed
    print("[PASS] Task state machine")


def test_orchestrator_init():
    """调度器初始化"""
    manager = TTSProviderManager()
    manager.register(MockProvider())
    orchestrator = Orchestrator(manager)
    assert orchestrator.output_dir.exists()
    print("[PASS] Orchestrator init")


async def test_orchestrator_run_task():
    """执行单个任务"""
    manager = TTSProviderManager()
    manager.register(MockProvider())
    manager.map_voice("mock", "mock")
    orchestrator = Orchestrator(manager)

    task = SynthesisTask(
        character_name="test_char",
        voice_id="mock",
        text_segments=["hello world"],
    )

    audio_path = await orchestrator.run_task(task, project_id=1)
    assert task.status == TaskStatus.done
    assert task.audio_path is not None
    assert task.duration is not None
    print(f"[PASS] Run task: {audio_path}")


def run_async_tests():
    """Run async tests."""
    asyncio.run(test_orchestrator_run_task())


if __name__ == "__main__":
    test_merge_same_character()
    test_merge_different_characters()
    test_merge_empty()
    test_merge_single()
    test_merge_long_chain()
    test_task_state_machine()
    test_orchestrator_init()
    run_async_tests()
    print("\nAll orchestrator tests passed!")
