"""Tests for the audio splicer."""
import sys, os, tempfile
sys.path.insert(0, "..")

from pydub import AudioSegment
from app.core.splicer import AudioSplicer, GAP_DIALOGUE, GAP_PARAGRAPH, GAP_CHAPTER, FADE_DURATION


# ─── Helpers ───────────────────────────────────────────────────────

def create_test_audio(duration_ms: int = 1000) -> AudioSegment:
    """创建测试音频（正弦波）"""
    import math
    sample_rate = 24000
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)]
    audio = AudioSegment(
        data=b''.join([s.to_bytes(2, 'little', signed=True) for s in samples]),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    return audio


def save_test_audio(audio: AudioSegment, path: str):
    """保存测试音频"""
    audio.export(path, format="wav")


# ─── Tests ─────────────────────────────────────────────────────────

def test_splice_empty():
    """空列表"""
    splicer = AudioSplicer()
    result = splicer.splice([])
    assert len(result) == 0
    print("[PASS] Splice empty")


def test_splice_single():
    """单个音频"""
    splicer = AudioSplicer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        save_test_audio(create_test_audio(1000), f.name)
        path = f.name
    segments = [{"audio_path": path, "chapter": "ch1", "order": 1}]
    result = splicer.splice(segments)
    assert len(result) > 0
    os.unlink(path)
    print("[PASS] Splice single")


def test_splice_multiple():
    """多个音频拼接"""
    splicer = AudioSplicer()
    paths = []
    for i in range(3):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_test_audio(create_test_audio(1000), f.name)
        paths.append(f.name)

    segments = [
        {"audio_path": paths[0], "chapter": "ch1", "order": 1},
        {"audio_path": paths[1], "chapter": "ch1", "order": 2},
        {"audio_path": paths[2], "chapter": "ch2", "order": 3},
    ]

    result = splicer.splice(segments)
    # 3个1秒音频 + 2个间隔 = 应该大于3秒
    assert splicer.get_duration(result) > 3.0

    for p in paths:
        try:
            os.unlink(p)
        except PermissionError:
            pass  # 文件可能还在被使用
    print(f"[PASS] Splice multiple: {splicer.get_duration(result):.2f}s")


def test_splice_chapter_gap():
    """章节间隔测试"""
    splicer = AudioSplicer()
    paths = []
    for i in range(2):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_test_audio(create_test_audio(1000), f.name)
        paths.append(f.name)

    segments = [
        {"audio_path": paths[0], "chapter": "ch1", "order": 1},
        {"audio_path": paths[1], "chapter": "ch2", "order": 2},
    ]

    result = splicer.splice(segments)
    # 2个1秒音频 + 章节间隔2秒 = 4秒
    duration = splicer.get_duration(result)
    assert duration > 3.5, f"Expected > 3.5s, got {duration}s"

    for p in paths:
        try:
            os.unlink(p)
        except PermissionError:
            pass
    print(f"[PASS] Chapter gap: {duration:.2f}s")


def test_splice_dialogue_gap():
    """对话间隔测试"""
    splicer = AudioSplicer()
    paths = []
    for i in range(3):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_test_audio(create_test_audio(1000), f.name)
        paths.append(f.name)

    segments = [
        {"audio_path": paths[0], "chapter": "ch1", "order": 1},
        {"audio_path": paths[1], "chapter": "ch1", "order": 2},
        {"audio_path": paths[2], "chapter": "ch1", "order": 3},
    ]

    result = splicer.splice(segments)
    # 3个1秒音频 + 2个对话间隔0.3秒 = 3.6秒
    duration = splicer.get_duration(result)
    assert duration > 3.5, f"Expected > 3.5s, got {duration}s"

    for p in paths:
        try:
            os.unlink(p)
        except PermissionError:
            pass
    print(f"[PASS] Dialogue gap: {duration:.2f}s")


def test_splice_by_chapter():
    """按章节拆分"""
    splicer = AudioSplicer()
    paths = []
    for i in range(4):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_test_audio(create_test_audio(1000), f.name)
        paths.append(f.name)

    segments = [
        {"audio_path": paths[0], "chapter": "ch1", "order": 1},
        {"audio_path": paths[1], "chapter": "ch1", "order": 2},
        {"audio_path": paths[2], "chapter": "ch2", "order": 3},
        {"audio_path": paths[3], "chapter": "ch2", "order": 4},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_files = splicer.splice_by_chapter(segments, tmpdir)
        assert len(output_files) == 2
        for f in output_files:
            assert os.path.exists(f)

    for p in paths:
        os.unlink(p)
    print(f"[PASS] Splice by chapter: {len(output_files)} files")


def test_fade_effects():
    """淡入淡出效果"""
    splicer = AudioSplicer(fade_duration=100)
    audio = create_test_audio(500)
    faded = splicer._apply_fade(audio)
    # 淡入淡出后的音频长度可能相同或略短
    assert len(faded) <= len(audio)
    assert len(faded) >= len(audio) - 100  # 允许一些误差
    print(f"[PASS] Fade effects: {len(audio)}ms -> {len(faded)}ms")


def test_get_duration():
    """获取时长"""
    splicer = AudioSplicer()
    audio = create_test_audio(2500)
    duration = splicer.get_duration(audio)
    assert abs(duration - 2.5) < 0.01
    print(f"[PASS] Get duration: {duration:.2f}s")


if __name__ == "__main__":
    test_splice_empty()
    test_splice_single()
    test_splice_multiple()
    test_splice_chapter_gap()
    test_splice_dialogue_gap()
    test_splice_by_chapter()
    test_fade_effects()
    test_get_duration()
    print("\nAll splicer tests passed!")
