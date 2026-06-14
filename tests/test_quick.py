"""
快速测试脚本 - 全部用 pyttsx3 离线合成
输出到 test_output 目录，不影响正式合成
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.core.parser import parse

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent
NOVEL_PATH = BASE_DIR / "novels" / "novel.txt"
LABELS_PATH = BASE_DIR / "novels" / "labels.txt"
OUTPUT_DIR = BASE_DIR / "test_output"
SEGMENTS_DIR = OUTPUT_DIR / "segments"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def synthesize_pyttsx3(text: str) -> bytes:
    """使用 pyttsx3 合成（每次重新初始化引擎）"""
    import tempfile
    import pyttsx3

    engine = pyttsx3.init()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        engine.stop()

        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    total_start = time.time()

    print("=" * 60)
    print("快速测试 - pyttsx3 离线合成")
    print("=" * 60)

    # 1. 解析小说
    t0 = time.time()
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    dialogues, characters = parse(novel_text, labels)
    print(f"\n[1/2] 解析完成: {len(dialogues)} 条对话, {len(characters)} 个角色 [{time.time()-t0:.2f}s]")

    # 2. 合成所有对话
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    segments = []
    total = len(dialogues)
    success = 0
    fail = 0

    print(f"\n[2/2] 开始合成 {total} 条对话...")

    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")
        text = dialogue["text"]
        chapter = dialogue.get("chapter", "unknown")

        filename = f"{i:05d}_{speaker}.wav"
        output_path = str(SEGMENTS_DIR / filename)

        try:
            audio_bytes = synthesize_pyttsx3(text)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            segments.append({
                "audio_path": output_path,
                "chapter": chapter,
                "order": i,
                "speaker": speaker,
            })
            success += 1

        except Exception as e:
            fail += 1
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=1000)
            silence.export(output_path, format="wav")
            segments.append({
                "audio_path": output_path,
                "chapter": chapter,
                "order": i,
                "speaker": speaker,
            })

        # 进度显示
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (total - i - 1)
        print(f"\r  [{i+1:4d}/{total}] 成功 {success:4d} 失败 {fail:2d} | 已用 {format_time(elapsed)} 剩余 ~{format_time(remaining)}   ", end="", flush=True)

    print()
    elapsed = time.time() - t0
    print(f"\n[合成] 完成: {format_time(elapsed)}")
    print(f"  成功: {success} 条")
    print(f"  失败: {fail} 条")

    # 3. 保存片段信息
    segments_path = OUTPUT_DIR / "segments.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  片段数量: {len(segments)}")
    print(f"  总耗时: {format_time(total_time)}")
    print(f"  平均每条: {elapsed/total:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
