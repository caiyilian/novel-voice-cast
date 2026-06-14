"""
Novel Voice Cast - 主入口
小说文本 + 标注 → 音频文件
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# 添加 backend 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step_parse(config: dict) -> tuple:
    """步骤1：解析小说"""
    from app.core.parser import parse

    t0 = time.time()
    novel_path = config["novel"]["text_path"]
    labels_path = config["novel"]["labels_path"]

    with open(novel_path, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    dialogues, characters = parse(novel_text, labels)
    elapsed = time.time() - t0

    print(f"  解析完成: {len(dialogues)} 条对话, {len(characters)} 个角色 [{elapsed:.2f}s]")
    return dialogues, characters, novel_text


def step_gender(config: dict, characters: list, dialogues: list, novel_text: str) -> dict:
    """步骤2：性别识别"""
    gender_path = Path("backend/data/gender_results.json")

    # 如果已有结果，直接加载
    if gender_path.exists():
        t0 = time.time()
        with open(gender_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        elapsed = time.time() - t0
        print(f"  加载已有结果: {len(results)} 个角色 [{elapsed:.2f}s]")
        return results

    # 如果没有结果，使用 LLM 识别
    if not config["features"].get("gender_identify", True):
        print("  跳过（已禁用）")
        return {}

    from app.core.gender_identifier import identify_gender
    from app.core.ollama_client import OllamaClient

    t0 = time.time()
    client = OllamaClient()
    results = {}

    for i, char_name in enumerate(characters):
        # 构建角色上下文
        text_parts = []
        for d in dialogues:
            if d.get("speaker") == char_name:
                text_parts.append(d["text"])
        char_text = "\n".join(text_parts[:50])

        try:
            result = identify_gender(char_name, char_text, client, max_tool_steps=5)
            results[char_name] = {
                "gender": result["gender"],
                "confidence": result["confidence"],
            }
            status = "✓" if result["confidence"] > 0.5 else "?"
            print(f"\r  [{i+1:2d}/{len(characters)}] {char_name}: {result['gender']} ({result['confidence']:.2f}) {status}   ", end="", flush=True)
        except Exception as e:
            results[char_name] = {"gender": "male", "confidence": 0.3}
            print(f"\r  [{i+1:2d}/{len(characters)}] {char_name}: error - {e}   ", end="", flush=True)

    print()
    elapsed = time.time() - t0

    # 保存结果
    gender_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gender_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  识别完成: {len(results)} 个角色 [{elapsed:.2f}s]")
    return results


def step_emotion(config: dict, dialogues: list, novel_text: str) -> list:
    """步骤3：情感标注"""
    emotion_path = Path("backend/data/emotion_results.json")

    # 如果已有结果，直接加载
    if emotion_path.exists():
        t0 = time.time()
        with open(emotion_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        elapsed = time.time() - t0
        print(f"  加载已有结果: {len(results)} 条 [{elapsed:.2f}s]")
        return results

    # 如果没有结果，跳过（需要先运行 label_emotions.py）
    print("  跳过（未找到结果文件，请先运行 label_emotions.py）")
    return []


def step_tts(config: dict, dialogues: list, gender_results: dict, emotion_results: list) -> list:
    """步骤4：TTS 合成（模拟）"""
    t0 = time.time()
    total = len(dialogues)

    # 模拟 TTS 合成
    segments = []
    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")
        chapter = dialogue.get("chapter", "unknown")

        # 模拟生成音频文件路径
        filename = f"{i:05d}_{speaker}.wav"
        output_path = str(Path(config["output"]["dir"]) / config["output"]["segments_dir"] / filename)

        segments.append({
            "audio_path": output_path,
            "chapter": chapter,
            "order": i,
            "speaker": speaker,
        })

        # 进度显示
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remaining = avg * (total - i - 1)
            print(f"  [{i+1:4d}/{total}] 模拟完成 | 剩余 ~{format_time(remaining)}   ", end="", flush=True)

    print()
    elapsed = time.time() - t0
    print(f"  合成完成: {len(segments)} 条 [{elapsed:.2f}s]")
    return segments


def step_splice(config: dict, segments: list) -> str:
    """步骤5：音频拼接（模拟）"""
    t0 = time.time()

    output_path = str(Path(config["output"]["dir"]) / config["output"]["filename"])
    print(f"  输出路径: {output_path}")

    elapsed = time.time() - t0
    print(f"  拼接完成 [{elapsed:.2f}s]")
    return output_path


def step_denoise(config: dict, audio_path: str) -> str:
    """步骤6：去噪（模拟）"""
    if not config["features"].get("denoise", True):
        print("  跳过（已禁用）")
        return audio_path

    t0 = time.time()

    denoised_path = audio_path.replace(".wav", "_denoised.wav")
    print(f"  输出路径: {denoised_path}")

    elapsed = time.time() - t0
    print(f"  去噪完成 [{elapsed:.2f}s]")
    return denoised_path


def main():
    parser = argparse.ArgumentParser(description="Novel Voice Cast")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 60)
    print("Novel Voice Cast")
    print("=" * 60)

    # 加载配置
    config = load_config(args.config)
    print(f"\n配置文件: {args.config}")

    # 步骤1：解析
    print(f"\n[1/6] 解析小说")
    dialogues, characters, novel_text = step_parse(config)

    # 步骤2：性别识别
    print(f"\n[2/6] 性别识别")
    gender_results = step_gender(config, characters, dialogues, novel_text)

    # 步骤3：情感标注
    print(f"\n[3/6] 情感标注")
    emotion_results = step_emotion(config, dialogues, novel_text)

    # 步骤4：TTS 合成
    print(f"\n[4/6] TTS 合成")
    segments = step_tts(config, dialogues, gender_results, emotion_results)

    # 步骤5：音频拼接
    print(f"\n[5/6] 音频拼接")
    audio_path = step_splice(config, segments)

    # 步骤6：去噪
    print(f"\n[6/6] 音频去噪")
    final_path = step_denoise(config, audio_path)

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  输出: {final_path}")
    print(f"  总耗时: {format_time(total_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
