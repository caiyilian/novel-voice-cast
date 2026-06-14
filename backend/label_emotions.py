"""
情感标注脚本 - 标注全卷对话的情感和语气
使用预计算的性别结果，跳过 LLM 识别
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from app.core.parser import parse
from app.core.emotion_labeler import label_emotion, EMOTIONS, TONES
from app.core.ollama_client import OllamaClient

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent.parent
NOVEL_PATH = BASE_DIR / "novels" / "novel.txt"
LABELS_PATH = BASE_DIR / "novels" / "labels.txt"
OUTPUT_PATH = BASE_DIR / "backend" / "data" / "emotion_results.json"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    total_start = time.time()

    print("=" * 60)
    print("情感标注 - 全卷")
    print("=" * 60)

    # 1. 读取文件
    t0 = time.time()
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"\n[1/3] 读取文件: {time.time()-t0:.2f}s")
    print(f"  小说: {len(novel_text.splitlines())} 行")
    print(f"  标注: {len(labels)} 条")

    # 2. 解析对话
    t0 = time.time()
    dialogues, characters = parse(novel_text, labels)
    print(f"\n[2/3] 解析对话: {time.time()-t0:.2f}s")
    print(f"  对话: {len(dialogues)} 条")
    print(f"  角色: {len(characters)} 个")

    # 3. 标注情感
    t0 = time.time()
    client = OllamaClient()
    total_dialogues = len(dialogues)
    success_count = 0
    fail_count = 0
    results = []

    print(f"\n[3/3] 开始标注 {total_dialogues} 条对话的情感...")

    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")
        text = dialogue["text"]
        line_num = dialogue.get("line", i + 1)

        try:
            result = label_emotion(
                dialogue_text=text,
                dialogue_line=line_num,
                dialogue_index=i,
                text=novel_text,
                client=client,
                max_tool_steps=5,
            )
            result["speaker"] = speaker
            result["text"] = text
            results.append(result)
            success_count += 1

        except Exception as e:
            fail_count += 1
            results.append({
                "dialogue_index": i,
                "speaker": speaker,
                "text": text,
                "emotion": "calm",
                "tone": "serious",
                "confidence": 0.0,
                "evidence": f"Error: {str(e)}",
            })

        # 进度显示 - 每条更新
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (total_dialogues - i - 1)
        emotion = results[-1].get("emotion", "?")
        tone = results[-1].get("tone", "?")
        print(f"\r  [{i+1:4d}/{total_dialogues}] "
              f"成功 {success_count:4d} 失败 {fail_count:2d} | "
              f"{speaker[:6]:6s} {emotion:10s} {tone:10s} | "
              f"已用 {format_time(elapsed)} 剩余 ~{format_time(remaining)}   ", end="", flush=True)

    print()  # 换行
    elapsed_total = time.time() - t0
    print(f"\n[标注] 完成: {format_time(elapsed_total)}")
    print(f"  成功: {success_count} 条")
    print(f"  失败: {fail_count} 条")

    # 4. 保存结果
    output_data = {
        "total": total_dialogues,
        "success": success_count,
        "fail": fail_count,
        "emotions": EMOTIONS,
        "tones": TONES,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("耗时汇总")
    print("=" * 60)
    print(f"  总耗时: {format_time(total_time)}")
    print(f"  平均每条: {elapsed_total/total_dialogues:.2f}s")
    print(f"  输出: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
