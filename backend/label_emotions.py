"""
情感标注脚本 - 标注全卷对话的情感和语气
使用预计算的性别结果，跳过 LLM 识别
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from app.core.parser import parse
from app.core.emotion_labeler import (
    EMOTION_PIPELINE_VERSION,
    EMOTIONS,
    TONES,
    emotion_source_hash,
    label_all_emotions,
)
from app.core.llm_client import LLMClient, SENSENOVA_FLASH_LITE_MODEL

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent.parent
NOVEL_PATH = BASE_DIR / "novels" / "novel.txt"
LABELS_PATH = BASE_DIR / "novels" / "labels.txt"
OUTPUT_PATH = BASE_DIR / "backend" / "data" / "emotion_results.json"


def configure_logging() -> None:
    log_path = BASE_DIR / "logs" / "label_emotions.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    configure_logging()
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
    client = LLMClient.for_flash_lite("emotion")
    total_dialogues = sum(
        1 for dialogue in dialogues
        if dialogue.get("speaker") and dialogue.get("speaker") != "旁白"
    )
    success_count = 0
    fail_count = 0

    print(f"\n[3/3] 开始标注 {total_dialogues} 条对话的情感...")

    checkpoint_path = BASE_DIR / "backend" / "data" / "emotion_results.checkpoint.json"
    result_map = label_all_emotions(
        dialogues,
        novel_text,
        client=client,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    results = [result_map[key] for key in sorted(result_map, key=int)]
    success_count = len(results)
    elapsed_total = time.time() - t0
    print(f"\n[标注] 完成: {format_time(elapsed_total)}")
    print(f"  成功: {success_count} 条")
    print(f"  失败: {fail_count} 条")

    adjudicated = sum(bool(item.get("adjudicated")) for item in results)
    low_confidence = sum(float(item.get("confidence", 0)) < 0.7 for item in results)
    retried = sum(int(item.get("item_attempts", 1)) > 1 for item in results)
    print("\n[质量审计]")
    print(f"  双 Agent 审核: {sum(bool(item.get('reviewed')) for item in results)}/{success_count}")
    print(f"  分歧裁决: {adjudicated} 条")
    print(f"  低置信度(<0.7): {low_confidence} 条")
    print(f"  条目级重试: {retried} 条")

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    cumulative_usage = checkpoint_payload.get("llm_usage", client.usage_summary())

    # 4. 保存结果
    output_data = {
        "total": total_dialogues,
        "success": success_count,
        "fail": fail_count,
        "emotions": EMOTIONS,
        "tones": TONES,
        "meta": {
            "model": SENSENOVA_FLASH_LITE_MODEL,
            "pipeline_version": EMOTION_PIPELINE_VERSION,
            "source_hash": emotion_source_hash(novel_text, dialogues),
        },
        "results": result_map,
        "llm_usage": cumulative_usage,
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
