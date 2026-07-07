"""插图规划 — 分析全卷小说，输出插图计划

直接双击或用终端运行：
    python run_illustration_plan.py

断点续跑：
    python run_illustration_plan.py --resume
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.core.illustration_planner import plan_illustrations
from app.core.llm_client import LLMClient

NOVEL_PATH = Path("novels/novel.txt")
LABELS_PATH = Path("novels/labels.txt")
OUTPUT_PATH = "output/illustration_plan.json"

RESUME = "--resume" in sys.argv
DEBUG = "--debug" in sys.argv


def main():
    if not NOVEL_PATH.is_file():
        print(f"错误: 小说文件不存在 {NOVEL_PATH}")
        return 1

    text = NOVEL_PATH.read_text(encoding="utf-8")

    labels = []
    if LABELS_PATH.is_file():
        labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()
        print(f"台词标注: {len(labels)} 行")

    print(f"小说: {NOVEL_PATH.name} ({len(text)} 字符, {len(text.splitlines())} 行)")
    print(f"输出: {OUTPUT_PATH}")
    if RESUME:
        print("模式: 断点续跑")
    if DEBUG:
        print("模式: DEBUG raw response logging")
    print()

    client = LLMClient()
    print(f"LLM: {client.log_summary()}")
    print()

    t0 = time.time()
    try:
        plan = plan_illustrations(
            text=text,
            labels=labels,
            client=client,
            output_path=OUTPUT_PATH,
            resume=RESUME,
            debug=DEBUG,
        )
    except Exception as e:
        print(f"\n错误: {e}")
        return 1

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f} 秒")
    print(f"插图提案: {len(plan)} 个")
    print(f"计划已保存到: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
