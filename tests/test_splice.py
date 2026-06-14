"""
音频拼接脚本 - 把 test_output/segments 拼接成整卷音频
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.core.splicer import AudioSplicer

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent
SEGMENTS_DIR = BASE_DIR / "test_output" / "segments"
SEGMENTS_JSON = BASE_DIR / "test_output" / "segments.json"
OUTPUT_PATH = BASE_DIR / "test_output" / "full_volume.wav"


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
    print("音频拼接")
    print("=" * 60)

    # 加载片段信息
    t0 = time.time()
    with open(SEGMENTS_JSON, "r", encoding="utf-8") as f:
        segments = json.load(f)
    print(f"\n[1/2] 加载片段: {len(segments)} 条 [{time.time()-t0:.2f}s]")

    # 拼接
    t0 = time.time()
    splicer = AudioSplicer()
    final_audio = splicer.splice(segments, output_path=str(OUTPUT_PATH))
    duration = splicer.get_duration(final_audio)
    print(f"\n[2/2] 拼接完成 [{time.time()-t0:.2f}s]")
    print(f"  时长: {format_time(duration)}")

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  输出: {OUTPUT_PATH}")
    print(f"  时长: {format_time(duration)}")
    print(f"  总耗时: {format_time(total_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
