"""
计算 VoxCPM 合成剩余时间
基于 output/segments 目录中文件的创建时间
"""
import os
from pathlib import Path
from datetime import datetime

SEGMENTS_DIR = Path(__file__).parent.parent.parent / "output" / "segments"
TOTAL_DIALOGUES = 3022  # 总对话数（包括旁白）

# 获取所有 wav 文件
files = sorted(SEGMENTS_DIR.glob("*.wav"))

if not files:
    print("没有找到音频文件")
    exit()

# 获取第一个和最后一个文件的时间
first_file = files[0]
last_file = files[-1]

# 获取创建时间
first_time = datetime.fromtimestamp(first_file.stat().st_ctime)
last_time = datetime.fromtimestamp(last_file.stat().st_ctime)

# 计算
completed = len(files)
remaining = TOTAL_DIALOGUES - completed
elapsed = (last_time - first_time).total_seconds()
avg_per_file = elapsed / completed if completed > 0 else 0
estimated_remaining = avg_per_file * remaining

# 格式化时间
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"

print("=" * 50)
print("VoxCPM 合成进度")
print("=" * 50)
print(f"  总对话数: {TOTAL_DIALOGUES}")
print(f"  已完成: {completed} ({completed/TOTAL_DIALOGUES*100:.1f}%)")
print(f"  剩余: {remaining}")
print(f"  ")
print(f"  第一个文件: {first_file.name}")
print(f"  创建时间: {first_time.strftime('%H:%M:%S')}")
print(f"  ")
print(f"  最新文件: {last_file.name}")
print(f"  创建时间: {last_time.strftime('%H:%M:%S')}")
print(f"  ")
print(f"  已用时间: {format_time(elapsed)}")
print(f"  平均每条: {avg_per_file:.1f}s")
print(f"  预计剩余: {format_time(estimated_remaining)}")
from datetime import timedelta
estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)
print(f"  预计完成: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)
