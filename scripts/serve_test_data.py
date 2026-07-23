"""生成字幕测试数据包并提供 HTTP 下载。

步骤:
  1. python scripts/serve_test_data.py        # 生成测试包，启动服务
  2. 内网穿透暴露 8765 端口
  3. 检查下载: http://公网IP:端口/test_data.zip

测试包内容:
  test_clip.mp4         — 视频片段（精确对应前 80 行，~10 分钟）
  novel_80.txt          — 前 80 行小说原文
  labels_80.txt         — 前 80 行说话人标注
  timestamps.json       — 每行精确时间戳（毫秒）
  segment_order.json    — 片段文件索引与行号的对应关系
  test_burn.py          — 可直接运行的测试脚本
  README.txt            — 任务说明
"""

import json
import os
import shutil
import subprocess
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

OUTPUT_DIR = Path("output")
WORK_DIR = Path("output/_test_data_pack")
ZIP_PATH = Path("output/test_data.zip")
SERVER_PORT = 8765
TEST_LINES = 80


def ms_to_srt(ms: int) -> str:
    s = ms / 1000
    h = int(s // 3600)
    m = int(s % 3600 // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def main():
    print("=" * 50)
    print("字幕测试数据包")
    print("=" * 50)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    # 1. 计算前 80 行精确时间戳
    print("\n[1/5] 计算时间戳...")
    from pydub import AudioSegment
    segments_dir = OUTPUT_DIR / "segments"
    wav_files = sorted(segments_dir.glob("*.wav"))

    timestamps = []
    offset = 0
    segment_order = []
    for i in range(TEST_LINES):
        wav = wav_files[i] if i < len(wav_files) else None
        dur = 2000
        if wav and wav.exists():
            try:
                dur = len(AudioSegment.from_file(str(wav)))
            except Exception:
                dur = 2000
        timestamps.append({
            "line": i + 1,
            "start_ms": offset,
            "end_ms": offset + dur,
            "duration_ms": dur,
        })
        segment_order.append({
            "line": i + 1,
            "segment_file": wav.name if wav else f"fallback_{i:05d}",
            "duration_ms": dur,
            "start_ms": offset,
            "end_ms": offset + dur,
        })
        offset += dur + 300  # 对话间隔

    total_dur_s = timestamps[-1]["end_ms"] / 1000
    print(f"  前 {TEST_LINES} 行: {total_dur_s:.0f} 秒 ({total_dur_s/60:.1f} 分钟)")

    # 写入时间戳文件
    with open(WORK_DIR / "timestamps.json", "w", encoding="utf-8") as f:
        json.dump(timestamps, f, ensure_ascii=False, indent=2)
    with open(WORK_DIR / "segment_order.json", "w", encoding="utf-8") as f:
        json.dump(segment_order, f, ensure_ascii=False, indent=2)

    # 2. 截取视频
    print("[2/5] 截取视频...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(OUTPUT_DIR / "illustration_video.mp4"),
        "-t", str(total_dur_s),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(WORK_DIR / "test_clip.mp4"),
    ], capture_output=True)
    clip_mb = (WORK_DIR / "test_clip.mp4").stat().st_size / 1024 / 1024
    print(f"  视频: test_clip.mp4 ({clip_mb:.0f} MB)")

    # 3. 提取原文 + 标注
    print("[3/5] 提取文本...")
    novel_lines = Path("novels/novel.txt").read_text(encoding="utf-8").splitlines()
    label_lines = Path("novels/labels.txt").read_text(encoding="utf-8").splitlines()

    with open(WORK_DIR / "novel_80.txt", "w", encoding="utf-8") as f:
        for i in range(min(TEST_LINES, len(novel_lines))):
            f.write(novel_lines[i] + "\n")
    with open(WORK_DIR / "labels_80.txt", "w", encoding="utf-8") as f:
        for i in range(min(TEST_LINES, len(label_lines))):
            f.write(label_lines[i] + "\n")

    # 4. 生成可直接运行的测试脚本
    print("[4/5] 生成测试脚本...")
    test_script = r'''"""
字幕烧录测试脚本。

直接运行:
    python test_burn.py

会在当前目录生成 subtitled_clip.mp4
"""

import json
from pathlib import Path
import subprocess

# 读取时间戳
with open("timestamps.json", encoding="utf-8") as f:
    timestamps = json.load(f)

# 读取原文 + 标注
novel = open("novel_80.txt", encoding="utf-8").read().splitlines()
labels = open("labels_80.txt", encoding="utf-8").read().splitlines()

def ms_to_srt(ms):
    s = ms / 1000
    h = int(s // 3600)
    m = int(s % 3600 // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")

# 生成 SRT 字幕文件
# 要求:
#   1. 中文每行 ≤ 16 字
#   2. 最多 2 行，超出拆成多条字幕
#   3. 断句在语义边界（句号/逗号/问号处）
#   4. 显示格式: [说话人] 原文内容
srt_lines = []
idx = 1
for i, ts in enumerate(timestamps):
    text = novel[i].strip() if i < len(novel) else ""
    speaker = labels[i].strip() if i < len(labels) else ""
    if not text:
        continue

    display = f"[{speaker}] {text}" if speaker else text

    # TODO: 按 16 字/行、最多 2 行规则处理 display
    # TODO: 超长时拆成多条字幕
    # 当前直接使用原文（未分行，待实现）
    srt_lines.append(str(idx))
    srt_lines.append(f"{ms_to_srt(ts['start_ms'])} --> {ms_to_srt(ts['end_ms'])}")
    srt_lines.append(display)
    srt_lines.append("")
    idx += 1

srt_path = Path("test_subtitles.srt")
srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
print(f"字幕文件: {srt_path} ({idx - 1} 条)")

# 烧录到视频
cmd = [
    "ffmpeg", "-y",
    "-i", "test_clip.mp4",
    "-vf", f"subtitles={srt_path}:force_style='FontName=SimHei,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,wrap_unicode=1'",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-c:a", "copy",
    "subtitled_clip.mp4",
]
print("烧录字幕...")
subprocess.run(cmd)
print("完成: subtitled_clip.mp4")
'''
    with open(WORK_DIR / "test_burn.py", "w", encoding="utf-8") as f:
        f.write(test_script)

    # 5. 写入 README
    print("[5/5] 写入 README...")
    readme = f"""字幕烧录任务 — 测试数据包
============================

文件说明
--------
  test_clip.mp4        视频片段（~{total_dur_s/60:.0f} 分钟，对应前 {TEST_LINES} 行小说原文）
  novel_80.txt         前 {TEST_LINES} 行小说原文（每行对应一条字幕）
  labels_80.txt        前 {TEST_LINES} 行说话人标注（与 novel_80.txt 逐行对齐）
  timestamps.json      每行的精确时间戳（start_ms / end_ms / duration_ms）
  segment_order.json   音频片段索引（行号 -> 文件名 -> 时间戳）
  test_burn.py         可直接运行的测试脚本

任务说明
--------
修改 test_burn.py 中的 TODO 部分，实现以下功能：

1. 中文分行：每行 ≤ 16 个汉字，最多 2 行，超出拆成多条字幕
2. 语义断句：在句号（。）、问号（？）、感叹号（！）、逗号（，）处断句
3. 格式：显示 "[说话人] 原文内容"
4. 烧录参数：FontName=SimHei, FontSize=20, wrap_unicode=1

完成后烧录到 test_clip.mp4 验证效果。

集成到主项目
----------
功能最终需要集成到 scripts/generate_video.py 中。

关键代码位置:
  scripts/generate_video.py        — 视频生成主脚本（需修改）
  output/illustration_plan.json    — 插图时间轴（702 张）
  output/segments/*.wav            — 音频片段（3026 个，用于时间戳）
  backend/app/core/splicer.py      — 拼接引擎（片段间间隔逻辑）
"""
    (WORK_DIR / "README.txt").write_text(readme, encoding="utf-8")

    # 6. 打包
    print(f"\n打包 -> {ZIP_PATH.name}...")
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", WORK_DIR)
    zip_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    shutil.rmtree(WORK_DIR)
    print(f"  完成 ({zip_mb:.0f} MB)")

    # 7. 启动 HTTP 服务
    print(f"\n启动 HTTP: http://localhost:{SERVER_PORT}")
    print(f"下载: http://localhost:{SERVER_PORT}/test_data.zip")
    print("内网穿透后: http://公网IP:端口/test_data.zip")
    print("Ctrl+C 停止\n")

    os.chdir(str(OUTPUT_DIR))
    server = HTTPServer(("0.0.0.0", SERVER_PORT), SimpleHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")


if __name__ == "__main__":
    main()