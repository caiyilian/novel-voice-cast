"""从插图 + 音频生成视频。

流程:
1. 读取所有音频片段 (output/segments/*.wav)，计算每行对应的时间戳
2. 读取插图计划，将每张插图映射到时间范围
3. 用 ffmpeg 合成视频

用法:
    .venv\Scripts\python scripts/generate_video.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from pydub import AudioSegment

SEGMENTS_DIR = Path("output/segments")
PLAN_PATH = Path("output/illustration_plan.json")
ILLUSTRATIONS_DIR = Path("output/illustrations")
AUDIO_PATH = Path("output/full_volume_bgm.mp3")
OUTPUT_PATH = Path("output/illustration_video.mp4")

# 拼接参数（与 splicer.py 一致）
GAP_DIALOGUE = 300    # 对话间隔 ms
GAP_PARAGRAPH = 1000  # 段落间隔 ms
GAP_CHAPTER = 2000    # 章节间隔 ms
FADE_DURATION = 50    # 淡入淡出 ms


def get_segment_duration(wav_path: Path) -> int:
    """返回音频片段时长（毫秒）"""
    try:
        seg = AudioSegment.from_file(str(wav_path))
        return len(seg)
    except Exception:
        return 1000  # fallback


def build_line_timestamps() -> list[int]:
    """计算每行对应的音频开始时间戳（毫秒）"""
    wav_files = sorted(SEGMENTS_DIR.glob("*.wav"))
    if not wav_files:
        print("错误: 没有找到音频片段")
        sys.exit(1)

    print(f"音频片段: {len(wav_files)} 个")

    # 计算每个片段的时长和时间戳
    seg_starts = []
    seg_durations = []
    offset = 0
    for i, wav in enumerate(wav_files):
        dur = get_segment_duration(wav)
        seg_starts.append(offset)
        seg_durations.append(dur)
        # 片段间间隔（与 splicer 一致）
        gap = GAP_DIALOGUE
        offset += dur + gap

    # 每个片段对应一行（按文件索引）
    # segments 文件名 00000.wav = line 1, 00001.wav = line 2, ...
    # 但小说总行数可能大于片段数（空行/标题行无音频）
    return seg_starts


def build_illustration_timeline(
    illustrations: list[dict],
    line_timestamps: list[int],
) -> list[dict]:
    """将插图映射到时间轴。

    每张插图根据 start_line 和 end_line 决定显示时间段。
    如果 start_line 对应的时间戳不可用，用前一个可用的时间戳。
    """
    timeline = []
    for p in illustrations:
        sl = p.get("start_line", 1)
        el = p.get("end_line", 1)
        title = p.get("title", "?")

        # 找到行号对应的时间戳
        start_ms = line_timestamps[sl - 1] if sl <= len(line_timestamps) else 0
        end_ms = line_timestamps[el - 1] if el <= len(line_timestamps) else start_ms + 3000
        # 加上片段本身的时长
        if el < len(line_timestamps):
            end_ms += 300  # 加一个间隔让画面停留

        if end_ms <= start_ms:
            end_ms = start_ms + 3000  # 至少显示 3 秒

        timeline.append({
            "title": title,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "image_idx": p.get("image_idx", 0),
        })

    return timeline


def main():
    print("=" * 50)
    print("插图视频生成")
    print("=" * 50)

    # 1. 加载数据
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    illustrations = plan.get("illustrations", plan) if isinstance(plan, dict) else plan
    print(f"插图: {len(illustrations)} 张")

    # 2. 检查插图文件
    img_files = sorted(ILLUSTRATIONS_DIR.glob("*.png"))
    print(f"插图文件: {len(img_files)} 个")
    if not img_files:
        print("错误: 没有找到插图文件")
        return 1

    # 建立 插图索引 -> 文件路径 映射
    for i, p in enumerate(illustrations):
        # 尝试按序号匹配
        expected = ILLUSTRATIONS_DIR / f"{i+1:04d}_{p.get('title','?')}.png"
        if expected.exists():
            p["image_idx"] = i
            p["image_path"] = str(expected)
        else:
            # fallback: 按顺序匹配
            p["image_path"] = str(img_files[i]) if i < len(img_files) else ""

    # 3. 计算时间戳
    print("\n计算时间戳...")
    line_ts = build_line_timestamps()
    total_duration_ms = line_ts[-1] + 1000 if line_ts else 21943000  # fallback to 6h
    print(f"音频总时长: {total_duration_ms / 1000 / 60:.0f} 分钟")

    # 4. 构建插图时间轴
    timeline = build_illustration_timeline(illustrations, line_ts)
    print(f"时间轴条目: {len(timeline)}")

    # 5. 生成 ffmpeg concat 文件
    print("\n生成视频...")
    audio_dur_s = 0
    try:
        import subprocess
        r = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(AUDIO_PATH.resolve()),
        ], capture_output=True, text=True, timeout=30)
        audio_dur_s = float(r.stdout.strip())
    except Exception:
        audio_dur_s = 21943.0

    print(f"音频时长: {audio_dur_s/60:.0f} 分钟")

    # 生成 concat 文件
    from PIL import Image
    concat_file = Path("output/_video_concat.txt").resolve()
    total_video_dur = 0
    with open(concat_file, "w", encoding="utf-8") as f:
        for item in timeline:
            img_path = illustrations[item["image_idx"]].get("image_path", "")
            if not img_path or not Path(img_path).exists():
                continue
            dur_s = max(3, item["duration_ms"] / 1000)
            abs_path = Path(img_path).resolve().as_posix()
            f.write(f"file '{abs_path}'\nduration {dur_s:.3f}\n")
            total_video_dur += dur_s

    # concat demuxer 会忽略最后一张图的 duration
    # 解决方法：在末尾加占位图 + dummy 图
    gap = audio_dur_s - total_video_dur
    print(f"  插图总长: {total_video_dur/60:.0f} 分")
    print(f"  需补足: {max(0,gap)/60:.0f} 分")

    if gap > 3:
        # 生成占位图 (896x1152 黑色)
        placeholder = Image.new("RGB", (896, 1152), (0, 0, 0))
        plc_path = Path("output/_placeholder.png").resolve()
        placeholder.save(str(plc_path))
        # 生成 dummy 图 (1x1 黑色)
        dummy = Image.new("RGB", (1, 1), (0, 0, 0))
        dmy_path = Path("output/_dummy.png").resolve()
        dummy.save(str(dmy_path))

        with open(concat_file, "a", encoding="utf-8") as f:
            f.write(f"file '{plc_path.as_posix()}'\n")
            f.write(f"duration {gap:.3f}\n")
            f.write(f"file '{dmy_path.as_posix()}'\n")
            f.write("duration 0.040\n")
        print(f"  已添加占位图 + dummy ({gap:.0f}s)")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-i", str(AUDIO_PATH.resolve()),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=896:1152",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(OUTPUT_PATH),
    ]

    print(f"  concat 文件: {concat_file}")

    # 6. 执行 ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-i", str(AUDIO_PATH.resolve()),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=896:1152",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(OUTPUT_PATH),
    ]

    print(f"  输出: {OUTPUT_PATH}")
    print(f"  命令: {' '.join(cmd)}")
    print("\n渲染中...")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
        print(f"\n完成! ({elapsed:.0f} 秒, {size_mb:.0f} MB)")
        print(f"输出: {OUTPUT_PATH}")
    else:
        err_text = result.stderr.decode("utf-8", errors="replace")[:500] if result.stderr else "(no stderr)"
        print(f"\n错误: {err_text}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())