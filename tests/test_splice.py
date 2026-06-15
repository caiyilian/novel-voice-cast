"""
音频拼接脚本 - 把 output/segments 拼接成整卷音频
"""
import os
import sys
import time
from pathlib import Path

# 添加 backend 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.splicer import AudioSplicer

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent.parent
SEGMENTS_DIR = BASE_DIR / "output" / "segments"
OUTPUT_WAV = BASE_DIR / "output" / "full_volume.wav"
OUTPUT_MP3 = BASE_DIR / "output" / "full_volume.mp3"


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

    # 加载片段信息（从目录读取）
    t0 = time.time()
    wav_files = sorted(SEGMENTS_DIR.glob("*.wav"), key=lambda x: int(x.stem))
    segments = [{"audio_path": str(f), "order": int(f.stem)} for f in wav_files]
    print(f"\n[1/2] 加载片段: {len(segments)} 条 [{time.time()-t0:.2f}s]")

    # 拼接
    t0 = time.time()
    total = len(segments)

    print(f"\n[2/2] 开始拼接 {total} 个片段...")

    splicer = AudioSplicer()

    # 手动拼接以显示进度
    from pydub import AudioSegment

    sorted_segments = sorted(segments, key=lambda x: x.get("order", 0))
    audio_segments = []
    loaded = 0
    failed = 0

    for i, seg in enumerate(sorted_segments):
        try:
            audio = AudioSegment.from_file(seg["audio_path"])
            audio_segments.append(audio)
            loaded += 1
        except Exception as e:
            failed += 1
            print(f"\n  警告: 加载失败 {seg['audio_path']}: {e}")

        # 进度显示
        if (i + 1) % 100 == 0 or i == total - 1:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remaining = avg * (total - i - 1)
            print(f"\r  [{i+1:4d}/{total}] 已加载 {loaded:4d} 失败 {failed:2d} | 剩余 ~{format_time(remaining)}   ", end="", flush=True)

    print()

    if not audio_segments:
        print("\n错误: 没有可用的音频片段")
        return

    # 拼接音频
    print(f"\n  拼接 {len(audio_segments)} 个片段...")
    t0 = time.time()

    result = audio_segments[0]
    for i in range(1, len(audio_segments)):
        result = result + splicer._create_silence(splicer.gap_dialogue) + audio_segments[i]

        if (i + 1) % 200 == 0 or i == len(audio_segments) - 1:
            elapsed = time.time() - t0
            print(f"\r  拼接进度: [{i+1:4d}/{len(audio_segments)}] | 已用 {format_time(elapsed)}   ", end="", flush=True)

    print()

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)
    result.export(str(OUTPUT_WAV), format="wav")
    result.export(str(OUTPUT_MP3), format="mp3", bitrate="64k")

    duration = len(result) / 1000.0
    wav_size = os.path.getsize(OUTPUT_WAV) / (1024 * 1024)
    mp3_size = os.path.getsize(OUTPUT_MP3) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\n  保存完成 [{elapsed:.2f}s]")

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  WAV: {OUTPUT_WAV} ({wav_size:.1f} MB)")
    print(f"  MP3: {OUTPUT_MP3} ({mp3_size:.1f} MB)")
    print(f"  时长: {format_time(duration)}")
    print(f"  总耗时: {format_time(total_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
