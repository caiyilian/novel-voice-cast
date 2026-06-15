"""
可视化音频波形 - 查看 holo.mp3 的语音分布
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# 加载音频（前60秒）
y, sr = librosa.load(str(Path(__file__).parent.parent / "backend" / "data" / "presets" / "holo.mp3"), duration=60)

print(f"采样率: {sr}")
print(f"时长: {len(y)/sr:.1f}s")
print(f"形状: {y.shape}")

# 计算包络
hop_length = 512
frame_length = 2048

# RMS 能量
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# 1. 波形
axes[0].set_title("Waveform (First 60s)")
librosa.display.waveshow(y, sr=sr, ax=axes[0], color='blue', alpha=0.7)
axes[0].set_xlabel("Time (s)")

# 2. RMS 能量
axes[1].set_title("RMS Energy")
axes[1].plot(times, rms, color='green')
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Energy")

# 3. 能量阈值判断语音区域
threshold = 0.01
speech_mask = rms > threshold
axes[2].set_title(f"Speech Detection (threshold={threshold})")
axes[2].fill_between(times, 0, speech_mask, color='orange', alpha=0.5)
axes[2].plot(times, rms, color='green', alpha=0.5)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Speech")

plt.tight_layout()
plt.savefig(str(Path(__file__).parent.parent / "output" / "holo_waveform.png"), dpi=150)
print(f"\n图片已保存: output/holo_waveform.png")

# 统计语音区域
speech_regions = []
in_speech = False
start = 0
for i, is_speech in enumerate(speech_mask):
    if is_speech and not in_speech:
        start = times[i]
        in_speech = True
    elif not is_speech and in_speech:
        speech_regions.append((start, times[i]))
        in_speech = False
if in_speech:
    speech_regions.append((start, times[-1]))

print(f"\n语音区域统计:")
print(f"  总区域数: {len(speech_regions)}")
for i, (start, end) in enumerate(speech_regions):
    print(f"  区域 {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")

# 找最长的连续语音区域
if speech_regions:
    longest = max(speech_regions, key=lambda x: x[1] - x[0])
    print(f"\n最长语音区域: {longest[0]:.1f}s - {longest[1]:.1f}s ({longest[1]-longest[0]:.1f}s)")
