"""
提取 holo.mp3 的语音部分，拼接成 30 秒参考音频
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment

# 加载音频
y, sr = librosa.load(str(Path(__file__).parent.parent / "backend" / "data" / "presets" / "holo.mp3"), duration=120)

# 检测语音区域
hop_length = 512
frame_length = 2048
threshold = 0.01

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# 找语音区域
speech_regions = []
in_speech = False
start = 0
for i, is_speech in enumerate(rms > threshold):
    if is_speech and not in_speech:
        start = times[i]
        in_speech = True
    elif not is_speech and in_speech:
        speech_regions.append((start, times[i]))
        in_speech = False
if in_speech:
    speech_regions.append((start, times[-1]))

print(f"检测到 {len(speech_regions)} 个语音区域")

# 合并相邻区域（间隔<0.3s的合并）
merged = []
for start, end in speech_regions:
    if merged and start - merged[-1][1] < 0.3:
        merged[-1] = (merged[-1][0], end)
    else:
        merged.append((start, end))

print(f"合并后 {len(merged)} 个区域")

# 计算总语音时长
total_speech = sum(end - start for start, end in merged)
print(f"总语音时长: {total_speech:.1f}s")

# 提取语音部分
audio = AudioSegment.from_mp3(str(Path(__file__).parent.parent / "backend" / "data" / "presets" / "holo.mp3"))

# 添加停顿（100ms）
pause = AudioSegment.silent(duration=100)

speech_audio = AudioSegment.empty()
for i, (start, end) in enumerate(merged):
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    speech_audio += audio[start_ms:end_ms]
    # 片段之间添加停顿（最后一个不加）
    if i < len(merged) - 1:
        speech_audio += pause

# 如果超过30秒，截取前30秒
if len(speech_audio) > 30000:
    speech_audio = speech_audio[:30000]

print(f"输出时长: {len(speech_audio)/1000:.1f}s")

# 保存
output_path = str(Path(__file__).parent.parent / "backend" / "data" / "presets" / "holo_speech_only.wav")
speech_audio.export(output_path, format="wav")
print(f"已保存: {output_path}")

# 统计
print(f"\n统计:")
print(f"  原始时长: 120s")
print(f"  语音时长: {total_speech:.1f}s")
print(f"  静音时长: {120 - total_speech:.1f}s")
print(f"  语音占比: {total_speech/120*100:.1f}%")
