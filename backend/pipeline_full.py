"""
小说转语音 - 完整流程脚本
使用预计算的性别结果，跳过 LLM 识别
"""
import asyncio
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from app.core.parser import parse
from app.core.splicer import AudioSplicer
from pydub import AudioSegment

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent.parent
NOVEL_PATH = BASE_DIR / "novels" / "novel.txt"
LABELS_PATH = BASE_DIR / "novels" / "labels.txt"
GENDER_PATH = BASE_DIR / "backend" / "data" / "gender_results.json"
OUTPUT_DIR = BASE_DIR / "output"
AUDIO_DIR = OUTPUT_DIR / "segments"

# TTS 配置
VOXCMP_CHARACTERS = ["劳伦斯", "罗伦斯", "赫萝"]
MALE_VOICE = "zh-CN-YunxiNeural"
FEMALE_VOICE = "zh-CN-XiaoxiaoNeural"

# 参考音频
REFERENCE_AUDIO = {
    "劳伦斯": str(BASE_DIR / "backend" / "data" / "presets" / "reference_speaker.wav"),
    "罗伦斯": str(BASE_DIR / "backend" / "data" / "presets" / "reference_speaker.wav"),
    "赫萝": str(BASE_DIR / "backend" / "data" / "presets" / "reference_speaker.wav"),
}


def load_gender_results() -> dict:
    with open(GENDER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_voice_config(char_name: str, gender: str) -> dict:
    if char_name in VOXCMP_CHARACTERS:
        ref_audio = REFERENCE_AUDIO.get(char_name)
        if ref_audio and os.path.exists(ref_audio):
            return {"engine": "voxcpm", "reference_audio": ref_audio}
    voice_id = MALE_VOICE if gender == "male" else FEMALE_VOICE
    return {"engine": "edge-tts", "voice_id": voice_id}


def save_audio(audio_bytes: bytes, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def synthesize_edge_tts_sync(text: str, voice_id: str) -> bytes:
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, voice_id)
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        return audio_data.getvalue()
    return asyncio.run(_run())


def synthesize_voxcpm_sync(text: str, reference_audio: str) -> bytes:
    escaped_text = text.replace('"', '\\"').replace('\n', ' ')
    script = f'''
import sys
sys.path.insert(0, r"E:\\projects\\音色克隆\\VoxCPM")
from voxcpm import VoxCPM
import soundfile as sf
import io

model = VoxCPM.from_pretrained(
    r"E:\\projects\\novel-voice-cast\\backend\\models\\VoxCPM2",
    load_denoiser=False,
)
wav = model.generate(
    text="{escaped_text}",
    reference_wav_path=r"{reference_audio}",
    cfg_value=2.0,
    inference_timesteps=10,
)
buf = io.BytesIO()
sf.write(buf, wav, model.tts_model.sample_rate, format="WAV")
sys.stdout.buffer.write(buf.getvalue())
'''
    temp_script = OUTPUT_DIR / "_temp_voxcpm.py"
    temp_script.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script)

    voxcpm_python = r"E:\projects\音色克隆\VoxCPM\.venv\python.exe"
    result = subprocess.run(
        [voxcpm_python, str(temp_script)],
        capture_output=True,
        timeout=120,
    )
    temp_script.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"VoxCPM error: {result.stderr.decode('utf-8', errors='ignore')}")
    return result.stdout


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    total_start = time.time()
    timings = {}

    print("=" * 60)
    print("小说转语音 - 完整流程")
    print("=" * 60)

    # 1. 读取文件
    t0 = time.time()
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    timings["读取文件"] = time.time() - t0
    print(f"\n[1/5] 读取文件: {timings['读取文件']:.2f}s")
    print(f"  小说: {len(novel_text.splitlines())} 行")
    print(f"  标注: {len(labels)} 条")

    # 2. 解析对话
    t0 = time.time()
    dialogues, characters = parse(novel_text, labels)
    timings["解析对话"] = time.time() - t0
    print(f"\n[2/5] 解析对话: {timings['解析对话']:.2f}s")
    print(f"  对话: {len(dialogues)} 条")
    print(f"  角色: {len(characters)} 个")

    # 3. 加载性别结果
    t0 = time.time()
    gender_results = load_gender_results()
    timings["加载性别"] = time.time() - t0
    print(f"\n[3/5] 加载性别结果: {timings['加载性别']:.2f}s")

    # 4. 构建角色配置
    char_voice_config = {}
    print(f"\n[4/5] TTS 配置:")
    for char_name in characters:
        gender_info = gender_results.get(char_name, {"gender": "male"})
        gender = gender_info["gender"]
        if gender == "unknown":
            gender = "male"
        config = get_voice_config(char_name, gender)
        char_voice_config[char_name] = config

        engine = config["engine"]
        if engine == "voxcpm":
            print(f"  {char_name}: VoxCPM (克隆)")
        else:
            voice_name = "男声" if "Yunxi" in config.get("voice_id", "") else "女声"
            print(f"  {char_name}: edge-tts ({voice_name})")

    # 5. 合成所有对话
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    segments = []
    total_dialogues = len(dialogues)
    success_count = 0
    fail_count = 0

    print(f"\n[5/5] 开始合成 {total_dialogues} 条对话...")

    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")
        text = dialogue["text"]
        chapter = dialogue.get("chapter", "unknown")

        voice_config = char_voice_config.get(speaker, {
            "engine": "edge-tts",
            "voice_id": MALE_VOICE,
        })

        filename = f"{i:05d}_{speaker}.wav"
        output_path = str(AUDIO_DIR / filename)

        try:
            if voice_config["engine"] == "voxcpm":
                audio_bytes = synthesize_voxcpm_sync(text, voice_config["reference_audio"])
            else:
                audio_bytes = synthesize_edge_tts_sync(text, voice_config["voice_id"])
            save_audio(audio_bytes, output_path)

            segments.append({
                "audio_path": output_path,
                "chapter": chapter,
                "order": i,
                "speaker": speaker,
            })
            success_count += 1

        except Exception as e:
            fail_count += 1
            silence = AudioSegment.silent(duration=1000)
            silence.export(output_path, format="wav")
            segments.append({
                "audio_path": output_path,
                "chapter": chapter,
                "order": i,
                "speaker": speaker,
            })

        # 进度显示 - 每条更新
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (total_dialogues - i - 1)
        engine_tag = "V" if voice_config["engine"] == "voxcpm" else "E"
        print(f"\r  [{i+1:4d}/{total_dialogues}] {engine_tag} "
              f"成功 {success_count:4d} 失败 {fail_count:2d} | "
              f"已用 {format_time(elapsed)} 剩余 ~{format_time(remaining)}   ", end="", flush=True)

    timings["TTS合成"] = time.time() - t0
    print()  # 换行
    print(f"\n[TTS] 完成: {format_time(timings['TTS合成'])}")
    print(f"  成功: {success_count} 条")
    print(f"  失败: {fail_count} 条")

    # 6. 拼接整卷音频
    t0 = time.time()
    splicer = AudioSplicer()
    final_output = str(OUTPUT_DIR / "full_volume.wav")
    final_audio = splicer.splice(segments, output_path=final_output)
    duration = splicer.get_duration(final_audio)
    timings["音频拼接"] = time.time() - t0

    print(f"\n[拼接] 完成: {format_time(timings['音频拼接'])}")
    print(f"  时长: {format_time(duration)}")

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("耗时汇总")
    print("=" * 60)
    for step, t in timings.items():
        print(f"  {step}: {format_time(t)}")
    print(f"  {'─' * 30}")
    print(f"  总耗时: {format_time(total_time)}")
    print("=" * 60)
    print(f"\n输出: {final_output}")
    print(f"音频时长: {format_time(duration)}")


if __name__ == "__main__":
    main()
