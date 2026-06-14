"""
TTS 速度对比测试
VoxCPM vs IndexTTS2 vs edge-tts
"""
import asyncio
import io
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# ========== 配置 ==========
TEST_TEXT = "你好，今天天气真不错，我们一起去散步吧。"
TEST_COUNT = 10
VOXCPM_VENV = r"E:\projects\音色克隆\VoxCPM\.venv\python.exe"
VOXCPM_MODEL = r"E:\projects\novel-voice-cast\backend\models\VoxCPM2"
REFERENCE_AUDIO = r"E:\projects\novel-voice-cast\backend\data\presets\reference_speaker.wav"
OUTPUT_DIR = r"E:\projects\novel-voice-cast\output\speed_test"


def test_edge_tts():
    """测试 edge-tts"""
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(TEST_TEXT, "zh-CN-YunxiNeural")
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        return audio_data.getvalue()

    t0 = time.time()
    for i in range(TEST_COUNT):
        asyncio.run(_run())
    elapsed = time.time() - t0
    return elapsed


def test_voxcpm():
    """测试 VoxCPM"""
    escaped_text = TEST_TEXT.replace('"', '\\"').replace('\n', ' ')
    script = f'''
import sys
sys.path.insert(0, r"E:\\projects\\音色克隆\\VoxCPM")
from voxcpm import VoxCPM
import soundfile as sf
import io

model = VoxCPM.from_pretrained(r"{VOXCPM_MODEL}", load_denoiser=False)
for i in range({TEST_COUNT}):
    wav = model.generate(
        text="{escaped_text}",
        reference_wav_path=r"{REFERENCE_AUDIO}",
        cfg_value=2.0,
        inference_timesteps=10,
    )
    buf = io.BytesIO()
    sf.write(buf, wav, model.tts_model.sample_rate, format="WAV")
print("done")
'''
    temp_script = os.path.join(OUTPUT_DIR, "_temp_voxcpm.py")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script)

    t0 = time.time()
    result = subprocess.run(
        [VOXCPM_VENV, temp_script],
        capture_output=True,
        timeout=300,
    )
    elapsed = time.time() - t0
    temp_script and os.remove(temp_script)
    return elapsed


def test_indextts2():
    """测试 IndexTTS2"""
    script = f'''
import sys
sys.path.insert(0, r"E:\\projects\\indextts")
from indextts import IndexTTS
import io

model = IndexTTS(model_path=r"E:\\projects\\novel-voice-cast\\backend\\models\\checkpoints\\IndexTTS-2")
for i in range({TEST_COUNT}):
    wav = model.infer(
        spk_audio_prompt=r"{REFERENCE_AUDIO}",
        text="{TEST_TEXT}",
    )
print("done")
'''
    temp_script = os.path.join(OUTPUT_DIR, "_temp_indextts2.py")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script)

    indextts_python = r"E:\projects\indextts\.venv\Scripts\python.exe"
    t0 = time.time()
    result = subprocess.run(
        [indextts_python, temp_script],
        capture_output=True,
        timeout=300,
    )
    elapsed = time.time() - t0
    temp_script and os.remove(temp_script)
    return elapsed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("TTS 速度对比测试")
    print(f"测试文本: {TEST_TEXT}")
    print(f"测试次数: {TEST_COUNT}")
    print("=" * 60)

    # edge-tts
    print("\n[edge-tts]")
    edge_time = test_edge_tts()
    print(f"  总耗时: {edge_time:.2f}s")
    print(f"  平均: {edge_time/TEST_COUNT:.2f}s/条")

    # VoxCPM
    print("\n[VoxCPM]")
    voxcpm_time = test_voxcpm()
    print(f"  总耗时: {voxcpm_time:.2f}s")
    print(f"  平均: {voxcpm_time/TEST_COUNT:.2f}s/条")

    # IndexTTS2
    print("\n[IndexTTS2]")
    indextts_time = test_indextts2()
    print(f"  总耗时: {indextts_time:.2f}s")
    print(f"  平均: {indextts_time/TEST_COUNT:.2f}s/条")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"  edge-tts:   {edge_time:.2f}s ({edge_time/TEST_COUNT:.2f}s/条)")
    print(f"  VoxCPM:     {voxcpm_time:.2f}s ({voxcpm_time/TEST_COUNT:.2f}s/条)")
    print(f"  IndexTTS2:  {indextts_time:.2f}s ({indextts_time/TEST_COUNT:.2f}s/条)")
    print("=" * 60)


if __name__ == "__main__":
    main()
