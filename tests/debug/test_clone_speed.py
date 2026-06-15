"""
VoxCPM2 vs IndexTTS2 速度对比测试
"""
import json
import os
import subprocess
import sys
import time

# ========== 配置 ==========
BASE_DIR = os.path.dirname(__file__)
TEST_TEXT = "你好，今天天气真不错，我们一起去散步吧。"
TEST_COUNT = 10
REFERENCE_AUDIO = os.path.join(BASE_DIR, "backend", "data", "presets", "reference_speaker.wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "speed_test")
VOXCPM_PYTHON = r"E:\projects\音色克隆\VoxCPM\.venv\python.exe"
VOXCPM_MODEL = os.path.join(BASE_DIR, "backend", "models", "VoxCPM2")
INDextts_PYTHON = r"E:\projects\indextts\.venv\Scripts\python.exe"
INDextts_MODEL = os.path.join(BASE_DIR, "backend", "models", "checkpoints", "IndexTTS-2")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}min"


def test_voxcpm():
    """测试 VoxCPM2"""
    escaped_text = TEST_TEXT.replace('"', '\\"').replace('\n', ' ')
    script = f'''
import sys
import json
import time
sys.path.insert(0, r"E:\\projects\\音色克隆\\VoxCPM")
from voxcpm import VoxCPM
import soundfile as sf
import io

# 加载模型（只加载一次）
print("加载 VoxCPM2 模型...", file=sys.stderr)
t0 = time.time()
model = VoxCPM.from_pretrained(
    r"{VOXCPM_MODEL}",
    load_denoiser=False,
)
load_time = time.time() - t0
print(f"模型加载: {{load_time:.2f}}s", file=sys.stderr)

# 合成 10 条
results = []
for i in range({TEST_COUNT}):
    t0 = time.time()
    wav = model.generate(
        text="{escaped_text}",
        reference_wav_path=r"{REFERENCE_AUDIO}",
        cfg_value=2.0,
        inference_timesteps=10,
    )
    elapsed = time.time() - t0
    results.append({{"index": i, "time": elapsed}})
    print(f"  [{{i+1}}/{TEST_COUNT}] {{elapsed:.2f}}s", file=sys.stderr)

# 输出结果
print(json.dumps({{"load_time": load_time, "results": results}}))
'''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    script_path = os.path.join(OUTPUT_DIR, "_test_voxcpm.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    print("=" * 60)
    print("VoxCPM2 测试")
    print("=" * 60)

    t0 = time.time()
    result = subprocess.run(
        [VOXCPM_PYTHON, script_path],
        capture_output=True,
        timeout=300,
    )
    total_time = time.time() - t0

    os.remove(script_path)

    if result.returncode != 0:
        print(f"错误: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
        return None

    try:
        data = json.loads(result.stdout.decode("utf-8"))
        print(f"\n模型加载: {data['load_time']:.2f}s")
        print(f"合成测试:")
        for r in data["results"]:
            print(f"  [{r['index']+1}/{TEST_COUNT}] {r['time']:.2f}s")
        avg_time = sum(r["time"] for r in data["results"]) / len(data["results"])
        print(f"\n平均: {avg_time:.2f}s/条")
        print(f"总计: {total_time:.2f}s")
        return {"load_time": data["load_time"], "avg_time": avg_time, "total_time": total_time}
    except Exception as e:
        print(f"解析失败: {e}")
        return None


def test_indextts2():
    """测试 IndexTTS2"""
    script = f'''
import sys
import json
import time
sys.path.insert(0, r"E:\\projects\\indextts")
from indextts.infer import IndexTTS

# 加载模型（只加载一次）
print("加载 IndexTTS2 模型...", file=sys.stderr)
t0 = time.time()
model = IndexTTS(
    cfg_path=r"{INDextts_MODEL}\\config.yaml",
    model_dir=r"{INDextts_MODEL}",
)
load_time = time.time() - t0
print(f"模型加载: {{load_time:.2f}}s", file=sys.stderr)

# 合成 10 条
results = []
for i in range({TEST_COUNT}):
    t0 = time.time()
    wav = model.infer(
        spk_audio_prompt=r"{REFERENCE_AUDIO}",
        text="{TEST_TEXT}",
    )
    elapsed = time.time() - t0
    results.append({{"index": i, "time": elapsed}})
    print(f"  [{{i+1}}/{TEST_COUNT}] {{elapsed:.2f}}s", file=sys.stderr)

# 输出结果
print(json.dumps({{"load_time": load_time, "results": results}}))
'''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    script_path = os.path.join(OUTPUT_DIR, "_test_indextts2.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    print("\n" + "=" * 60)
    print("IndexTTS2 测试")
    print("=" * 60)

    t0 = time.time()
    result = subprocess.run(
        [INDextts_PYTHON, script_path],
        capture_output=True,
        timeout=300,
    )
    total_time = time.time() - t0

    os.remove(script_path)

    if result.returncode != 0:
        print(f"错误: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
        return None

    try:
        data = json.loads(result.stdout.decode("utf-8"))
        print(f"\n模型加载: {data['load_time']:.2f}s")
        print(f"合成测试:")
        for r in data["results"]:
            print(f"  [{r['index']+1}/{TEST_COUNT}] {r['time']:.2f}s")
        avg_time = sum(r["time"] for r in data["results"]) / len(data["results"])
        print(f"\n平均: {avg_time:.2f}s/条")
        print(f"总计: {total_time:.2f}s")
        return {"load_time": data["load_time"], "avg_time": avg_time, "total_time": total_time}
    except Exception as e:
        print(f"解析失败: {e}")
        return None


def main():
    # 只测试 IndexTTS2
    indextts_result = test_indextts2()

    # 汇总
    print("\n" + "=" * 60)
    print("IndexTTS2 结果")
    print("=" * 60)
    if indextts_result:
        print(f"模型加载: {indextts_result['load_time']:.2f}s")
        print(f"平均合成: {indextts_result['avg_time']:.2f}s/条")
        print(f"总耗时: {indextts_result['total_time']:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
