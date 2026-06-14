"""
VoxCPM 批量处理方案
方案1: 文件中转（VoxCPM 保存音频到文件，主脚本读取）
方案2: 持久化服务（VoxCPM 作为常驻服务，通过 HTTP 接收请求）

两种方案都支持情感标注（自然语言控制情绪）
"""
import json
import os
import sys
import time
from pathlib import Path

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent.parent
VOXCPM_MODEL = str(BASE_DIR / "backend" / "models" / "VoxCPM2")
VOXCPM_PYTHON = r"E:\projects\音色克隆\VoxCPM\.venv\python.exe"
VOXCPM_SCRIPT_DIR = r"E:\projects\音色克隆\VoxCPM"
AUDIO_DIR = BASE_DIR / "output" / "voxcpm_batch"


# ========== 情感映射 ==========

# emotion + tone → 自然语言描述（VoxCPM 使用中文前缀控制情感）
def build_emotion_prefix(emotion: str = None, tone: str = None) -> str:
    """构建情感前缀，用于 VoxCPM 文本前缀"""
    emotion_map = {
        "happy": "欢快活泼",
        "sad": "低落悲伤",
        "angry": "愤怒生气",
        "surprised": "惊讶震惊",
        "calm": "平静冷静",
        "nervous": "紧张焦虑",
        "cold": "冷漠淡漠",
    }
    tone_map = {
        "loud": "大声",
        "soft": "轻声",
        "whisper": "低语",
        "gentle": "温柔",
        "serious": "严肃",
        "sarcastic": "讽刺",
        "stutter": "结巴",
    }

    parts = []
    if emotion and emotion in emotion_map:
        parts.append(emotion_map[emotion])
    if tone and tone in tone_map:
        parts.append(tone_map[tone])

    if parts:
        return "（" + "，".join(parts) + "）"
    return ""


# ========== 方案1: 文件中转 ==========

def create_batch_file_script():
    """创建 VoxCPM 批量处理脚本（文件中转方式）"""
    script = '''
import sys
import json
import os
import time
sys.path.insert(0, r"E:\\projects\\音色克隆\\VoxCPM")
from voxcpm import VoxCPM
import soundfile as sf

def main():
    # 读取任务列表
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # 加载模型（只加载一次）
    print("加载 VoxCPM 模型...", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        r"E:\\projects\\novel-voice-cast\\backend\\models\\VoxCPM2",
        load_denoiser=False,
    )
    print("模型加载完成", file=sys.stderr)

    # 处理每个任务
    results = []
    for i, task in enumerate(tasks):
        index = task["index"]
        text = task["text"]
        ref_audio = task["reference_audio"]
        emotion_prefix = task.get("emotion_prefix", "")

        # 拼接情感前缀 + 文本
        full_text = f"{emotion_prefix}{text}" if emotion_prefix else text
        full_text = full_text.replace('"', '\\\\"').replace('\\n', ' ')

        try:
            wav = model.generate(
                text=full_text,
                reference_wav_path=ref_audio,
                cfg_value=2.0,
                inference_timesteps=10,
            )

            # 保存到文件
            output_path = os.path.join(output_dir, f"{index:05d}.wav")
            sf.write(output_path, wav, model.tts_model.sample_rate)
            results.append({"index": index, "status": "ok", "path": output_path})
        except Exception as e:
            results.append({"index": index, "status": "error", "error": str(e)})

        # 进度
        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            print(f"  [{i+1}/{len(tasks)}]", file=sys.stderr)

    # 输出结果
    print(json.dumps(results))

if __name__ == "__main__":
    main()
'''
    return script


def run_batch_file_method(tasks: list, emotion_data: dict = None) -> dict:
    """
    方案1: 文件中转
    tasks: [{"index": 0, "text": "...", "reference_audio": "..."}, ...]
    emotion_data: {"0": {"emotion": "happy", "tone": "gentle"}, ...}
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # 为每个任务添加情感前缀
    for task in tasks:
        idx = str(task["index"])
        if emotion_data and idx in emotion_data:
            emo = emotion_data[idx]
            task["emotion_prefix"] = build_emotion_prefix(
                emo.get("emotion"), emo.get("tone")
            )
        else:
            task["emotion_prefix"] = ""

    # 保存任务列表
    input_file = AUDIO_DIR / "tasks.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    # 保存批量处理脚本
    script_content = create_batch_file_script()
    script_path = AUDIO_DIR / "batch_process.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    # 调用 VoxCPM Python 执行
    import subprocess
    print(f"\n  [方案1] 文件中转: 处理 {len(tasks)} 条...")
    t0 = time.time()

    result = subprocess.run(
        [VOXCPM_PYTHON, str(script_path), str(input_file), str(AUDIO_DIR)],
        capture_output=True,
        timeout=1800,  # 30 分钟超时
    )

    elapsed = time.time() - t0
    print(f"  [方案1] 完成 [{elapsed:.1f}s]")

    if result.returncode != 0:
        print(f"  错误: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
        return {}

    # 读取结果
    try:
        results = json.loads(result.stdout.decode("utf-8"))
        return {r["index"]: r for r in results}
    except Exception as e:
        print(f"  解析结果失败: {e}")
        return {}


# ========== 方案2: 持久化服务 ==========

def create_voxcpm_server_script():
    """创建 VoxCPM 持久化服务脚本"""
    script = '''
import sys
import json
import os
import time
import http.server
import urllib.parse
sys.path.insert(0, r"E:\\projects\\音色克隆\\VoxCPM")
from voxcpm import VoxCPM
import soundfile as sf
import io
import base64

class VoxCPMHandler(http.server.BaseHTTPRequestHandler):
    model = None

    def do_POST(self):
        if self.path == "/synthesize":
            self.handle_synthesize()
        elif self.path == "/shutdown":
            self.handle_shutdown()
        else:
            self.send_error(404)

    def handle_synthesize(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        task = json.loads(body.decode("utf-8"))

        text = task["text"]
        ref_audio = task["reference_audio"]
        emotion_prefix = task.get("emotion_prefix", "")

        full_text = f"{emotion_prefix}{text}" if emotion_prefix else text
        full_text = full_text.replace('"', '\\\\"").replace("\\n", " ")

        try:
            wav = self.model.generate(
                text=full_text,
                reference_wav_path=ref_audio,
                cfg_value=2.0,
                inference_timesteps=10,
            )

            # 转为 base64
            buf = io.BytesIO()
            sf.write(buf, wav, self.model.tts_model.sample_rate, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            response = {"status": "ok", "audio": audio_b64}
        except Exception as e:
            response = {"status": "error", "error": str(e)}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def handle_shutdown(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
        # 延迟关闭
        import threading
        threading.Timer(0.5, lambda: os._exit(0)).start()

    def log_message(self, format, *args):
        pass  # 禁用日志

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

    print(f"加载 VoxCPM 模型...", flush=True)
    VoxCPMHandler.model = VoxCPM.from_pretrained(
        r"E:\\projects\\novel-voice-cast\\backend\\models\\VoxCPM2",
        load_denoiser=False,
    )
    print(f"模型加载完成，服务启动于 port {port}", flush=True)

    server = http.server.HTTPServer(("127.0.0.1", port), VoxCPMHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
'''
    return script


class VoxCPMClient:
    """VoxCPM 服务客户端"""

    def __init__(self, port: int = 8765):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._process = None

    def start_server(self):
        """启动 VoxCPM 服务"""
        import subprocess

        script_content = create_voxcpm_server_script()
        script_path = AUDIO_DIR / "voxcpm_server.py"
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        self._process = subprocess.Popen(
            [VOXCPM_PYTHON, str(script_path), str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 等待服务启动
        import urllib.request
        for i in range(30):
            time.sleep(1)
            try:
                urllib.request.urlopen(self.base_url, timeout=1)
                print(f"  VoxCPM 服务已启动 (port {self.port})")
                return True
            except Exception:
                continue

        print("  VoxCPM 服务启动超时")
        return False

    def synthesize(self, text: str, reference_audio: str,
                   emotion: str = None, tone: str = None) -> bytes:
        """调用 VoxCPM 服务合成"""
        import urllib.request
        import base64

        emotion_prefix = build_emotion_prefix(emotion, tone)

        task = {
            "text": text,
            "reference_audio": reference_audio,
            "emotion_prefix": emotion_prefix,
        }

        req = urllib.request.Request(
            f"{self.base_url}/synthesize",
            data=json.dumps(task).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if result["status"] == "ok":
            return base64.b64decode(result["audio"])
        else:
            raise RuntimeError(f"VoxCPM error: {result.get('error')}")

    def shutdown(self):
        """关闭 VoxCPM 服务"""
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.base_url}/shutdown", timeout=5)
        except Exception:
            pass
        if self._process:
            self._process.wait(timeout=10)


def run_batch_server_method(tasks: list, emotion_data: dict = None) -> dict:
    """
    方案2: 持久化服务
    tasks: [{"index": 0, "text": "...", "reference_audio": "..."}, ...]
    emotion_data: {"0": {"emotion": "happy", "tone": "gentle"}, ...}
    """
    client = VoxCPMClient()

    # 启动服务
    if not client.start_server():
        return {}

    results = {}
    try:
        print(f"\n  [方案2] 持久化服务: 处理 {len(tasks)} 条...")
        t0 = time.time()

        for i, task in enumerate(tasks):
            idx = task["index"]
            text = task["text"]
            ref_audio = task["reference_audio"]

            # 获取情感标注
            emo = emotion_data.get(str(idx), {}) if emotion_data else {}

            try:
                audio_bytes = client.synthesize(
                    text, ref_audio,
                    emotion=emo.get("emotion"),
                    tone=emo.get("tone"),
                )

                # 保存到文件
                output_path = AUDIO_DIR / f"{idx:05d}.wav"
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

                results[idx] = {"status": "ok", "path": str(output_path)}
            except Exception as e:
                results[idx] = {"status": "error", "error": str(e)}

            # 进度
            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                elapsed = time.time() - t0
                avg = elapsed / (i + 1)
                remaining = avg * (len(tasks) - i - 1)
                print(f"\r    [{i+1:4d}/{len(tasks)}] 已用 {elapsed:.1f}s 剩余 ~{remaining:.1f}s   ", end="", flush=True)

        print()
        elapsed = time.time() - t0
        print(f"  [方案2] 完成 [{elapsed:.1f}s]")

    finally:
        client.shutdown()

    return results


# ========== 测试 ==========

if __name__ == "__main__":
    # 测试数据
    test_tasks = [
        {"index": 0, "text": "你好，今天天气真不错。", "reference_audio": "backend/data/presets/reference_speaker.wav"},
        {"index": 1, "text": "我们一起去散步吧！", "reference_audio": "backend/data/presets/reference_speaker.wav"},
    ]
    test_emotion = {
        "0": {"emotion": "happy", "tone": "gentle"},
        "1": {"emotion": "calm", "tone": "serious"},
    }

    print("=" * 60)
    print("VoxCPM 批量处理测试")
    print("=" * 60)

    # 测试方案1
    print("\n[方案1] 文件中转")
    results1 = run_batch_file_method(test_tasks, test_emotion)
    print(f"  结果: {results1}")

    # 测试方案2
    print("\n[方案2] 持久化服务")
    results2 = run_batch_server_method(test_tasks, test_emotion)
    print(f"  结果: {results2}")
