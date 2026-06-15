"""
Novel Voice Cast - 完整流程
所有对话都用 VoxCPM 音色克隆
"""
import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# 添加 backend 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.core.parser import parse
from app.core.splicer import AudioSplicer
from pydub import AudioSegment


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def get_reference_audio(speaker: str, gender: str, config: dict) -> str:
    """获取角色的参考音频"""
    # 旁白和空speaker使用旁白音频
    if not speaker or speaker == "旁白":
        return config.get("characters", {}).get("旁白", "backend/data/presets/design_male_deep.wav")

    # 优先使用配置中指定的音频
    if speaker in config.get("characters", {}):
        return config["characters"][speaker]

    # 使用默认音频（根据性别）
    default = config.get("default_audio", {})
    if gender == "female":
        return default.get("female", "backend/data/presets/reference_speaker.wav")
    else:
        return default.get("male", "backend/data/presets/design_elderly.wav")


def build_emotion_prefix(emotion: str = None, tone: str = None) -> str:
    """构建情感前缀"""
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
        return "(" + "，".join(parts) + ")"
    return ""


def step_parse(config: dict):
    """步骤1：解析小说"""
    t0 = time.time()
    novel_path = config["novel"]["text_path"]
    labels_path = config["novel"]["labels_path"]

    with open(novel_path, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    dialogues, characters = parse(novel_text, labels)
    elapsed = time.time() - t0

    print(f"  解析完成: {len(dialogues)} 条对话, {len(characters)} 个角色 [{elapsed:.2f}s]")
    return dialogues, characters, novel_text


def step_gender(config: dict, characters: list, dialogues: list, novel_text: str) -> dict:
    """步骤2：性别识别"""
    gender_path = Path("backend/data/gender_results.json")

    if gender_path.exists():
        t0 = time.time()
        with open(gender_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        elapsed = time.time() - t0
        print(f"  加载已有结果: {len(results)} 个角色 [{elapsed:.2f}s]")
        return results

    from app.core.gender_identifier import identify_gender
    from app.core.ollama_client import OllamaClient

    t0 = time.time()
    client = OllamaClient()
    results = {}

    for i, char_name in enumerate(characters):
        text_parts = []
        for d in dialogues:
            if d.get("speaker") == char_name:
                text_parts.append(d["text"])
        char_text = "\n".join(text_parts[:50])

        try:
            result = identify_gender(char_name, char_text, client, max_tool_steps=5)
            results[char_name] = {
                "gender": result["gender"],
                "confidence": result["confidence"],
            }
            print(f"\r  [{i+1:2d}/{len(characters)}] {char_name}: {result['gender']}   ", end="", flush=True)
        except Exception as e:
            results[char_name] = {"gender": "male", "confidence": 0.3}
            print(f"\r  [{i+1:2d}/{len(characters)}] {char_name}: error   ", end="", flush=True)

    print()
    elapsed = time.time() - t0

    gender_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gender_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  识别完成: {len(results)} 个角色 [{elapsed:.2f}s]")
    return results


def step_emotion(config: dict, dialogues: list, novel_text: str, force_reprocess: bool = False) -> dict:
    """步骤3：情感标注（旁白跳过）"""
    emotion_path = Path("backend/data/emotion_results.json")

    # 如果不强制重新处理，且已有结果，直接加载
    if not force_reprocess and emotion_path.exists():
        t0 = time.time()
        with open(emotion_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        elapsed = time.time() - t0
        print(f"  加载已有结果: {len(results)} 条 [{elapsed:.2f}s]")
        return results

    from app.core.emotion_labeler import label_emotion
    from app.core.ollama_client import OllamaClient

    t0 = time.time()
    client = OllamaClient()
    results = {}
    total = len(dialogues)
    need_emotion = sum(1 for d in dialogues if d.get("speaker") and d.get("speaker") != "旁白")
    processed = 0
    skipped = 0

    print(f"  需要标注: {need_emotion} 条 (跳过旁白和非人物)")

    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")

        # 旁白或空speaker跳过情感标注
        if speaker == "旁白" or not speaker:
            skipped += 1
            continue

        text = dialogue["text"]
        line_num = dialogue.get("line", i + 1)

        try:
            result = label_emotion(
                dialogue_text=text,
                dialogue_line=line_num,
                dialogue_index=i,
                text=novel_text,
                client=client,
                max_tool_steps=5,
            )
            results[str(i)] = {
                "emotion": result.get("emotion", "calm"),
                "tone": result.get("tone", "serious"),
            }
        except Exception as e:
            results[str(i)] = {"emotion": "calm", "tone": "serious"}

        processed += 1

        # 进度显示 - 只计算需要标注的对话
        elapsed = time.time() - t0
        avg = elapsed / processed if processed > 0 else 0
        remaining = avg * (need_emotion - processed)
        print(f"\r  [{processed:4d}/{need_emotion}] 已标注 {processed:4d} | 剩余 ~{format_time(remaining)}   ", end="", flush=True)

    print()
    elapsed = time.time() - t0

    emotion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(emotion_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    print(f"  标注完成: {len(results)} 条 [{elapsed:.2f}s]")
    return results


def step_tts(config: dict, dialogues: list, gender_results: dict, emotion_results: dict) -> list:
    """步骤4：TTS 合成（全部用 VoxCPM）"""
    t0 = time.time()
    total = len(dialogues)

    # 准备输出目录
    output_dir = Path(config["output"]["dir"])
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # 构建任务列表
    tasks = []
    skipped_existing = 0
    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")
        text = dialogue["text"]
        chapter = dialogue.get("chapter", "unknown")

        # 检查是否已存在音频文件
        filename = f"{i:05d}.wav"
        output_path = str(segments_dir / filename)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            skipped_existing += 1
            continue

        # 获取性别
        gender_info = gender_results.get(speaker, {"gender": "male"})
        gender = gender_info.get("gender", "male")

        # 获取参考音频
        ref_audio = get_reference_audio(speaker, gender, config)
        ref_audio_path = os.path.join(os.path.dirname(__file__), ref_audio)

        # 获取情感标注（旁白和空speaker跳过）
        emotion_prefix = ""
        if speaker and speaker != "旁白":
            emo = emotion_results.get(str(i), {})
            emotion_prefix = build_emotion_prefix(emo.get("emotion"), emo.get("tone"))

        tasks.append({
            "index": i,
            "text": text,
            "reference_audio": ref_audio_path,
            "emotion_prefix": emotion_prefix,
            "chapter": chapter,
            "speaker": speaker,
            "output_path": str(segments_dir / f"{i:05d}.wav"),
        })

    print(f"  准备合成 {len(tasks)} 条对话... (跳过已有: {skipped_existing})")

    # 创建 VoxCPM 批量处理脚本
    script = create_voxcpm_script(tasks, config)
    script_path = output_dir / "_batch_voxcpm.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    # 调用 VoxCPM
    voxcpm_python = r"E:\projects\音色克隆\VoxCPM\.venv\python.exe"
    print(f"  开始 VoxCPM 批量合成...")

    # 实时读取进度
    process = subprocess.Popen(
        [voxcpm_python, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    # 实时读取 stderr 显示进度
    while True:
        line = process.stderr.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            print(f"\r  {line.strip()}   ", end="", flush=True)

    process.wait()
    result = subprocess.CompletedProcess(process.args, process.returncode, "", "")

    script_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"\n  VoxCPM 错误: {result.stderr[:500]}")
        # 降级：用 pyttsx3 合成失败的部分
        print("  降级使用 pyttsx3 合成...")
        return fallback_pyttsx3(tasks, segments_dir)

    # 读取结果
    results_file = output_dir / "voxcpm_results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        elapsed = time.time() - t0
        print(f"  合成完成: {len(results)} 条 [{format_time(elapsed)}]")
    else:
        results = []

    # 构建 segments 列表（包含所有对话，不仅仅是本次合成的）
    segments = []
    for i, dialogue in enumerate(dialogues):
        speaker = dialogue.get("speaker", "")
        chapter = dialogue.get("chapter", "unknown")
        output_path = str(segments_dir / f"{i:05d}.wav")
        segments.append({
            "audio_path": output_path,
            "chapter": chapter,
            "order": i,
            "speaker": speaker,
        })

    return segments


def create_voxcpm_script(tasks: list, config: dict) -> str:
    """创建 VoxCPM 批量处理脚本"""
    tasks_json = json.dumps(tasks, ensure_ascii=False)
    model_path = os.path.join(os.path.dirname(__file__), 'backend', 'models', 'VoxCPM2')
    results_path = os.path.join(os.path.dirname(__file__), 'output', 'voxcpm_results.json')

    script = f'''
import sys
import json
import os
import time
sys.path.insert(0, r"E:\\projects\\音色克隆\\VoxCPM")
from voxcpm import VoxCPM
import soundfile as sf

def main():
    # 读取任务列表
    tasks = json.loads(r"""{tasks_json}""")

    # 加载模型（只加载一次）
    print("加载 VoxCPM 模型...", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        r"{model_path}",
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
        output_path = task["output_path"]

        # 拼接情感前缀 + 文本
        full_text = f"{{emotion_prefix}}{{text}}" if emotion_prefix else text
        full_text = full_text.replace('"', '\\"').replace("\\n", " ")

        try:
            wav = model.generate(
                text=full_text,
                reference_wav_path=ref_audio,
                cfg_value=2.0,
                inference_timesteps=10,
            )

            # 保存到文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, wav, model.tts_model.sample_rate)
            results.append({{"index": index, "status": "ok"}})
        except Exception as e:
            results.append({{"index": index, "status": "error", "error": str(e)}})

        # 进度
        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            print(f"  [{{i+1}}/{{len(tasks)}}]", file=sys.stderr)

    # 保存结果
    with open(r"{results_path}", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成: {{len(results)}} 条", file=sys.stderr)

if __name__ == "__main__":
    main()
'''
    return script


def fallback_pyttsx3(tasks: list, segments_dir: Path) -> list:
    """降级：用 pyttsx3 合成"""
    import pyttsx3
    import tempfile

    engine = pyttsx3.init()
    results = []

    for i, task in enumerate(tasks):
        output_path = task["output_path"]
        text = task["text"]

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            engine.stop()

            with open(temp_path, "rb") as f:
                audio_bytes = f.read()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            os.remove(temp_path)
            results.append({"index": task["index"], "status": "ok"})
        except Exception as e:
            results.append({"index": task["index"], "status": "error"})

        if (i + 1) % 50 == 0 or i == len(tasks) - 1:
            print(f"\r  [pyttsx3] [{i+1:4d}/{len(tasks)}]   ", end="", flush=True)

    print()
    return results


def step_splice(config: dict, segments: list) -> str:
    """步骤5：音频拼接"""
    t0 = time.time()

    output_dir = config["output"]["dir"]
    filename = config["output"]["filename"]
    fmt = config["output"].get("format", "mp3")
    bitrate = config["output"].get("bitrate", "64k")

    output_path = str(Path(output_dir) / f"{filename}.{fmt}")
    print(f"  输出格式: {fmt} ({bitrate})")

    splicer = AudioSplicer()
    final_audio = splicer.splice(segments, output_path=output_path)

    # 获取文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    duration = len(final_audio) / 1000.0

    elapsed = time.time() - t0
    print(f"  保存完成 [{elapsed:.2f}s]")
    print(f"  文件大小: {file_size:.1f} MB")

    return output_path, duration


def main():
    parser = argparse.ArgumentParser(description="Novel Voice Cast")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N条对话（0=全部）")
    parser.add_argument("--range", type=str, default="", help="处理对话范围，如 100-200")
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 60)
    print("Novel Voice Cast - 完整流程")
    print("=" * 60)

    # 加载配置
    config = load_config(args.config)
    print(f"\n配置文件: {args.config}")
    if args.limit > 0:
        print(f"限制: 只处理前 {args.limit} 条对话")
    elif args.range:
        print(f"范围: 处理第 {args.range} 条对话")

    # 步骤1：解析
    print(f"\n[1/4] 解析小说")
    dialogues, characters, novel_text = step_parse(config)

    # 应用限制或范围
    if args.limit > 0:
        dialogues = dialogues[:args.limit]
        characters = list(set(d.get("speaker", "") for d in dialogues if d.get("speaker")))
        print(f"  限制模式: 只处理 {len(dialogues)} 条对话, {len(characters)} 个角色")
    elif args.range:
        parts = args.range.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            dialogues = dialogues[start:end]
            characters = list(set(d.get("speaker", "") for d in dialogues if d.get("speaker")))
            print(f"  范围模式: 处理第 {start}-{end} 条对话, 共 {len(dialogues)} 条, {len(characters)} 个角色")

            # 输出 debug.txt
            debug_path = Path(config["output"]["dir"]) / "debug.txt"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"对话范围: {start}-{end}\n")
                f.write(f"共 {len(dialogues)} 条对话, {len(characters)} 个角色\n")
                f.write("=" * 60 + "\n\n")
                for i, d in enumerate(dialogues):
                    f.write(f"[{start + i}] {d.get('speaker', '?')}: {d.get('text', '')}\n")
            print(f"  debug 输出: {debug_path}")

    # 步骤2：性别识别
    print(f"\n[2/4] 性别识别")
    gender_results = step_gender(config, characters, dialogues, novel_text)

    # 步骤3：情感标注
    print(f"\n[3/4] 情感标注")
    emotion_enabled = config.get("features", {}).get("emotion_label", True)
    force_reprocess = config.get("features", {}).get("force_reprocess", False)

    if not emotion_enabled:
        print("  跳过（已禁用）")
        emotion_results = {}
    else:
        emotion_results = step_emotion(config, dialogues, novel_text, force_reprocess=force_reprocess)

    # 步骤4：TTS 合成
    print(f"\n[4/4] TTS 合成")
    segments = step_tts(config, dialogues, gender_results, emotion_results)

    # 步骤5：音频拼接
    print(f"\n[5/5] 音频拼接")
    output_path, duration = step_splice(config, segments)

    # 汇总
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  输出: {output_path}")
    print(f"  时长: {format_time(duration)}")
    print(f"  总耗时: {format_time(total_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
