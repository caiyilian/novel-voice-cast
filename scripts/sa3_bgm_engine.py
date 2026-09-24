"""Stable Audio 3 引擎适配层：为 novel-voice-cast 流水线提供 BGM 生成能力。

与 run_bgm_generate.py（ACE-Step 版）保持相同的输出约定：
    output/bgm/<index:03d>_<clip>.mp3  +  bgm_manifest.json

Stable Audio 3 的关键差异：
  - 纯器乐、44.1kHz 立体声、可变长度（medium 6m20s / small 2m）
  - prompt 用长段落自然语言（官方 Prompt Guide），无 500 字符硬限
  - 无 cfg 概念（post-trained 模型），steps=8 是原生步数
  - 输出 24-bit wav，需自行编码 mp3

用法：
    python scripts/sa3_bgm_engine.py --segments backend/data/bgm_segments.json \
        --output-dir output/bgm --duration 90 --clips-per-segment 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))  # bgm_segmenter 依赖 app.* 包路径
SA3_ROOT = Path(os.environ.get("SA3_ROOT", "E:/projects/stable-audio-3"))
SA3_PYTHON = SA3_ROOT / ".venv" / "Scripts" / "python.exe"
if not SA3_PYTHON.is_file():  # Linux
    SA3_PYTHON = SA3_ROOT / ".venv" / "bin" / "python"

try:
    from app.core.bgm_segmenter import INSTRUMENTATION_POOLS
except Exception:  # 独立运行时允许降级（不阻断音频生成）
    INSTRUMENTATION_POOLS: dict[str, tuple[str, ...]] = {
        "daily": ("solo nylon-string guitar with light shaker",),
    }

BGM_GENERATION_VERSION = 5
SA3_MODEL = "small-music"

DEFAULT_SEGMENTS = PROJECT_ROOT / "backend" / "data" / "bgm_segments.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "bgm"
DEFAULT_DURATION = 90.0
DEFAULT_CLIPS_PER_SEGMENT = 4
DEFAULT_STEPS = 8


def load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("segments", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(f"无法从 {path} 解析 segments")


def build_clip_seed(segment_index: int, clip_index: int) -> int:
    """与 ACE-Step 版同构的稳定 seed，保证可复现。"""
    return int(segment_index) * 10_007 + int(clip_index) * 1_009 + 17


def inject_local_weights(model_name: str, ckpt_dir: Path) -> None:
    """把本地权重按 HF 缓存布局注入，使 hf_hub_download 直接命中缓存（离线可用）。

    用「文件名 + 大小」生成 commit 标识，不读取文件内容——在 RAID 存储上
    对 8.6G 权重做 sha256 会耗时数分钟。
    """
    import hashlib
    import shutil

    from huggingface_hub import constants

    repo_id = f"stabilityai/stable-audio-3-{model_name}"
    repo_dir = Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    files = ["model_config.json", "model.safetensors"]
    sub = ckpt_dir / "t5gemma-b-b-ul2"
    if sub.is_dir():
        files += [f"t5gemma-b-b-ul2/{p.name}" for p in sorted(sub.iterdir()) if p.is_file()]
    present = [f for f in files if (ckpt_dir / f).is_file()]
    signature = "|".join(f"{f}:{(ckpt_dir / f).stat().st_size}" for f in present)
    commit = hashlib.sha256(signature.encode()).hexdigest()
    snapshot = repo_dir / "snapshots" / commit
    blobs = repo_dir / "blobs"
    for d in (snapshot, blobs, repo_dir / "refs"):
        d.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")
    count = 0
    for name in present:
        src = ckpt_dir / name
        blob = blobs / Path(name).name
        if not blob.exists():
            try:
                os.link(src, blob)
            except OSError:
                shutil.copy2(src, blob)
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(blob)
        except OSError:
            shutil.copy2(blob, target)
        count += 1
    print(f"  本地权重已注入缓存: {repo_id} ({count} 文件, commit {commit[:12]})", flush=True)


def clip_filename(segment_index: int, clip_index: int) -> str:
    return f"{int(segment_index):03d}_{int(clip_index)}.mp3"


def diversify_prompts(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨段去重：为每个场景补一段「差异化编制」描述，避免相邻段落音色雷同。

    6 小时以上的长音频里，最大的听感杀手是「连续若干段同一套乐器 + 同一节奏型」。
    做法：
      1. 按 bgm_type 分组，组内轮转 INSTRUMENTATION_POOLS，保证同情绪不同编制；
      2. 若相邻段落 texturn 相同，则强制换下一个编制（并微调 BPM）；
      3. 把结果追加到 prompt 尾部（Stable Audio 3 对详细描述响应好）。
    """
    out = [dict(s) for s in segments]
    pool_cursor: dict[str, int] = {}
    last_instrumentation: str | None = None
    for global_pos, segment in enumerate(out, 1):
        # 记录全局序号，供分片运行时保持文件名与 mixer 的 {seg_idx+1:03d} 约定一致
        segment["_global_index"] = global_pos
        bgm_type = str(segment.get("bgm_type", "")) or "daily"
        pool = INSTRUMENTATION_POOLS.get(bgm_type) or INSTRUMENTATION_POOLS["daily"]
        size = len(pool)
        cursor = pool_cursor.get(bgm_type, 0) % size
        instrumentation = pool[cursor]
        # 只有恰好与上一段撞车时才再跳一位，保证步长为 1、池成员充分利用
        if instrumentation == last_instrumentation and size > 1:
            cursor = (cursor + 1) % size
            instrumentation = pool[cursor]
        pool_cursor[bgm_type] = cursor + 1
        last_instrumentation = instrumentation

        prompt = str(segment.get("bgm_music_prompt", "")).strip()
        # 已有的 LLM instrumentation 若和轮转结果不同，以轮转结果为准（保证多样性）
        segment["sa3_instrumentation"] = instrumentation
        if prompt:
            prompt = f"{prompt.rstrip(' .')} Feature {instrumentation}."
        else:
            prompt = f"Instrumental underscore featuring {instrumentation}."
        segment["bgm_music_prompt"] = prompt
    return out


def _quality_metrics(audio: Any, sample_rate: int) -> tuple[float, float]:
    """低频谱平坦度 + 谐波/打击乐比，用于识别噪声/打击乐主导的失败产物。

    与 ACE-Step 版保持一致的口径，便于横向比较。
    """
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import resample_poly, stft

    samples = np.asarray(audio, dtype=np.float32)
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples.reshape(-1)
    if sample_rate != 8_000:
        mono = resample_poly(mono, 8_000, int(sample_rate))
    _, _, spectrum = stft(mono, fs=8_000, nperseg=1_024, noverlap=768, boundary=None)
    magnitude = np.abs(spectrum).astype(np.float64) + 1e-10
    power = np.square(magnitude) + 1e-12
    frame_flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    spectral_flatness = float(np.mean(frame_flatness))

    harmonic = median_filter(magnitude, size=(1, 31), mode="nearest")
    percussive = median_filter(magnitude, size=(31, 1), mode="nearest")
    harmonic_power = np.square(harmonic)
    percussive_power = np.square(percussive)
    mask = harmonic_power / (harmonic_power + percussive_power + 1e-12)
    separated_harmonic = np.square(magnitude * mask).sum()
    separated_percussive = np.square(magnitude * (1.0 - mask)).sum()
    harmonicity_db = float(
        10.0 * np.log2((separated_harmonic + 1e-10) / (separated_percussive + 1e-10)) * 3.0103
    )
    return spectral_flatness, harmonicity_db


def validate_clip(path: Path, expected_duration: float) -> tuple[bool, str]:
    """拒绝截断/静音/噪声主导的产物。"""
    import numpy as np
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as exc:
        return False, f"无法读取音频: {exc}"
    duration = info.frames / info.samplerate
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        return False, "音频流元数据无效"
    if abs(duration - expected_duration) > max(2.0, expected_duration * 0.04):
        return False, f"时长不符: {duration:.2f}s vs {expected_duration:.2f}s"
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(rms) or not np.isfinite(peak):
        return False, "音频含非有限采样值"
    if rms < 0.001 or peak < 0.01:
        return False, f"音频近似静音 (rms={rms:.6f}, peak={peak:.6f})"
    flatness, harmonicity_db = _quality_metrics(audio, info.samplerate)
    if flatness > 0.010 and harmonicity_db < 10.0:
        return False, (
            f"噪声/打击乐主导 (spectral_flatness={flatness:.5f}, harmonicity={harmonicity_db:.2f}dB)"
        )
    return True, (
        f"duration={duration:.2f}s rms={rms:.5f} peak={peak:.5f} "
        f"flatness={flatness:.5f} harmonicity={harmonicity_db:.2f}dB"
    )


def main() -> int:
    # 默认值从 config.yaml 的 bgm 段读取，保持单一配置源
    cfg_bgm: dict[str, Any] = {}
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if cfg_path.is_file():
        try:
            import yaml

            cfg_bgm = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("bgm", {}) or {}
        except Exception:
            cfg_bgm = {}

    parser = argparse.ArgumentParser(description="用 Stable Audio 3 生成 BGM（每场景多条款式）")
    parser.add_argument("--segments", default=str(DEFAULT_SEGMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--duration", type=float, default=float(cfg_bgm.get("clip_duration", DEFAULT_DURATION)))
    parser.add_argument(
        "--clips-per-segment",
        type=int,
        default=int(cfg_bgm.get("clips_per_segment", DEFAULT_CLIPS_PER_SEGMENT)),
    )
    parser.add_argument("--steps", type=int, default=int(cfg_bgm.get("steps", DEFAULT_STEPS)))
    parser.add_argument("--model", default=str(cfg_bgm.get("model", SA3_MODEL)))
    parser.add_argument("--force", action="store_true", help="忽略已有产物重新生成")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个场景（调试用）")
    parser.add_argument("--shard", default=None, help="分片 i/N（分布式用，如 0/3）")
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="本地权重目录（含 model_config.json/model.safetensors/t5gemma-b-b-ul2）；"
        "提供后先注入 HF 缓存再加载，适合服务器离线环境",
    )
    args = parser.parse_args()

    segments_path = Path(args.segments)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "bgm_manifest.json"

    segments = load_segments(segments_path)
    if args.limit:
        segments = segments[: args.limit]
    # 跨段差异化：轮转乐器编制，避免相邻段落音色雷同
    # 注意：必须在分片前做，否则各分片的轮转状态不连续
    segments = diversify_prompts(segments)
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        segments = segments[shard_i::shard_n]
        print(f"分片 {shard_i}/{shard_n} -> {len(segments)} 个场景", flush=True)
    print(f"场景数: {len(segments)}，每场景 {args.clips_per_segment} 条 × {args.duration:.0f}s", flush=True)

    # 让 stable_audio_3 能 import（它装在独立 venv 里）
    sys.path.insert(0, str(SA3_ROOT))
    import torchaudio

    from stable_audio_3 import StableAudioModel

    # 若提供了本地权重目录，先注入 HF 缓存布局再加载。
    # 目的：stable_audio_3 的 ModelConfig.resolve() 只走 hf_hub_download，
    # 而官方仓库在 HF 上是 gated（未授权会 403）。用本地权重离线加载最可靠。
    # 注意：用「文件大小之和」做 commit 标识，避免读取 8.6G 文件算 sha256（RAID 上极慢）。
    if args.ckpt_dir:
        inject_local_weights(args.model, Path(args.ckpt_dir))

    print(f"加载模型 {args.model} …", flush=True)
    t0 = time.time()
    model = StableAudioModel.from_pretrained(args.model)
    print(f"模型就绪（{time.time()-t0:.1f}s），设备 {model.device}", flush=True)
    sample_rate = model.model.sample_rate

    # 分片运行时各进程写独立 manifest，最后再合并，避免并发覆盖
    if args.shard:
        shard_tag = args.shard.replace("/", "_")
        manifest_path = out_dir / f"bgm_manifest.shard{shard_tag}.json"
    manifest: dict[str, Any] = {
        "generation_version": BGM_GENERATION_VERSION,
        "engine": "stable-audio-3",
        "model": args.model,
        "steps": args.steps,
        "duration_per_segment": args.duration,
        "clips_per_segment": args.clips_per_segment,
        "total_segments": len(segments),
        "segments": {},
    }
    if manifest_path.is_file() and not args.force:
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("generation_version") == BGM_GENERATION_VERSION:
                manifest["segments"] = old.get("segments", {})
        except Exception:
            pass

    done = 0
    failed: list[str] = []
    t_start = time.time()
    for position, segment in enumerate(segments, 1):
        # 场景序号必须是全局位置（1-based），分片时也要用原始顺序，
        # 否则文件名会与其它分片冲突、且与 mixer 的 {seg_idx+1:03d} 约定不符。
        index = int(segment.get("_global_index", segment.get("segment_index", position)))
        prompt = str(segment.get("bgm_music_prompt", "")).strip()
        if not prompt:
            prompt = str(segment.get("bgm_evidence", "")).strip() or "cinematic ambient underscore"
        key = str(index)
        entry = manifest["segments"].get(key, {})
        clips_meta: list[dict[str, Any]] = list(entry.get("clips", []))
        have = {c.get("clip_index") for c in clips_meta}

        print(f"\n[{position}/{len(segments)}] 场景 {index}  {prompt[:70]}…", flush=True)
        for clip_index in range(args.clips_per_segment):
            name = clip_filename(index, clip_index)
            target = out_dir / name
            if not args.force and clip_index in have and target.is_file():
                print(f"    · {name} 已存在，跳过", flush=True)
                continue
            seed = build_clip_seed(index, clip_index)
            t1 = time.time()
            audio = model.generate(
                prompt=prompt,
                duration=args.duration,
                steps=args.steps,
                seed=seed,
            )
            tmp = out_dir / f".{name}.{uuid.uuid4().hex}.tmp.wav"
            torchaudio.save(str(tmp), audio[0].cpu(), sample_rate)
            valid, detail = validate_clip(tmp, args.duration)
            if not valid:
                tmp.unlink(missing_ok=True)
                print(f"    [REJECT] {name}: {detail}", flush=True)
                failed.append(name)
                continue
            # 编码为 mp3（与下游 bgm-mix 的输入约定一致）
            mp3_tmp = tmp.with_suffix(".tmp.mp3")
            rc = os.system(
                f'ffmpeg -y -v error -i "{tmp}" -b:a 256k -ar 48000 "{mp3_tmp}"'
            )
            tmp.unlink(missing_ok=True)
            if rc != 0 or not mp3_tmp.is_file():
                print(f"    [FAIL] {name}: mp3 编码失败", flush=True)
                failed.append(name)
                continue
            os.replace(mp3_tmp, target)
            clips_meta = [c for c in clips_meta if c.get("clip_index") != clip_index]
            clips_meta.append(
                {
                    "clip_index": clip_index,
                    "file": name,
                    "seed": seed,
                    "duration": args.duration,
                    "quality_validated": True,
                    "qc": detail,
                }
            )
            clips_meta.sort(key=lambda c: c.get("clip_index", 0))
            print(f"    ✓ {name}  ({time.time()-t1:.1f}s)  {detail}", flush=True)
            done += 1
        manifest["segments"][key] = {
            "index": index,
            "bgm_type": segment.get("bgm_type", "unknown"),
            "prompt": prompt,
            "clips": clips_meta,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    print(f"\n=== 完成：{done} 条，耗时 {elapsed:.1f}s ({elapsed/60:.1f} min) ===", flush=True)
    if failed:
        print(f"=== 失败 {len(failed)} 条: {failed[:10]} ===", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
