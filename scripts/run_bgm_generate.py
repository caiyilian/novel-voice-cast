r"""
Stage 7-c: BGM generation per segment via ACE-Step SDK.

⚠️  Run with ACE-Step venv's Python:
    ACE-Step-1.5\.venv\Scripts\python.exe run_bgm_generate.py

Usage:
    ACE-Step-1.5\.venv\Scripts\python.exe run_bgm_generate.py [--force] [--duration 30]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── Path setup: add ACE-Step project to sys.path ──────────────────────
ACE_DIR = Path(__file__).resolve().parent.parent / "ACE-Step-1.5"
PROJECT_ROOT = ACE_DIR.parent  # One level up: the novel-voice-cast root
sys.path.insert(0, str(ACE_DIR))

# ACE-Step SDK imports are lazy (inside _init_ace_step) so that dry-run
# mode does not trigger CUDA initialisation.
os.environ.setdefault("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
os.environ.setdefault("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
os.environ.setdefault("ACESTEP_DEVICE", "auto")
os.environ.setdefault("ACESTEP_INIT_LLM", "false")
os.environ.setdefault("ACESTEP_CPU_OFFLOAD", "false")
os.environ.setdefault("ACESTEP_OFFLOAD_DIT_TO_CPU", "false")

BGM_PROCESS_RESTART_EXIT_CODE = 75
DEFAULT_PROCESS_CLIP_LIMIT = 16


def _env_flag(name: str, default: bool) -> bool:
    fallback = "true" if default else "false"
    return os.environ.get(name, fallback).lower() not in {"0", "false", "no"}


def _resolve_offload_policy() -> tuple[bool, bool]:
    cpu_offload = _env_flag("ACESTEP_CPU_OFFLOAD", False)
    offload_dit = cpu_offload and _env_flag("ACESTEP_OFFLOAD_DIT_TO_CPU", False)
    return cpu_offload, offload_dit


def _install_model_offload_sync_guard(
    handler: Any,
    synchronize: Callable[[], None] | None = None,
) -> bool:
    """Synchronize CUDA work before and after moving DiT back to CPU.

    ACE-Step decodes semantic audio codes inside a VAE residency context.  With
    DiT offloading enabled this means the DiT is moved to CUDA, used by the
    tokenizer/detokenizer, and immediately moved back to CPU while the VAE is
    still resident.  CUDA kernels are asynchronous; on Windows the repeated
    unsynchronized transition can terminate the process with 0xC0000005 rather
    than raising a catchable Python exception.
    """
    if not (
        getattr(handler, "offload_to_cpu", False)
        and getattr(handler, "offload_dit_to_cpu", False)
    ):
        return False
    if getattr(handler, "_novel_voice_cast_offload_sync_guard", False):
        return True

    if synchronize is None:
        import torch

        if not torch.cuda.is_available():
            return False
        synchronize = torch.cuda.synchronize

    original_context = handler._load_model_context

    @contextmanager
    def synchronized_context(model_name: str):
        with original_context(model_name):
            try:
                yield
            finally:
                if model_name == "model":
                    synchronize()
        if model_name == "model":
            synchronize()

    handler._load_model_context = synchronized_context
    handler._novel_voice_cast_offload_sync_guard = True
    return True


def _init_ace_step(
    model: str | None = None,
    lm_model: str | None = None,
    lm_backend: str | None = None,
) -> tuple[Any, Any]:
    """Initialize the quality DiT and 5Hz semantic LM with explicit checks."""
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    model = model or os.environ.get("ACESTEP_CONFIG_PATH", ACE_STEP_MODEL)
    lm_model = lm_model or os.environ.get("ACESTEP_LM_MODEL_PATH", ACE_STEP_LM_MODEL)
    lm_backend = lm_backend or os.environ.get("ACESTEP_LM_BACKEND", "vllm")
    cpu_offload, offload_dit = _resolve_offload_policy()
    dit_handler = AceStepHandler()
    status, success = dit_handler.initialize_service(
        project_root=str(ACE_DIR),
        config_path=model,
        device="auto",
        # The 1.7B vLLM backend remains resident on the 12 GB RTX 3060. Move
        # the large DiT back to CPU before VAE decode as well; otherwise VRAM
        # fragmentation eventually leaves <0.5 GB and ACE-Step falls back to
        # an hours-long CPU VAE decode.
        offload_to_cpu=cpu_offload,
        offload_dit_to_cpu=offload_dit,
    )
    if not success:
        raise RuntimeError(f"ACE-Step DiT initialization failed: {status}")
    if _install_model_offload_sync_guard(dit_handler):
        print("  CUDA synchronization guard enabled for DiT offload", flush=True)

    if os.environ.get("ACESTEP_INIT_LLM", "false").lower() in {"0", "false", "no"}:
        return dit_handler, None
    llm_handler = LLMHandler()
    status, success = llm_handler.initialize(
        checkpoint_dir=str(ACE_DIR / "checkpoints"),
        lm_model_path=lm_model,
        backend=lm_backend,
        device="auto",
        offload_to_cpu=False,
        dtype=None,
    )
    if not success:
        raise RuntimeError(f"ACE-Step 5Hz LM initialization failed: {status}")
    return dit_handler, llm_handler

# ── Project paths (relative to project root, one level up from ACE_DIR) ──
PROJECT_ROOT = ACE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.bgm_generator import (
    ACE_STEP_MODEL,
    ACE_STEP_LM_MODEL,
    BGM_GENERATION_VERSION,
    build_bgm_seed,
    build_manifest,
    build_segment_bgm_prompt,
    load_segments,
    save_manifest,
)

# Resolve defaults relative to project root (os.chdir has moved us to ACE_DIR)
_DEFAULT_SEGMENTS = str(PROJECT_ROOT / "backend/data/bgm_segments.json")
_DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output/bgm")

# ── Defaults ──
DEFAULT_DURATION = 60       # seconds per BGM clip
DEFAULT_INFERENCE_STEPS = 8  # turbo model's native step count
MAX_NOISE_SPECTRAL_FLATNESS = 0.010
MIN_NOISE_HARMONICITY_DB = 10.0


def _music_quality_metrics(audio: Any, sample_rate: int) -> tuple[float, float]:
    """Return low-band spectral flatness and harmonic/percussive balance.

    The failed SFT+5Hz run produced broadband, footstep-like pulses that still
    passed duration/RMS checks.  These two inexpensive measurements separate
    that failure mode from the known-good turbo music while avoiding any
    content-dependent genre classifier.
    """
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import resample_poly, stft

    samples = np.asarray(audio, dtype=np.float32)
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples.reshape(-1)
    if sample_rate != 8_000:
        mono = resample_poly(mono, 8_000, int(sample_rate))
    _, _, spectrum = stft(
        mono,
        fs=8_000,
        nperseg=1_024,
        noverlap=768,
        boundary=None,
    )
    magnitude = np.abs(spectrum).astype(np.float64) + 1e-10
    power = np.square(magnitude) + 1e-12
    frame_flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    spectral_flatness = float(np.mean(frame_flatness))

    harmonic = median_filter(magnitude, size=(1, 31), mode="nearest")
    percussive = median_filter(magnitude, size=(31, 1), mode="nearest")
    harmonic_power = np.square(harmonic)
    percussive_power = np.square(percussive)
    harmonic_mask = harmonic_power / (harmonic_power + percussive_power + 1e-12)
    separated_harmonic = np.square(magnitude * harmonic_mask).sum()
    separated_percussive = np.square(magnitude * (1.0 - harmonic_mask)).sum()
    harmonicity_db = float(
        10.0 * np.log10((separated_harmonic + 1e-10) / (separated_percussive + 1e-10))
    )
    return spectral_flatness, harmonicity_db


def validate_generated_clip(path: Path, expected_duration: float) -> tuple[bool, str]:
    """Reject truncated, silent, or malformed clips before checkpointing them."""
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            return False, "invalid audio stream metadata"
        if abs(duration - expected_duration) > max(2.0, expected_duration * 0.04):
            return False, f"duration mismatch: {duration:.2f}s vs {expected_duration:.2f}s"
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        peak = float(np.max(np.abs(audio)))
        if not np.isfinite(rms) or not np.isfinite(peak):
            return False, "audio contains non-finite samples"
        if rms < 0.001 or peak < 0.01:
            return False, f"audio is effectively silent (rms={rms:.6f}, peak={peak:.6f})"
        flatness, harmonicity_db = _music_quality_metrics(audio, info.samplerate)
        if (
            flatness > MAX_NOISE_SPECTRAL_FLATNESS
            and harmonicity_db < MIN_NOISE_HARMONICITY_DB
        ):
            return False, (
                "audio is noise/percussion dominated "
                f"(spectral_flatness={flatness:.5f}, harmonicity={harmonicity_db:.2f}dB)"
            )
        return True, (
            f"duration={duration:.2f}s rms={rms:.5f} peak={peak:.5f} "
            f"spectral_flatness={flatness:.5f} harmonicity={harmonicity_db:.2f}dB"
        )
    except Exception as exc:
        return False, f"audio validation failed: {exc}"


def generate_clip(
    dit_handler: Any,
    llm_handler: Any,
    caption: str,
    output_path: Path,
    duration: float = DEFAULT_DURATION,
    inference_steps: int = DEFAULT_INFERENCE_STEPS,
    seed: int = -1,
    bpm: int | None = None,
    guidance_scale: float = 7.0,
    thinking: bool = True,
) -> bool:
    """Generate a single BGM clip and save to output_path.

    Args:
        dit_handler: Initialized ACE-Step DiT handler (AceStepHandler).
        caption: English caption for the desired music.
        output_path: Where to save the generated MP3.
        duration: Target duration in seconds.
        inference_steps: Number of diffusion steps.
        seed: Random seed (-1 = random). Use different seeds for variation.

    Returns:
        True if generation succeeded.
    """
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = GenerationParams(
        task_type="text2music",
        caption=caption[:500],
        lyrics="[Instrumental]",
        duration=duration,
        inference_steps=inference_steps,
        instrumental=True,
        bpm=bpm,
        thinking=thinking,
        guidance_scale=guidance_scale,
        sampler_mode="heun",
        lm_temperature=0.7,
        lm_cfg_scale=2.0,
        use_cot_caption=True,
        use_cot_lyrics=False,
        enable_normalization=True,
        normalization_db=-3.0,
        fade_in_duration=1.0,
        fade_out_duration=1.5,
        seed=seed,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3",
        mp3_bitrate="256k",
        mp3_sample_rate=48000,
    )
    result = generate_music(
        dit_handler,
        llm_handler if thinking else None,
        params,
        config,
        save_dir=str(output_path.parent),
    )

    if not result.success:
        print(f"    [FAIL] 生成失败: {result.error}")
        return False

    # The SDK saves to save_dir with a generated filename; rename to our target
    saved_files = result.audios
    if saved_files:
        src = Path(saved_files[0]["path"])
        if src != output_path:
            # Handle rename across devices / directories
            if src.exists():
                temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp.mp3")
                src.replace(temporary)
                valid, detail = validate_generated_clip(temporary, duration)
                if not valid:
                    temporary.unlink(missing_ok=True)
                    print(f"    [REJECT] {detail}")
                    return False
                os.replace(temporary, output_path)
                print(f"    [QC] {detail}")
        else:
            valid, detail = validate_generated_clip(output_path, duration)
            if not valid:
                output_path.unlink(missing_ok=True)
                print(f"    [REJECT] {detail}")
                return False
            print(f"    [QC] {detail}")
        return output_path.is_file() and output_path.stat().st_size > 0

    print(f"    [WARN] 未找到输出音频文件")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7-c: generate BGM per segment via ACE-Step")
    parser.add_argument("--segments", default=_DEFAULT_SEGMENTS, help="BGM segments JSON")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory for BGM clips")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help=f"Seconds per BGM clip (default: {DEFAULT_DURATION})")
    parser.add_argument("--inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS, help=f"Diffusion steps (default: {DEFAULT_INFERENCE_STEPS})")
    parser.add_argument("--model", default=ACE_STEP_MODEL, help="ACE-Step DiT checkpoint")
    parser.add_argument("--lm-model", default=ACE_STEP_LM_MODEL, help="ACE-Step 5Hz LM checkpoint")
    parser.add_argument("--lm-backend", default="vllm", choices=("vllm", "pt"))
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload models between phases (normally unnecessary for turbo on 12 GB VRAM)",
    )
    parser.add_argument(
        "--offload-dit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload DiT before VAE decode (only useful with the optional semantic LM)",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the experimental 5Hz semantic LM (disabled for stable BGM generation)",
    )
    parser.add_argument("--clip-attempts", type=int, default=3, help="Quality retry count per clip")
    parser.add_argument(
        "--process-clip-limit",
        type=int,
        default=DEFAULT_PROCESS_CLIP_LIMIT,
        help=(
            "Checkpoint and request a clean process restart after this many newly generated clips "
            f"(default: {DEFAULT_PROCESS_CLIP_LIMIT}; 0 disables)"
        ),
    )
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890"))
    parser.add_argument("--force", action="store_true", help="Regenerate all clips even if cached")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated, skip ACE-Step")
    parser.add_argument("--clips-per-segment", type=int, default=3, help="Number of BGM clips per segment (default: 3, use >1 for variation)")
    args = parser.parse_args()

    # Resolve CLI paths before ACE-Step changes its working directory for model
    # discovery. Relative paths therefore keep normal command-line semantics.
    segments_path = Path(args.segments).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "bgm_manifest.json"
    clips_per_segment = args.clips_per_segment
    thinking = bool(args.thinking)
    if args.proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[name] = args.proxy
    start_time = time.time()

    print("=" * 60)
    print("Stage 7-c: BGM Generation via ACE-Step-1.5")
    print("=" * 60)
    print(f"SDK: {ACE_DIR}")
    print(f"Segments: {segments_path}")
    print(f"Output:   {output_dir}")
    print(f"Duration: {args.duration}s per clip, {args.inference_steps} steps")
    print(f"Clips per segment: {clips_per_segment}")
    print(f"Quality model: {args.model} + {args.lm_model if thinking else 'no LM'}")
    print(f"Memory policy: cpu_offload={args.cpu_offload}, offload_dit={args.offload_dit}")
    print(f"Process clip limit: {args.process_clip_limit or 'disabled'}")

    # ── Load segments ──
    if not segments_path.exists():
        print(f"ERROR: segments file not found: {segments_path}")
        return 1
    segments = load_segments(segments_path)
    print(f"Loaded {len(segments)} segments")

    # Add segment_index if missing (0-based or 1-based)
    for i, s in enumerate(segments):
        if "segment_index" not in s:
            s["segment_index"] = i + 1

    # ── Check existing manifest ──
    existing_manifest = None
    if manifest_path.exists() and not args.force:
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)
        if (
            existing_manifest.get("generation_version") != BGM_GENERATION_VERSION
            or existing_manifest.get("model") != args.model
            or existing_manifest.get("lm_model") != (args.lm_model if thinking else None)
            or existing_manifest.get("thinking") != thinking
            or existing_manifest.get("guidance_scale") != args.guidance_scale
            or existing_manifest.get("duration_per_segment") != args.duration
            or existing_manifest.get("inference_steps") != args.inference_steps
            or existing_manifest.get("clips_per_segment") != clips_per_segment
        ):
            existing_manifest = None

    manifest_template = build_manifest(
        segments,
        args.duration,
        args.duration * len(segments),
        0,
        clips_per_segment,
        output_dir=output_dir,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        thinking=thinking,
    )
    manifest_template["model"] = args.model
    manifest_template["lm_model"] = args.lm_model if thinking else None
    manifest = json.loads(json.dumps(manifest_template))
    for entry in manifest["segments"].values():
        entry["clips"] = []

    # Copy only fully matching completed clips into a clean current manifest.
    if existing_manifest:
        old_segments = existing_manifest.get("segments", {})
        for key, expected_entry in manifest_template["segments"].items():
            old_entry = old_segments.get(key, {})
            if (
                old_entry.get("bgm_type") != expected_entry["bgm_type"]
                or old_entry.get("prompt") != expected_entry["prompt"]
            ):
                continue
            old_clips = old_entry.get("clips", [])
            for expected_clip in expected_entry["clips"]:
                clip = next(
                    (
                        item
                        for item in old_clips
                        if item.get("clip_index") == expected_clip["clip_index"]
                        and item.get("base_seed", item.get("seed")) == expected_clip["seed"]
                    ),
                    None,
                )
                path = output_dir / expected_clip["file"]
                if clip and path.is_file() and path.stat().st_size > 0:
                    manifest["segments"][key]["clips"].append(clip)

    to_generate = []
    for s in segments:
        idx = s["segment_index"]
        bgm_type = s.get("bgm_type", "unknown")
        caption = build_segment_bgm_prompt(s)

        for clip_i in range(clips_per_segment):
            out_file = output_dir / f"{idx:03d}_{clip_i}.mp3"

            # Skip if already generated
            if not args.force:
                seg_entry = manifest.get("segments", {}).get(str(idx), {})
                clips_list = seg_entry.get("clips", []) if isinstance(seg_entry, dict) else []
                clip_meta = next((c for c in clips_list if c.get("clip_index") == clip_i), None)
                if (
                    seg_entry.get("prompt") == caption
                    and clip_meta
                    and clip_meta.get("base_seed", clip_meta.get("seed")) == build_bgm_seed(idx, clip_i)
                    and out_file.exists()
                ):
                    print(f"  [{idx:03d}:{clip_i}] [SKIP] {bgm_type:12s} -> {out_file.name} (cached)")
                    continue

            to_generate.append({
                "segment_index": idx,
                "clip_index": clip_i,
                "bgm_type": bgm_type,
                "title": s.get("title", ""),
                "caption": caption,
                "output": out_file,
                "seed": build_bgm_seed(idx, clip_i),
                "bpm": s.get("bgm_tempo_bpm"),
            })

    total_clips = len(to_generate)
    print(f"\n需生成: {total_clips} / {len(segments) * clips_per_segment} 个片段")
    print(f"跳过:   {len(segments) * clips_per_segment - total_clips} 个片段 (已有缓存)\n")

    if args.dry_run:
        for g in to_generate:
            print(f"  [{g['segment_index']:03d}:{g['clip_index']}] {g['bgm_type']:12s} → {g['output'].name} (seed={g['seed']})")
        print("\nDry run complete.")
        return 0

    if not to_generate:
        print("所有片段已生成，无需处理。")
        manifest["generation_seconds"] = round(time.time() - start_time, 1)
        save_manifest(manifest, manifest_path)
        print(f"清单已更新: {manifest_path}")
        return 0

    # ── Initialize ACE-Step DiT model ──
    print("\n[1/2] 加载 ACE-Step DiT 模型...")
    t0 = time.time()
    os.chdir(str(ACE_DIR))
    os.environ["ACESTEP_CONFIG_PATH"] = args.model
    os.environ["ACESTEP_LM_MODEL_PATH"] = args.lm_model
    os.environ["ACESTEP_LM_BACKEND"] = args.lm_backend
    os.environ["ACESTEP_INIT_LLM"] = "true" if thinking else "false"
    os.environ["ACESTEP_CPU_OFFLOAD"] = "true" if args.cpu_offload else "false"
    os.environ["ACESTEP_OFFLOAD_DIT_TO_CPU"] = "true" if args.offload_dit else "false"
    initialized = _init_ace_step()
    dit_handler, llm_handler = initialized[0], initialized[1]
    print(f"  模型加载完成 [{time.time() - t0:.1f}s]\n")

    # ── Generate BGM clips ──
    print(f"[2/2] 生成 {len(to_generate)} 个 BGM 片段...")
    success_count = 0
    for g in to_generate:
        idx = g["segment_index"]
        clip_i = g["clip_index"]
        bgm_type = g["bgm_type"]
        caption = g["caption"]
        out_path = g["output"]
        title = g["title"]
        seed = g["seed"]

        print(f"\n  [{idx:03d}:{clip_i}/{len(segments):03d}] {bgm_type:12s} | {title} (seed={seed})")
        t1 = time.time()

        ok = False
        used_seed = seed
        for attempt in range(1, max(1, args.clip_attempts) + 1):
            used_seed = seed + (attempt - 1) * 7_919
            try:
                ok = generate_clip(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    caption=caption,
                    output_path=out_path,
                    duration=args.duration,
                    inference_steps=args.inference_steps,
                    seed=used_seed,
                    bpm=g.get("bpm"),
                    guidance_scale=args.guidance_scale,
                    thinking=thinking,
                )
            except Exception as exc:
                ok = False
                print(f"    [ERROR] generation attempt failed: {exc}")
            if ok:
                break
            print(f"    [RETRY] attempt {attempt}/{max(1, args.clip_attempts)}")

        if ok:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"    [OK] {out_path.name} ({size_mb:.1f} MB) [{time.time() - t1:.1f}s]")
            success_count += 1
            expected_clips = manifest_template["segments"][str(idx)]["clips"]
            clip_metadata = next(
                item for item in expected_clips if item["clip_index"] == clip_i
            )
            clip_metadata = dict(
                clip_metadata,
                seed=used_seed,
                quality_validated=True,
            )
            if used_seed != seed:
                clip_metadata["base_seed"] = seed
            completed_clips = manifest["segments"][str(idx)]["clips"]
            completed_clips[:] = [
                item for item in completed_clips if item["clip_index"] != clip_i
            ]
            completed_clips.append(clip_metadata)
            completed_clips.sort(key=lambda item: item["clip_index"])
            manifest["generation_seconds"] = round(time.time() - start_time, 1)
            save_manifest(manifest, manifest_path)
            if (
                args.process_clip_limit > 0
                and success_count >= args.process_clip_limit
                and success_count < len(to_generate)
            ):
                remaining = len(to_generate) - success_count
                print(
                    f"\n[PROCESS RESTART] checkpointed {success_count} new clips; "
                    f"{remaining} remain. Requesting a clean ACE-Step restart.",
                    flush=True,
                )
                return BGM_PROCESS_RESTART_EXIT_CODE
        else:
            print(f"    [FAIL] [{time.time() - t1:.1f}s]")

    # ── Build and save manifest ──
    total_duration = args.duration * len(segments)
    elapsed = time.time() - start_time
    manifest["total_bgm_duration"] = total_duration
    manifest["generation_seconds"] = round(elapsed, 1)
    save_manifest(manifest, manifest_path)

    print(f"\n{'=' * 60}")
    print(f"完成: {success_count}/{len(to_generate)} 个片段生成成功")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"清单: {manifest_path}")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 60}")

    return 0 if success_count == len(to_generate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
