"""
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
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup: add ACE-Step project to sys.path ──────────────────────
ACE_DIR = Path(__file__).resolve().parent / "ACE-Step-1.5"
PROJECT_ROOT = ACE_DIR.parent  # One level up: the novel-voice-cast root
os.chdir(str(ACE_DIR))
sys.path.insert(0, str(ACE_DIR))

# ACE-Step SDK imports are lazy (inside _init_ace_step) so that dry-run
# mode does not trigger CUDA initialisation.
os.environ.setdefault("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
os.environ.setdefault("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
os.environ.setdefault("ACESTEP_DEVICE", "auto")
os.environ.setdefault("ACESTEP_INIT_LLM", "false")  # DiT only (no LM), saves VRAM


def _init_ace_step() -> tuple:
    """Lazy import & init of ACE-Step SDK.  Returns (AceStepHandler, GenerationParams, GenerationConfig, generate_music)."""
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(ACE_DIR),
        config_path="acestep-v15-turbo",
        device="cuda",
    )
    return dit_handler, GenerationParams, GenerationConfig, generate_music

# ── Project paths (relative to project root, one level up from ACE_DIR) ──
PROJECT_ROOT = ACE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.bgm_generator import (
    build_manifest,
    get_bgm_prompt,
    load_segments,
    save_manifest,
)

# Resolve defaults relative to project root (os.chdir has moved us to ACE_DIR)
_DEFAULT_SEGMENTS = str(PROJECT_ROOT / "backend/data/bgm_segments.json")
_DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output/bgm")
_MANIFEST_PATH = PROJECT_ROOT / "output/bgm/bgm_manifest.json"

# ── Defaults ──
DEFAULT_DURATION = 30       # seconds per BGM clip
DEFAULT_INFERENCE_STEPS = 8  # turbo model steps


def generate_clip(
    dit_handler: Any,
    caption: str,
    output_path: Path,
    duration: float = DEFAULT_DURATION,
    inference_steps: int = DEFAULT_INFERENCE_STEPS,
    seed: int = -1,
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
        caption=caption,
        duration=duration,
        inference_steps=inference_steps,
        instrumental=True,
        seed=seed,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3",
        mp3_bitrate="192k",
        mp3_sample_rate=44100,
    )
    result = generate_music(dit_handler, None, params, config, save_dir=str(output_path.parent))

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
                src.replace(output_path)
        return True

    print(f"    [WARN] 未找到输出音频文件")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7-c: generate BGM per segment via ACE-Step")
    parser.add_argument("--segments", default=_DEFAULT_SEGMENTS, help="BGM segments JSON")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR, help="Output directory for BGM clips")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help=f"Seconds per BGM clip (default: {DEFAULT_DURATION})")
    parser.add_argument("--inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS, help=f"Diffusion steps (default: {DEFAULT_INFERENCE_STEPS})")
    parser.add_argument("--force", action="store_true", help="Regenerate all clips even if cached")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated, skip ACE-Step")
    parser.add_argument("--clips-per-segment", type=int, default=3, help="Number of BGM clips per segment (default: 3, use >1 for variation)")
    args = parser.parse_args()

    segments_path = Path(args.segments)
    output_dir = Path(args.output_dir)
    clips_per_segment = args.clips_per_segment
    start_time = time.time()

    print("=" * 60)
    print("Stage 7-c: BGM Generation via ACE-Step-1.5")
    print("=" * 60)
    print(f"SDK: {ACE_DIR}")
    print(f"Segments: {segments_path}")
    print(f"Output:   {output_dir}")
    print(f"Duration: {args.duration}s per clip, {args.inference_steps} steps")
    print(f"Clips per segment: {clips_per_segment}")

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
    manifest = None
    if _MANIFEST_PATH.exists() and not args.force:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    to_generate = []
    for s in segments:
        idx = s["segment_index"]
        bgm_type = s.get("bgm_type", "unknown")
        caption = get_bgm_prompt(bgm_type)

        for clip_i in range(clips_per_segment):
            out_file = output_dir / f"{idx:03d}_{clip_i}.mp3"

            # Skip if already generated
            if manifest and not args.force:
                seg_entry = manifest.get("segments", {}).get(str(idx), {})
                clips_list = seg_entry.get("clips", []) if isinstance(seg_entry, dict) else []
                clip_meta = next((c for c in clips_list if c.get("clip_index") == clip_i), None)
                if clip_meta and out_file.exists():
                    print(f"  [{idx:03d}:{clip_i}] [SKIP] {bgm_type:12s} -> {out_file.name} (cached)")
                    continue

            to_generate.append({
                "segment_index": idx,
                "clip_index": clip_i,
                "bgm_type": bgm_type,
                "title": s.get("title", ""),
                "caption": caption,
                "output": out_file,
                "seed": clip_i * 100,  # different seed per clip
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
        # Still build/update manifest
        manifest = build_manifest(segments, args.duration, args.duration * len(segments), 0, clips_per_segment)
        save_manifest(manifest)
        print(f"清单已更新: {_MANIFEST_PATH}")
        return 0

    # ── Initialize ACE-Step DiT model ──
    print("\n[1/2] 加载 ACE-Step DiT 模型...")
    t0 = time.time()
    dit_handler, GenerationParams_cls, GenerationConfig_cls, _ = _init_ace_step()
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

        ok = generate_clip(
            dit_handler=dit_handler,
            caption=caption,
            output_path=out_path,
            duration=args.duration,
            inference_steps=args.inference_steps,
            seed=seed,
        )

        if ok:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"    [OK] {out_path.name} ({size_mb:.1f} MB) [{time.time() - t1:.1f}s]")
            success_count += 1
        else:
            print(f"    [FAIL] [{time.time() - t1:.1f}s]")

    # ── Build and save manifest ──
    total_duration = args.duration * len(segments)
    elapsed = time.time() - start_time
    manifest = build_manifest(segments, args.duration, total_duration, elapsed, clips_per_segment)
    save_manifest(manifest)

    print(f"\n{'=' * 60}")
    print(f"完成: {success_count}/{len(to_generate)} 个片段生成成功")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"清单: {_MANIFEST_PATH}")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 60}")

    return 0 if success_count == len(to_generate) else 1


if __name__ == "__main__":
    raise SystemExit(main())