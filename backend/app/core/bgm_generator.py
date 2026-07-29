"""
Stage 7-c: BGM generation per segment via ACE-Step SDK.

Provides the prompt mapping from BGM type → ACE-Step caption, and a CLI
entry point that is designed to be invoked with the ACE-Step venv's Python.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BGM_SEGMENTS_PATH = Path("backend/data/bgm_segments.json")
BGM_OUTPUT_DIR = Path("output/bgm")
BGM_MANIFEST_PATH = BGM_OUTPUT_DIR / "bgm_manifest.json"
BGM_GENERATION_VERSION = 4
ACE_STEP_MODEL = "acestep-v15-turbo"
ACE_STEP_LM_MODEL = "acestep-5Hz-lm-1.7B"

# ── ACE-Step caption prompts per BGM type ──────────────────────────────
# Each caption is an English prompt optimized for ACE-Step's text2music mode.
# Duration, BPM, and key are set dynamically; these captions describe the
# genre, instrumentation, and atmosphere.

BGM_PROMPTS: Dict[str, str] = {
    "suspense": (
        "tense suspenseful dark ambient music, mysterious thriller soundtrack, "
        "low strings, eerie synth pads, building tension, cinematic"
    ),
    "daily": (
        "light cheerful acoustic background music, warm and gentle, "
        "soft guitar and piano, everyday life mood, heartwarming"
    ),
    "battle": (
        "intense powerful orchestral battle music, driving percussion, "
        "brass stabs, epic war drums, high energy action scene"
    ),
    "sad": (
        "melancholic emotional piano music, sorrowful strings, "
        "slow tempo, gentle and touching, reflective atmosphere"
    ),
    "romantic": (
        "romantic warm music, gentle piano and strings, "
        "tender atmosphere, soft and loving, intimate"
    ),
    "epic": (
        "epic orchestral music, grand sweeping cinematics, "
        "powerful brass and strings, majestic, heroic adventure"
    ),
    "comedy": (
        "playful quirky music, lighthearted comedic melody, "
        "whimsical instruments, cheerful and bouncy, cartoon-like"
    ),
    "horror": (
        "dark disturbing horror music, dissonant strings, "
        "creepy atmosphere, low drones, frightening, unsettling"
    ),
}

# Fallback for unknown types
BGM_PROMPTS_DEFAULT = (
    "ambient background music, cinematic atmosphere, "
    "neutral and unobtrusive, soft pads"
)


def get_bgm_prompt(bgm_type: str) -> str:
    """Return the ACE-Step caption for a BGM type, or the default fallback."""
    return BGM_PROMPTS.get(bgm_type, BGM_PROMPTS_DEFAULT)


def build_segment_bgm_prompt(segment: Dict[str, Any]) -> str:
    """Build a scene-specific instrumental caption from reviewed evidence."""
    base = get_bgm_prompt(str(segment.get("bgm_type", "unknown")))
    reviewed = re.sub(r"\s+", " ", str(segment.get("bgm_music_prompt", ""))).strip()
    if reviewed:
        details = []
        if segment.get("bgm_tempo_bpm"):
            details.append(f"approximately {int(segment['bgm_tempo_bpm'])} BPM")
        if segment.get("bgm_key_mode"):
            details.append(str(segment["bgm_key_mode"]))
        if segment.get("bgm_narrative_arc"):
            details.append(f"{segment['bgm_narrative_arc']} dramatic arc")
        detail_text = ", ".join(details)
        tail = (
            f". {detail_text}." if detail_text else "."
        ) + (
            " Instrumental score with an evolving melody and clear harmony; narration-friendly "
            "arrangement, gentle dynamics, and controlled low end. No vocals, speech, sound "
            "effects, repetitive thumps, footstep-like pulses, noise beds, static drones, or "
            "abrupt ending."
        )
        # ACE-Step consumes at most 500 caption characters. Preserve the
        # musical/noise exclusions at the tail instead of silently truncating
        # them after an overlong reviewed scene prompt.
        reviewed_limit = max(80, 500 - len(tail))
        prompt = reviewed[:reviewed_limit]
        if len(reviewed) > reviewed_limit:
            sentence_end = max(prompt.rfind(mark) for mark in ".;!?")
            if sentence_end >= 80:
                prompt = prompt[: sentence_end + 1]
            elif " " in prompt:
                prompt = prompt.rsplit(" ", 1)[0]
        prompt = prompt.rstrip(" ,.;:")
        return prompt + tail
    evidence = str(segment.get("bgm_evidence", "")).split(" | Review:", 1)[0]
    evidence = re.sub(r"^Primary:\s*", "", evidence, flags=re.IGNORECASE)
    evidence = re.sub(r"\s+", " ", evidence).strip()
    if len(evidence) > 220:
        evidence = evidence[:220].rsplit(" ", 1)[0]
    evidence = evidence.rstrip(" ,.;:")
    if not evidence:
        evidence = re.sub(r"\s+", " ", str(segment.get("description", ""))).strip()
    if not evidence:
        evidence = "The scene maintains the requested mood without a major emotional shift"

    direction = (
        f"{base}. Scene-specific emotional direction: {evidence}. "
        "Instrumental underscore with a coherent melody and harmonic movement, no vocals, "
        "restrained dynamics, and no repetitive thumps, noise beds, or sound effects."
    )
    return direction


def build_bgm_seed(segment_index: int, clip_index: int) -> int:
    """Return a stable seed unique to a segment and its variation."""
    return int(segment_index) * 10_007 + int(clip_index) * 1_009 + 17


def load_segments(path: Path = BGM_SEGMENTS_PATH) -> List[Dict[str, Any]]:
    """Load bgm_segments.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_segments_to_generate(
    segments: List[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Return segments that still need BGM generation.

    Args:
        segments: All BGM segments.
        manifest: Existing manifest (maps segment_index → output file).
        force: If True, regenerate all regardless of manifest.

    Returns:
        Segments that need generation (those without a file in manifest).
    """
    if force or manifest is None:
        return list(segments)

    existing = set(manifest.get("segments", {}).keys())
    return [s for s in segments if str(s.get("segment_index", 0)) not in existing]


def build_manifest(
    segments: List[Dict[str, Any]],
    duration_per_segment: float,
    total_duration: float,
    elapsed: float,
    clips_per_segment: int = 1,
    output_dir: Path = BGM_OUTPUT_DIR,
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    thinking: bool = False,
) -> Dict[str, Any]:
    """Build the BGM manifest dict."""
    segment_map: Dict[str, Dict[str, Any]] = {}
    for s in segments:
        idx = s.get("segment_index", 0)
        bgm_type = s.get("bgm_type", "unknown")
        prompt = build_segment_bgm_prompt(s)

        clips_list = [
            {
                "clip_index": ci,
                "file": f"{idx:03d}_{ci}.mp3",
                "path": str(output_dir / f"{idx:03d}_{ci}.mp3"),
                "seed": build_bgm_seed(idx, ci),
            }
            for ci in range(clips_per_segment)
        ]

        segment_map[str(idx)] = {
            "bgm_type": bgm_type,
            "prompt": prompt,
            "duration": duration_per_segment,
            "clips": clips_list,
        }
    return {
        "generation_version": BGM_GENERATION_VERSION,
        "model": ACE_STEP_MODEL,
        "lm_model": ACE_STEP_LM_MODEL if thinking else None,
        "thinking": thinking,
        "guidance_scale": guidance_scale,
        "segments": segment_map,
        "duration_per_segment": duration_per_segment,
        "inference_steps": inference_steps,
        "total_segments": len(segments),
        "clips_per_segment": clips_per_segment,
        "total_bgm_duration": total_duration,
        "generation_seconds": round(elapsed, 1),
    }


def save_manifest(manifest: Dict[str, Any], path: Path = BGM_MANIFEST_PATH) -> Path:
    """Save BGM manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))
    return path
