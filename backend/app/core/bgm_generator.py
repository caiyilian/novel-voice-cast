"""
Stage 7-c: BGM generation per segment via ACE-Step SDK.

Provides the prompt mapping from BGM type → ACE-Step caption, and a CLI
entry point that is designed to be invoked with the ACE-Step venv's Python.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BGM_SEGMENTS_PATH = Path("backend/data/bgm_segments.json")
BGM_OUTPUT_DIR = Path("output/bgm")
BGM_MANIFEST_PATH = BGM_OUTPUT_DIR / "bgm_manifest.json"

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
) -> Dict[str, Any]:
    """Build the BGM manifest dict."""
    segment_map: Dict[str, Dict[str, Any]] = {}
    for s in segments:
        idx = s.get("segment_index", 0)
        bgm_type = s.get("bgm_type", "unknown")

        clips_list = [
            {
                "clip_index": ci,
                "file": f"{idx:03d}_{ci}.mp3",
                "path": str(BGM_OUTPUT_DIR / f"{idx:03d}_{ci}.mp3"),
            }
            for ci in range(clips_per_segment)
        ]

        segment_map[str(idx)] = {
            "bgm_type": bgm_type,
            "duration": duration_per_segment,
            "clips": clips_list,
        }
    return {
        "segments": segment_map,
        "duration_per_segment": duration_per_segment,
        "total_segments": len(segments),
        "clips_per_segment": clips_per_segment,
        "total_bgm_duration": total_duration,
        "generation_seconds": round(elapsed, 1),
    }


def save_manifest(manifest: Dict[str, Any], path: Path = BGM_MANIFEST_PATH) -> Path:
    """Save BGM manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
