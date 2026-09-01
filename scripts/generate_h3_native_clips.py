"""Generate an H3-native illustrated audiobook without anchoring every old image.

The first beat of each coarse story/BGM scene uses text-to-video.  Subsequent
beats in the same scene use the prior H3 clip's continuation frame as their
first-frame reference.  This preserves local visual continuity while allowing
H3 to establish higher-quality compositions instead of inheriting all legacy
illustrations.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from PIL import Image, ImageChops, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.generate_h3_clips import (  # noqa: E402
    CHECKPOINT_VERSION,
    H3Client,
    H3GenerationError,
    H3JobResumeRequired,
    atomic_write_json,
    build_timeline,
    download_completed_job,
    h3_frame_count,
    load_audited_plan,
    load_plan,
    probe_video,
    safe_title,
    sha256_file,
    sha256_text,
    utc_now,
    validate_downloaded_clip,
    wait_for_job,
)
from app.core.illustration_planner import (  # noqa: E402
    _format_character_cards,
    _format_visual_memory_for_chunk,
    _load_character_cards,
    _load_visual_memory,
)
from app.core.subtitles import (  # noqa: E402
    build_segment_timestamps,
    discover_segment_files,
    load_subtitle_entries,
)


LOGGER = logging.getLogger("h3_native")
CONTINUOUS_CHECKPOINT_VERSION = 3
SHOT_PLAN_VERSION = 1
FREEZE_DURATION_RE = re.compile(r"freeze_duration:\s*([0-9.]+)")
BLACK_DURATION_RE = re.compile(r"black_duration:([0-9.]+)")


class H3MotionQualityError(H3GenerationError):
    def __init__(self, message: str, metrics: Mapping[str, Any]):
        super().__init__(message)
        self.metrics = dict(metrics)


def load_scene_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("segments")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Scene segmentation is missing or empty: {path}")
    segments = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Scene segment {index + 1} is not an object")
        start = int(item.get("start_line", 0))
        end = int(item.get("end_line", 0))
        if start <= 0 or end < start:
            raise ValueError(f"Scene segment {index + 1} has an invalid line range")
        segments.append(dict(item))
    return segments


def scene_for_line(line: int, segments: Sequence[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    for index, segment in enumerate(segments):
        if int(segment["start_line"]) <= line <= int(segment["end_line"]):
            return index, segment
    following = [
        (index, segment)
        for index, segment in enumerate(segments)
        if int(segment["start_line"]) > line
    ]
    if following:
        return following[0]
    return len(segments) - 1, segments[-1]


def requested_duration_for_interval(
    interval_seconds: float,
    *,
    minimum_duration: int,
    maximum_duration: int,
) -> tuple[int, float, float]:
    """Prefer the longest clip that fits; short beats use a trimmed minimum clip."""

    for requested in range(maximum_duration, minimum_duration - 1, -1):
        actual = h3_frame_count(requested) / 24
        if actual <= interval_seconds + 1e-6:
            return requested, actual, actual
    actual = h3_frame_count(minimum_duration) / 24
    return minimum_duration, actual, max(1 / 24, interval_seconds)


def continuous_duration_parts(
    interval_seconds: float,
    *,
    minimum_duration: int,
    maximum_duration: int,
) -> list[tuple[int, float, float, float, float]]:
    """Split one narration interval into balanced, fully covered H3 clips.

    Each tuple is ``(requested, expected, usable, start, end)``.  H3's
    17k+5 frame quantisation means an integer request is not exactly that many
    seconds, so the smallest request whose real output covers each balanced
    part is selected.  A very short interval still generates one minimum clip
    and trims it; no part ever asks the renderer to extend a final frame.
    """

    interval = max(1 / 1000, float(interval_seconds))
    maximum_actual = h3_frame_count(maximum_duration) / 24
    part_count = max(1, math.ceil((interval - 1e-9) / maximum_actual))
    parts: list[tuple[int, float, float, float, float]] = []
    for part_index in range(part_count):
        start = interval * part_index / part_count
        end = interval * (part_index + 1) / part_count
        usable = end - start
        selected: tuple[int, float] | None = None
        for requested in range(minimum_duration, maximum_duration + 1):
            expected = h3_frame_count(requested) / 24
            if expected + 1e-9 >= usable:
                selected = requested, expected
                break
        if selected is None:  # Protected by part_count, retained as a hard invariant.
            raise ValueError(
                f"Could not cover {usable:.6f}s with an H3 clip up to "
                f"{maximum_duration}s"
            )
        parts.append((selected[0], selected[1], usable, start, end))
    if abs(parts[-1][4] - interval) > 1e-6:
        raise AssertionError("Continuous H3 duration split lost source coverage")
    return parts


def shot_phase(part_index: int, part_count: int) -> str:
    if part_count <= 1:
        return "complete"
    if part_index == 0:
        return "establish"
    if part_index == part_count - 1:
        return "settle"
    if part_index * 2 < part_count:
        return "develop"
    return "react"


def camera_direction(part_index: int, part_count: int, phase: str) -> str:
    if part_count <= 1:
        return "Use one stable medium or wide composition and a single restrained camera move."
    if phase == "establish":
        return "Begin with a readable wide or medium establishing composition and reveal spatial geography."
    if phase == "settle":
        return "Ease into a calm medium composition that gives the following editorial cut a clean endpoint."
    variations = (
        "Use a gentle lateral track that reveals only already-described story elements.",
        "Move to a closer view of the literal focal subject for visible gesture and expression.",
        "Use a motivated reverse or reaction angle while keeping screen direction consistent.",
        "Use a restrained foreground-detail or object insert only when that detail exists in the beat.",
        "Return to a medium relational composition with stable character and object geography.",
    )
    return variations[(part_index - 1) % len(variations)]


def build_native_prompt(
    item: Mapping[str, Any],
    scene: Mapping[str, Any],
    *,
    duration_seconds: int,
    uses_first_frame: bool,
    uses_last_frame: bool = False,
    part_index: int = 0,
    part_count: int = 1,
    phase: str | None = None,
    camera: str = "",
    composition_direction: str = "",
    identity_context: str = "",
    audio_context: str = "",
) -> str:
    visual = str(item.get("prompt", "")).strip()
    description = str(item.get("description", "")).strip()
    title = str(item.get("title", "story beat")).strip()
    scene_title = str(scene.get("title", "story scene")).strip()
    scene_description = str(scene.get("description", "")).strip()
    if uses_first_frame and uses_last_frame:
        reference = (
            "How the reference pictures align with the target video — Picture 1 "
            "aligns with 0.00 seconds; Picture 2 aligns with "
            f"{duration_seconds:.2f} seconds.\n\n"
        )
    elif uses_first_frame:
        reference = "For the target video, at 0.00 seconds, Picture 1 is fully referenced.\n\n"
    elif uses_last_frame:
        reference = (
            f"For the target video, at {duration_seconds:.2f} seconds, Picture 1 is "
            "fully referenced as the exact ending composition.\n\n"
        )
    else:
        reference = ""
    opening = (
        "Continue naturally from Picture 1, preserving the exact same character faces, "
        "hair, clothing, age, body proportions, objects, palette, linework, lighting, "
        "camera geography, and environment. "
        if uses_first_frame
        else "Establish a polished original composition for this story scene. "
    )
    phase_name = phase or shot_phase(part_index, part_count)
    phase_direction = {
        "establish": "Establish geography and begin the beat's first visible action.",
        "develop": "Advance the same visible action with a motivated change in staging or camera distance.",
        "react": "Show the literal reaction or consequence while preserving spatial continuity.",
        "settle": "Complete the visible action and settle naturally for the next editorial cut.",
        "complete": "Show one complete, restrained action with a clear beginning and ending.",
    }.get(phase_name, "Advance the visible story action without inventing new plot events.")
    camera_instruction = str(camera).strip() or camera_direction(
        part_index, part_count, phase_name
    )
    ending = (
        "Finish exactly on the supplied ending reference without morphing identities. "
        if uses_last_frame
        else ""
    )
    composition = str(composition_direction).strip()
    if composition:
        composition = f"Aspect-ratio composition: {composition} "
    identity = str(identity_context).strip()
    if identity:
        identity = (
            "Continuity bible (binding; do not contradict these facts):\n"
            + identity
            + "\n"
        )
    audio_cues = str(audio_context).strip()
    if audio_cues:
        audio_cues = (
            "Audio-locked timing cues for this exact micro-shot (use only concrete visible "
            "actions or reactions consistent with the audited visual target; never draw or "
            "speak these words, and never literalize metaphors):\n"
            + audio_cues
            + "\n"
        )
    return (
        reference
        + "integrated_multimodal_description: [Shot 1] High-end 2D anime feature-film "
        f"look, coherent cinematic direction. {opening}"
        f"This is micro-shot {part_index + 1} of {part_count} for the current beat. "
        f"{phase_direction} {camera_instruction} {ending}{composition}"
        "Depict only literal visible story elements. Use subtle breathing, natural blinking, "
        "cloth and hair motion, physically plausible object movement, restrained acting, "
        "stable anatomy, and one motivated camera move. Do not invent extra people, animals, "
        "props, costumes, or supernatural effects. Avoid face drift, body morphing, sudden "
        "wardrobe changes, or unmotivated cuts. "
        f"Scene ({scene_title}): {scene_description} "
        f"Current beat ({title}): {description} Visual target: {visual} "
        f"{identity}{audio_cues}"
        f"Let the action settle into a clean, holdable composition before {duration_seconds:.2f} seconds.\n\n"
        "overall_soundscape: Natural ambience only. No spoken dialogue and no voiceover; "
        "the final VoxCPM dialogue and production mix are supplied separately.\n\n"
        "non_diegetic_music: N/A\n\n"
        "No captions, subtitles, written text, logos, signatures, or watermarks. Keep eyes, "
        "hands, teeth, faces, and identity stable."
    )


def extract_continuation_frame(
    clip_path: Path,
    output_path: Path,
    *,
    timestamp_seconds: float,
    media_duration_seconds: float,
    ffmpeg: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.extracting{output_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    # Container duration points just after the final frame.  Seeking exactly to
    # that value returns an empty image, so stay at least two H3 frames inside
    # the stream and never select a frame after the narration beat boundary.
    target = min(
        max(0.0, float(timestamp_seconds) - 1 / 24),
        max(0.0, float(media_duration_seconds) - 2 / 24),
    )
    errors = []
    try:
        for fallback in (0.0, 0.25, 0.5):
            temporary.unlink(missing_ok=True)
            seek = max(0.0, target - fallback)
            command = [
                ffmpeg,
                "-y",
                "-nostdin",
                "-ss",
                f"{seek:.6f}",
                "-i",
                str(clip_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                str(temporary),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
            )
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                os.replace(temporary, output_path)
                return
            errors.append(result.stderr[-500:])
        raise H3GenerationError(
            "Could not extract H3 continuation frame after safe seek fallbacks: "
            + errors[-1]
        )
    finally:
        temporary.unlink(missing_ok=True)


def inspect_motion_quality(
    clip_path: Path,
    *,
    usable_duration: float,
    ffmpeg: str,
    max_freeze_ratio: float,
    max_black_ratio: float,
) -> dict[str, Any]:
    """Decode the used range and reject long exact freezes or black output."""

    duration = max(1 / 24, float(usable_duration))
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(clip_path),
        "-vf",
        "freezedetect=n=-60dB:d=1.0,blackdetect=d=0.5:pix_th=0.02",
        "-an",
        "-f",
        "null",
        os.devnull,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=max(120, int(duration * 20)),
    )
    if result.returncode != 0:
        raise H3GenerationError(
            f"FFmpeg could not decode H3 clip for quality inspection: {result.stderr[-500:]}"
        )
    freeze_durations = [float(value) for value in FREEZE_DURATION_RE.findall(result.stderr)]
    black_durations = [float(value) for value in BLACK_DURATION_RE.findall(result.stderr)]
    longest_freeze = max(freeze_durations, default=0.0)
    black_seconds = min(duration, sum(black_durations))
    freeze_ratio = longest_freeze / duration
    black_ratio = black_seconds / duration
    passed = freeze_ratio <= max_freeze_ratio and black_ratio <= max_black_ratio
    metrics = {
        "status": "passed" if passed else "failed",
        "analyzed_seconds": round(duration, 6),
        "longest_freeze_seconds": round(longest_freeze, 6),
        "freeze_ratio": round(freeze_ratio, 6),
        "black_seconds": round(black_seconds, 6),
        "black_ratio": round(black_ratio, 6),
        "max_freeze_ratio": max_freeze_ratio,
        "max_black_ratio": max_black_ratio,
    }
    if not passed:
        raise H3MotionQualityError(
            "H3 motion quality gate failed: "
            f"freeze_ratio={freeze_ratio:.3f} (max {max_freeze_ratio:.3f}), "
            f"black_ratio={black_ratio:.3f} (max {max_black_ratio:.3f})",
            metrics,
        )
    return metrics


def image_similarity(first: Path, second: Path) -> float:
    with Image.open(first) as first_image, Image.open(second) as second_image:
        left = first_image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        right = second_image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        difference = ImageChops.difference(left, right)
        channel_means = ImageStat.Stat(difference).mean
    return max(0.0, min(1.0, 1.0 - sum(channel_means) / (len(channel_means) * 255)))


def inspect_anchor_quality(
    clip_path: Path,
    *,
    first_frame: Path | None,
    last_frame: Path | None,
    media_duration_seconds: float,
    ffmpeg: str,
    min_anchor_similarity: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "min_anchor_similarity": min_anchor_similarity,
        "first_frame_similarity": None,
        "last_frame_similarity": None,
    }
    checks = (
        ("first_frame_similarity", first_frame, 0.0),
        ("last_frame_similarity", last_frame, media_duration_seconds),
    )
    for key, reference, timestamp in checks:
        if reference is None:
            continue
        extracted = clip_path.with_name(
            f".{clip_path.stem}.{os.getpid()}.{key}.png"
        )
        try:
            extract_continuation_frame(
                clip_path,
                extracted,
                timestamp_seconds=timestamp,
                media_duration_seconds=media_duration_seconds,
                ffmpeg=ffmpeg,
            )
            similarity = image_similarity(reference, extracted)
            metrics[key] = round(similarity, 6)
            if similarity < min_anchor_similarity:
                failed = {**metrics, "status": "failed", "failed_anchor": key}
                raise H3MotionQualityError(
                    f"H3 anchor quality gate failed: {key}={similarity:.3f} "
                    f"(min {min_anchor_similarity:.3f})",
                    failed,
                )
        finally:
            extracted.unlink(missing_ok=True)
    metrics["status"] = "passed"
    return metrics


def build_records(
    plan: Sequence[dict[str, Any]],
    timeline: Sequence[dict[str, Any]],
    scenes: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    frames_dir: Path,
    minimum_duration: int,
    maximum_duration: int,
    limit: int | None,
    mode: str = "native-chain",
    max_chain_length: int = 3,
) -> list[dict[str, Any]]:
    if mode not in {"native-chain", "continuous-chain"}:
        raise ValueError(f"Unsupported native H3 mode: {mode}")
    if max_chain_length < 1:
        raise ValueError("max_chain_length must be positive")
    count = len(plan) if limit is None else min(len(plan), max(0, limit))
    records: list[dict[str, Any]] = []
    chain_position = 0
    previous_scene_index: int | None = None
    for beat_index in range(count):
        item = plan[beat_index]
        interval = int(timeline[beat_index]["duration_ms"]) / 1000
        scene_index, scene = scene_for_line(int(item.get("start_line", 1)), scenes)
        title = safe_title(item.get("title"), f"clip_{beat_index + 1:04d}")
        if mode == "continuous-chain":
            parts = continuous_duration_parts(
                interval,
                minimum_duration=minimum_duration,
                maximum_duration=maximum_duration,
            )
        else:
            requested, expected, usable = requested_duration_for_interval(
                interval,
                minimum_duration=minimum_duration,
                maximum_duration=maximum_duration,
            )
            parts = [(requested, expected, usable, 0.0, usable)]
        for part_index, (requested, expected, usable, coverage_start, coverage_end) in enumerate(parts):
            scene_changed = scene_index != previous_scene_index
            if scene_changed:
                chain_position = 0
            if mode == "continuous-chain":
                continuity_reset = chain_position == 0
                current_chain_position = chain_position
                chain_position = (chain_position + 1) % max_chain_length
            else:
                continuity_reset = scene_changed
                current_chain_position = chain_position
                chain_position += 1
            previous_scene_index = scene_index
            global_index = len(records)
            phase = shot_phase(part_index, len(parts))
            camera = camera_direction(part_index, len(parts), phase)
            static_identity = sha256_text(
                {
                    "mode": mode,
                    "beat_index": beat_index,
                    "part_index": part_index,
                    "part_count": len(parts),
                    "item": item,
                    "timeline": timeline[beat_index],
                    "scene_index": scene_index,
                    "scene": scene,
                    "requested_duration": requested,
                    "expected_duration": expected,
                    "usable_duration": usable,
                    "coverage_start_seconds": coverage_start,
                    "coverage_end_seconds": coverage_end,
                    "continuity_reset": continuity_reset,
                    "max_chain_length": max_chain_length,
                    "phase": phase,
                    "camera_direction": camera,
                }
            )
            suffix = (
                f"{beat_index + 1:04d}_p{part_index + 1:02d}_{title}"
                if mode == "continuous-chain"
                else f"{beat_index + 1:04d}_{title}"
            )
            records.append(
                {
                    "index": global_index,
                    "beat_index": beat_index,
                    "part_index": part_index,
                    "part_count": len(parts),
                    "shot_id": f"b{beat_index + 1:04d}-p{part_index + 1:02d}",
                    "shot_phase": phase,
                    "camera_direction": camera,
                    "title": title,
                    "scene_index": scene_index,
                    "scene_title": str(scene.get("title", "")),
                    "status": "pending",
                    "attempts": 0,
                    "job_id": None,
                    "submitted_at": None,
                    "completed_at": None,
                    "output_file": None,
                    "output_sha256": None,
                    "continuation_frame": None,
                    "continuation_frame_sha256": None,
                    "duration_seconds": None,
                    "requested_duration": requested,
                    "expected_duration": expected,
                    "usable_duration": usable,
                    "coverage_start_seconds": coverage_start,
                    "coverage_end_seconds": coverage_end,
                    "interval_duration": interval,
                    "continuity_reset": continuity_reset,
                    "chain_position": current_chain_position,
                    "uses_first_frame": None,
                    "uses_last_frame": False,
                    "input_frame": None,
                    "input_frame_sha256": None,
                    "target_frame": None,
                    "target_frame_sha256": None,
                    "legacy_reused": False,
                    "static_identity": static_identity,
                    "fingerprint": None,
                    "quality": None,
                    "error_summary": None,
                    "output_target": str((output_dir / f"{suffix}.mp4").resolve()),
                    "frame_target": str(
                        (frames_dir / f"{suffix}_continuation.png").resolve()
                    ),
                }
            )
    return records


def prepare_checkpoint(
    path: Path,
    records: Sequence[dict[str, Any]],
    *,
    endpoint: str,
    width: int,
    height: int,
    minimum_duration: int,
    maximum_duration: int,
    scene_source_hash: str,
    resume: bool,
    mode: str = "native-chain",
    max_chain_length: int = 3,
    max_freeze_ratio: float = 0.65,
    max_black_ratio: float = 0.20,
    min_anchor_similarity: float = 0.90,
) -> dict[str, Any]:
    source_hash = sha256_text(
        {
            "records": [record["static_identity"] for record in records],
            "scene_source_hash": scene_source_hash,
            "width": width,
            "height": height,
            "minimum_duration": minimum_duration,
            "maximum_duration": maximum_duration,
            "mode": mode,
            "max_chain_length": max_chain_length,
            "max_freeze_ratio": max_freeze_ratio,
            "max_black_ratio": max_black_ratio,
            "min_anchor_similarity": min_anchor_similarity,
        }
    )
    beat_count = 1 + max((int(record["beat_index"]) for record in records), default=-1)
    required_seconds = sum(
        max(
            float(record["coverage_end_seconds"])
            for record in records
            if int(record["beat_index"]) == beat_index
        )
        for beat_index in range(beat_count)
    )
    fresh = {
        "version": (
            CONTINUOUS_CHECKPOINT_VERSION if mode == "continuous-chain" else CHECKPOINT_VERSION
        ),
        "mode": mode,
        "endpoint": endpoint.rstrip("/"),
        "width": width,
        "height": height,
        "minimum_duration": minimum_duration,
        "maximum_duration": maximum_duration,
        "max_chain_length": max_chain_length,
        "max_freeze_ratio": max_freeze_ratio,
        "max_black_ratio": max_black_ratio,
        "min_anchor_similarity": min_anchor_similarity,
        "scene_source_hash": scene_source_hash,
        "source_hash": source_hash,
        "coverage": {
            "beat_count": beat_count,
            "clip_count": len(records),
            "required_seconds": round(required_seconds, 6),
            "planned_seconds": round(
                sum(float(record["usable_duration"]) for record in records), 6
            ),
            "completed_seconds": 0.0,
            "complete": False,
        },
        "updated_at": utc_now(),
        "clips": [dict(record) for record in records],
    }
    if not resume or not path.is_file():
        return fresh
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable H3 native checkpoint: %s", path)
        return fresh
    expected_version = (
        CONTINUOUS_CHECKPOINT_VERSION if mode == "continuous-chain" else CHECKPOINT_VERSION
    )
    base_compatible = (
        isinstance(old, dict)
        and old.get("version") == expected_version
        and old.get("mode") == mode
        and old.get("endpoint") == endpoint.rstrip("/")
        and old.get("width") == width
        and old.get("height") == height
        and old.get("minimum_duration") == minimum_duration
        and old.get("maximum_duration") == maximum_duration
        and old.get("max_chain_length", 3) == max_chain_length
        and float(old.get("max_freeze_ratio", 0.65)) == max_freeze_ratio
        and float(old.get("max_black_ratio", 0.20)) == max_black_ratio
        and float(old.get("min_anchor_similarity", -1.0)) == min_anchor_similarity
        and isinstance(old.get("clips"), list)
    )
    exact_legacy_compatible = (
        base_compatible
        and old.get("source_hash") == source_hash
        and len(old["clips"]) == len(records)
    )
    if not base_compatible or (mode != "continuous-chain" and not exact_legacy_compatible):
        LOGGER.warning("Ignoring incompatible H3 native checkpoint: %s", path)
        return fresh
    previous_by_identity = {
        str(previous.get("static_identity")): previous
        for previous in old["clips"]
        if isinstance(previous, dict) and previous.get("static_identity")
    }
    structural_keys = {
        "index",
        "beat_index",
        "part_index",
        "part_count",
        "shot_id",
        "shot_phase",
        "camera_direction",
        "title",
        "scene_index",
        "scene_title",
        "requested_duration",
        "expected_duration",
        "usable_duration",
        "coverage_start_seconds",
        "coverage_end_seconds",
        "interval_duration",
        "continuity_reset",
        "chain_position",
        "audio_start_ms",
        "audio_end_ms",
        "source_lines",
        "audio_cues",
        "static_identity",
    }
    reused = 0
    for current in fresh["clips"]:
        previous = previous_by_identity.get(str(current["static_identity"]))
        if not isinstance(previous, dict):
            continue
        structural = {key: current.get(key) for key in structural_keys}
        output_target = current["output_target"]
        frame_target = current["frame_target"]
        current.update(previous)
        current.update(structural)
        current["output_target"] = output_target
        current["frame_target"] = frame_target
        reused += 1
    if mode == "continuous-chain" and old.get("source_hash") != source_hash:
        LOGGER.info(
            "Reused %d/%d unchanged continuous H3 records after shot-plan change",
            reused,
            len(records),
        )
    update_checkpoint_summary(fresh)
    return fresh


def update_checkpoint_summary(checkpoint: dict[str, Any]) -> None:
    records = checkpoint.get("clips", [])
    if not isinstance(records, list):
        return
    coverage = checkpoint.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
        checkpoint["coverage"] = coverage
    completed = [
        record
        for record in records
        if isinstance(record, dict) and record.get("status") == "success"
    ]
    coverage["clip_count"] = len(records)
    coverage["completed_clips"] = len(completed)
    coverage["completed_seconds"] = round(
        sum(float(record.get("usable_duration") or 0) for record in completed), 6
    )
    coverage["complete"] = bool(records) and len(completed) == len(records)


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    update_checkpoint_summary(checkpoint)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(path, checkpoint)


def write_shot_plan(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    scene_source_hash: str,
) -> None:
    fields = (
        "index",
        "beat_index",
        "part_index",
        "part_count",
        "shot_id",
        "shot_phase",
        "camera_direction",
        "title",
        "scene_index",
        "scene_title",
        "requested_duration",
        "expected_duration",
        "usable_duration",
        "coverage_start_seconds",
        "coverage_end_seconds",
        "interval_duration",
        "continuity_reset",
        "chain_position",
        "audio_start_ms",
        "audio_end_ms",
        "source_lines",
        "audio_cues",
        "static_identity",
    )
    shots = [{key: record.get(key) for key in fields} for record in records]
    payload = {
        "version": SHOT_PLAN_VERSION,
        "mode": "continuous-chain",
        "scene_source_hash": scene_source_hash,
        "source_hash": sha256_text([shot["static_identity"] for shot in shots]),
        "beat_count": 1 + max((int(shot["beat_index"]) for shot in shots), default=-1),
        "clip_count": len(shots),
        "required_seconds": round(sum(float(shot["usable_duration"]) for shot in shots), 6),
        "shots": shots,
    }
    if path.is_file():
        try:
            if json.loads(path.read_text(encoding="utf-8")) == payload:
                return
        except (OSError, json.JSONDecodeError):
            pass
    atomic_write_json(path, payload)


def build_identity_contexts(
    plan: Sequence[Mapping[str, Any]],
    *,
    character_cards_path: Path | None,
    visual_memory_path: Path | None,
) -> dict[int, str]:
    cards = _load_character_cards(character_cards_path) if character_cards_path else {}
    memory = _load_visual_memory(visual_memory_path) if visual_memory_path else {}
    contexts: dict[int, str] = {}
    for beat_index, item in enumerate(plan):
        raw_names = item.get("characters", [])
        names = (
            [str(name).strip() for name in raw_names if str(name).strip()]
            if isinstance(raw_names, list)
            else []
        )
        if not names:
            continue
        start_line = max(1, int(item.get("start_line", 1)))
        end_line = max(start_line, int(item.get("end_line", start_line)))
        blocks = [
            _format_character_cards(names, cards).strip(),
            _format_visual_memory_for_chunk(
                names,
                memory,
                start_line,
                end_line,
            ).strip(),
        ]
        context = "\n".join(block for block in blocks if block)
        if context:
            contexts[beat_index] = context
    return contexts


def load_performance_controls(paths: Sequence[Path]) -> dict[int, str]:
    controls: dict[int, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring unreadable performance directions: %s", path)
            continue
        raw = payload.get("results") if isinstance(payload, dict) else payload
        values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            try:
                dialogue_index = int(item.get("dialogue_index"))
            except (TypeError, ValueError):
                continue
            control = str(item.get("performance_control", "")).strip()
            if control:
                controls[dialogue_index] = control
    return controls


def format_audio_context(cues: Sequence[Mapping[str, Any]]) -> str:
    context_lines = []
    for cue in cues:
        cue_start = float(cue.get("visible_from_seconds", 0.0))
        cue_end = float(cue.get("visible_until_seconds", cue_start))
        line = (
            f"[{cue_start:.2f}s-{cue_end:.2f}s] line {cue.get('source_line', 0)}, "
            f"{cue.get('speaker', '')}: {cue.get('text', '')}"
        )
        if cue.get("performance_control"):
            line += f" | acting: {cue['performance_control']}"
        context_lines.append(line)
    return "\n".join(context_lines)


def attach_audio_locked_cues(
    records: Sequence[dict[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
    entries: Sequence[Any],
    timestamps: Sequence[Mapping[str, Any]],
    *,
    performance_paths: Sequence[Path],
) -> None:
    """Attach the exact speech and acting cues overlapping every micro-shot."""

    if len(entries) != len(timestamps):
        raise ValueError("Subtitle entries and timestamps do not match")
    starts = [int(item["start_ms"]) for item in timestamps]
    ends = [int(item["end_ms"]) for item in timestamps]
    controls = load_performance_controls(performance_paths)
    for record in records:
        beat_index = int(record["beat_index"])
        beat_start = int(timeline[beat_index]["start_ms"])
        absolute_start = beat_start + round(float(record["coverage_start_seconds"]) * 1000)
        absolute_end = beat_start + round(float(record["coverage_end_seconds"]) * 1000)
        left = bisect_right(ends, absolute_start)
        right = bisect_left(starts, absolute_end)
        cues = []
        for dialogue_index in range(left, right):
            entry = entries[dialogue_index]
            cue_start_ms = int(timestamps[dialogue_index]["start_ms"])
            cue_end_ms = int(timestamps[dialogue_index]["end_ms"])
            text = str(getattr(entry, "text", "")).strip()
            speaker = str(getattr(entry, "speaker", "")).strip()
            source_line = int(getattr(entry, "source_line", 0) or 0)
            cue = {
                "dialogue_index": dialogue_index,
                "source_line": source_line,
                "speaker": speaker,
                "text": text,
                "audio_start_ms": cue_start_ms,
                "audio_end_ms": cue_end_ms,
                "visible_from_seconds": round(
                    max(0, cue_start_ms - absolute_start) / 1000,
                    3,
                ),
                "visible_until_seconds": round(
                    max(0, min(absolute_end, cue_end_ms) - absolute_start) / 1000,
                    3,
                ),
            }
            control = controls.get(dialogue_index)
            if control:
                cue["performance_control"] = control
            cues.append(cue)
        record["audio_start_ms"] = absolute_start
        record["audio_end_ms"] = absolute_end
        record["source_lines"] = sorted(
            {int(cue["source_line"]) for cue in cues if int(cue["source_line"]) > 0}
        )
        record["audio_cues"] = cues
        record["static_identity"] = sha256_text(
            {
                "base_static_identity": record["static_identity"],
                "audio_start_ms": absolute_start,
                "audio_end_ms": absolute_end,
                "audio_cues": cues,
            }
        )


def reset_record(record: dict[str, Any], *, keep_attempts: bool = False) -> None:
    attempts = int(record.get("attempts") or 0) if keep_attempts else 0
    record.update(
        status="pending",
        attempts=attempts,
        job_id=None,
        submitted_at=None,
        completed_at=None,
        output_file=None,
        output_sha256=None,
        continuation_frame=None,
        continuation_frame_sha256=None,
        duration_seconds=None,
        quality=None,
        legacy_reused=False,
        legacy_job_recovery=False,
        migrated_from=None,
        error_summary=None,
    )


def validate_cached_record(
    record: Mapping[str, Any],
    *,
    fingerprint: str,
    width: int,
    height: int,
    ffprobe: str,
) -> bool:
    if record.get("status") != "success":
        return False
    quality = record.get("quality")
    if not isinstance(quality, Mapping) or quality.get("status") != "passed":
        return False
    legacy_reused = bool(record.get("legacy_reused"))
    if not legacy_reused and record.get("fingerprint") != fingerprint:
        return False
    output = record.get("output_file")
    frame = record.get("continuation_frame")
    if not output or not frame:
        return False
    output_path = Path(str(output))
    frame_path = Path(str(frame))
    if not output_path.is_file() or not frame_path.is_file():
        return False
    if legacy_reused:
        metadata = probe_video(output_path, ffprobe)
        if metadata["width"] != width or metadata["height"] != height:
            return False
        if float(metadata["duration"]) + 1 / 24 < float(record["usable_duration"]):
            return False
    else:
        validate_downloaded_clip(
            output_path,
            width=width,
            height=height,
            expected_duration=float(record["expected_duration"]),
            ffprobe=ffprobe,
        )
    return True


def migrate_legacy_records(
    checkpoint: dict[str, Any],
    legacy_checkpoint_path: Path,
    *,
    checkpoint_path: Path,
    keyframes_dir: Path,
    width: int,
    height: int,
    ffmpeg: str,
    ffprobe: str,
    max_freeze_ratio: float,
    max_black_ratio: float,
) -> int:
    """Reuse compatible legacy beat clips without modifying their files.

    Only the first micro-shot of each beat is eligible.  A new continuation
    frame is extracted at the new micro-shot boundary, so subsequent clips
    continue from exactly the frame that the continuous renderer will cut on.
    """

    if not legacy_checkpoint_path.is_file():
        return 0
    try:
        legacy = json.loads(legacy_checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable legacy H3 checkpoint: %s", legacy_checkpoint_path)
        return 0
    legacy_records = legacy.get("clips") if isinstance(legacy, dict) else None
    if (
        not isinstance(legacy_records, list)
        or legacy.get("mode") != "native-chain"
        or int(legacy.get("width", 0)) != width
        or int(legacy.get("height", 0)) != height
    ):
        LOGGER.warning("Legacy H3 checkpoint is not compatible with %dx%d: %s", width, height, legacy_checkpoint_path)
        return 0
    by_beat = {
        int(record.get("index", index)): record
        for index, record in enumerate(legacy_records)
        if isinstance(record, dict)
    }
    migrated = 0
    recovered_jobs = 0
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    for record in checkpoint.get("clips", []):
        if not isinstance(record, dict) or int(record.get("part_index", -1)) != 0:
            continue
        if record.get("status") == "success" or record.get("job_id"):
            continue
        beat_index = int(record["beat_index"])
        old = by_beat.get(beat_index)
        if not isinstance(old, dict):
            continue
        old_job_id = str(old.get("job_id") or "").strip()
        if old.get("status") in {"queued", "running"} and old_job_id:
            record.update(
                status="queued",
                attempts=max(1, int(old.get("attempts") or 1)),
                job_id=old_job_id,
                submitted_at=old.get("submitted_at"),
                continuity_reset=True,
                chain_position=0,
                legacy_job_recovery=True,
                migrated_from={
                    "checkpoint": str(legacy_checkpoint_path.resolve()),
                    "record_index": beat_index,
                    "remote_job_id": old_job_id,
                },
                error_summary="Recovering an already-submitted legacy H3 job",
            )
            recovered_jobs += 1
            save_checkpoint(checkpoint_path, checkpoint)
            continue
        if old.get("status") != "success":
            continue
        output_value = old.get("output_file")
        if not output_value:
            continue
        output = Path(str(output_value)).resolve()
        if not output.is_file():
            continue
        try:
            metadata = probe_video(output, ffprobe)
            if metadata["width"] != width or metadata["height"] != height:
                continue
            if float(metadata["duration"]) + 1 / 24 < float(record["usable_duration"]):
                continue
            quality = inspect_motion_quality(
                output,
                usable_duration=float(record["usable_duration"]),
                ffmpeg=ffmpeg,
                max_freeze_ratio=max_freeze_ratio,
                max_black_ratio=max_black_ratio,
            )
            opening = keyframes_dir / f"{beat_index + 1:04d}_legacy_opening.png"
            continuation = Path(str(record["frame_target"]))
            if not opening.is_file():
                extract_continuation_frame(
                    output,
                    opening,
                    timestamp_seconds=0.0,
                    media_duration_seconds=float(metadata["duration"]),
                    ffmpeg=ffmpeg,
                )
            extract_continuation_frame(
                output,
                continuation,
                timestamp_seconds=float(record["usable_duration"]),
                media_duration_seconds=float(metadata["duration"]),
                ffmpeg=ffmpeg,
            )
        except (OSError, ValueError, H3GenerationError) as exc:
            LOGGER.warning("Could not migrate legacy H3 beat %d: %s", beat_index + 1, exc)
            continue
        record.update(
            status="success",
            completed_at=utc_now(),
            output_file=str(output),
            output_sha256=old.get("output_sha256") or sha256_file(output),
            continuation_frame=str(continuation.resolve()),
            continuation_frame_sha256=sha256_file(continuation),
            duration_seconds=round(float(metadata["duration"]), 6),
            continuity_reset=True,
            chain_position=0,
            uses_first_frame=False,
            uses_last_frame=False,
            input_frame=None,
            input_frame_sha256=None,
            legacy_reused=True,
            legacy_job_recovery=False,
            migrated_from={
                "checkpoint": str(legacy_checkpoint_path.resolve()),
                "record_index": beat_index,
                "opening_frame": str(opening.resolve()),
            },
            quality={**quality, "source": "legacy-reused"},
            error_summary=None,
        )
        migrated += 1
        save_checkpoint(checkpoint_path, checkpoint)
    if migrated or recovered_jobs:
        LOGGER.info(
            "Migrated %d compatible legacy clips and recovered %d remote jobs into "
            "the continuous checkpoint",
            migrated,
            recovered_jobs,
        )
    return migrated + recovered_jobs


def run_generation(
    plan: Sequence[dict[str, Any]],
    scenes: Sequence[dict[str, Any]],
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    endpoint: str,
    width: int,
    height: int,
    request_timeout: float,
    poll_seconds: float,
    job_timeout: float,
    max_attempts: int,
    max_freeze_ratio: float,
    max_black_ratio: float,
    min_anchor_similarity: float,
    composition_direction: str,
    identity_contexts: Mapping[int, str],
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    client = H3Client(endpoint, timeout=request_timeout)
    client.health()
    previous_scene: int | None = None
    previous_frame: Path | None = None
    records = checkpoint["clips"]
    for index, record in enumerate(records):
        scene_index = int(record["scene_index"])
        if (
            scene_index != previous_scene
            or bool(record.get("continuity_reset"))
            or bool(record.get("legacy_reused"))
            or bool(record.get("legacy_job_recovery"))
        ):
            previous_frame = None
        input_frame = previous_frame if previous_frame and previous_frame.is_file() else None
        input_hash = sha256_file(input_frame) if input_frame is not None else None
        uses_first_frame = input_frame is not None
        target_value = record.get("target_frame")
        target_frame = Path(str(target_value)) if target_value else None
        if target_frame is not None and not target_frame.is_file():
            target_frame = None
        target_hash = sha256_file(target_frame) if target_frame is not None else None
        uses_last_frame = target_frame is not None
        beat_index = int(record.get("beat_index", index))
        _, scene = scene_for_line(int(plan[beat_index].get("start_line", 1)), scenes)
        prompt = build_native_prompt(
            plan[beat_index],
            scene,
            duration_seconds=int(record["requested_duration"]),
            uses_first_frame=uses_first_frame,
            uses_last_frame=uses_last_frame,
            part_index=int(record.get("part_index", 0)),
            part_count=int(record.get("part_count", 1)),
            phase=str(record.get("shot_phase", "")) or None,
            camera=str(record.get("camera_direction", "")),
            composition_direction=composition_direction,
            identity_context=identity_contexts.get(beat_index, ""),
            audio_context=format_audio_context(record.get("audio_cues", [])),
        )
        fingerprint = sha256_text(
            {
                "static_identity": record["static_identity"],
                "prompt": prompt,
                "input_frame_sha256": input_hash,
                "target_frame_sha256": target_hash,
                "uses_first_frame": uses_first_frame,
                "uses_last_frame": uses_last_frame,
            }
        )
        try:
            if validate_cached_record(
                record,
                fingerprint=fingerprint,
                width=width,
                height=height,
                ffprobe=ffprobe,
            ):
                if record.get("fingerprint") != fingerprint:
                    record["fingerprint"] = fingerprint
                    save_checkpoint(checkpoint_path, checkpoint)
                previous_scene = scene_index
                previous_frame = Path(str(record["continuation_frame"]))
                LOGGER.info("[%d/%d] already complete: %s", index + 1, len(records), record["title"])
                continue
        except (OSError, ValueError, H3GenerationError):
            pass
        if record.get("legacy_job_recovery"):
            record["fingerprint"] = fingerprint
        elif record.get("fingerprint") != fingerprint:
            reset_record(record)
        record.update(
            fingerprint=fingerprint,
            uses_first_frame=uses_first_frame,
            uses_last_frame=uses_last_frame,
            input_frame=str(input_frame) if input_frame is not None else None,
            input_frame_sha256=input_hash,
            target_frame=str(target_frame) if target_frame is not None else None,
            target_frame_sha256=target_hash,
        )
        save_checkpoint(checkpoint_path, checkpoint)

        while (
            str(record.get("job_id") or "").strip()
            or int(record.get("attempts") or 0) < max_attempts
        ):
            job_id = str(record.get("job_id") or "").strip()
            if not job_id:
                attempt = int(record.get("attempts") or 0) + 1
                mode = (
                    "FL2VA"
                    if uses_first_frame and uses_last_frame
                    else "I2VA"
                    if uses_first_frame
                    else "L2VA"
                    if uses_last_frame
                    else "T2VA"
                )
                LOGGER.info(
                    "[%d/%d] submitting %s %s (%ss, scene=%s, attempt %d/%d)",
                    index + 1,
                    len(records),
                    mode,
                    record["title"],
                    record["requested_duration"],
                    record["scene_title"],
                    attempt,
                    max_attempts,
                )
                try:
                    job_id = client.submit_request(
                        prompt=prompt,
                        width=width,
                        height=height,
                        duration=int(record["requested_duration"]),
                        first_frame=input_frame,
                        last_frame=target_frame,
                    )
                except (requests.RequestException, H3GenerationError) as exc:
                    record.update(attempts=attempt, status="pending", error_summary=str(exc))
                    save_checkpoint(checkpoint_path, checkpoint)
                    if attempt >= max_attempts:
                        break
                    time.sleep(min(60.0, poll_seconds * attempt))
                    continue
                record.update(
                    attempts=attempt,
                    status="queued",
                    job_id=job_id,
                    submitted_at=utc_now(),
                    error_summary=None,
                )
                save_checkpoint(checkpoint_path, checkpoint)
            try:
                postprocess_ready = False
                record["status"] = "running"
                save_checkpoint(checkpoint_path, checkpoint)
                wait_for_job(
                    client,
                    job_id,
                    poll_seconds=poll_seconds,
                    job_timeout=job_timeout,
                )
                output_path = Path(record["output_target"])
                frame_path = Path(record["frame_target"])
                download_completed_job(client, job_id, output_path)
                try:
                    if record.get("legacy_job_recovery"):
                        metadata = probe_video(output_path, ffprobe)
                        if metadata["width"] != width or metadata["height"] != height:
                            raise H3GenerationError(
                                "Recovered legacy H3 job has incompatible dimensions"
                            )
                        if float(metadata["duration"]) + 1 / 24 < float(
                            record["usable_duration"]
                        ):
                            raise H3GenerationError(
                                "Recovered legacy H3 job is too short for its micro-shot"
                            )
                    else:
                        metadata = validate_downloaded_clip(
                            output_path,
                            width=width,
                            height=height,
                            expected_duration=float(record["expected_duration"]),
                            ffprobe=ffprobe,
                        )
                except OSError as exc:
                    raise H3JobResumeRequired(
                        f"H3 clip was downloaded but local validation could not run: {exc}"
                    ) from exc
                quality = inspect_motion_quality(
                    output_path,
                    usable_duration=min(
                        float(record["usable_duration"]), float(metadata["duration"])
                    ),
                    ffmpeg=ffmpeg,
                    max_freeze_ratio=max_freeze_ratio,
                    max_black_ratio=max_black_ratio,
                )
                anchor_quality = inspect_anchor_quality(
                    output_path,
                    first_frame=input_frame,
                    last_frame=target_frame,
                    media_duration_seconds=float(metadata["duration"]),
                    ffmpeg=ffmpeg,
                    min_anchor_similarity=min_anchor_similarity,
                )
                quality = {**quality, **anchor_quality, "status": "passed"}
                record.update(
                    status="postprocessing",
                    output_file=str(output_path),
                    output_sha256=sha256_file(output_path),
                    duration_seconds=round(float(metadata["duration"]), 6),
                    quality=quality,
                    error_summary=None,
                )
                save_checkpoint(checkpoint_path, checkpoint)
                postprocess_ready = True
                extract_continuation_frame(
                    output_path,
                    frame_path,
                    timestamp_seconds=min(
                        float(record["usable_duration"]), float(metadata["duration"])
                    ),
                    media_duration_seconds=float(metadata["duration"]),
                    ffmpeg=ffmpeg,
                )
                record.update(
                    status="success",
                    completed_at=utc_now(),
                    output_file=str(output_path),
                    output_sha256=sha256_file(output_path),
                    continuation_frame=str(frame_path),
                    continuation_frame_sha256=sha256_file(frame_path),
                    duration_seconds=round(float(metadata["duration"]), 6),
                    legacy_reused=bool(record.get("legacy_job_recovery")),
                    legacy_job_recovery=False,
                    error_summary=None,
                )
                save_checkpoint(checkpoint_path, checkpoint)
                previous_scene = scene_index
                previous_frame = frame_path
                break
            except KeyboardInterrupt:
                save_checkpoint(checkpoint_path, checkpoint)
                raise
            except H3JobResumeRequired as exc:
                record.update(status="queued", error_summary=str(exc))
                save_checkpoint(checkpoint_path, checkpoint)
                raise
            except (OSError, ValueError, requests.RequestException, H3GenerationError) as exc:
                LOGGER.error("[%d/%d] native H3 attempt failed: %s", index + 1, len(records), exc)
                if isinstance(exc, H3MotionQualityError):
                    failures = record.get("quality_failures")
                    if not isinstance(failures, list):
                        failures = []
                    failures.append(
                        {
                            "attempt": int(record.get("attempts") or 0),
                            "checked_at": utc_now(),
                            **exc.metrics,
                        }
                    )
                    record["quality_failures"] = failures
                    record["quality"] = exc.metrics
                if postprocess_ready:
                    record.update(
                        status="postprocess_failed",
                        completed_at=utc_now(),
                        continuation_frame=None,
                        continuation_frame_sha256=None,
                        error_summary=str(exc),
                    )
                    save_checkpoint(checkpoint_path, checkpoint)
                    raise H3GenerationError(
                        "H3 clip downloaded successfully but local continuation-frame "
                        "postprocessing failed; the remote job is preserved for resume"
                    ) from exc
                record.update(
                    status="pending",
                    job_id=None,
                    completed_at=utc_now(),
                    output_file=None,
                    output_sha256=None,
                    continuation_frame=None,
                    continuation_frame_sha256=None,
                    duration_seconds=None,
                    legacy_job_recovery=False,
                    error_summary=str(exc),
                )
                save_checkpoint(checkpoint_path, checkpoint)
                if int(record.get("attempts") or 0) < max_attempts:
                    time.sleep(min(60.0, poll_seconds * int(record["attempts"])))
        if record.get("status") != "success":
            raise H3GenerationError(
                f"Native H3 clip {index + 1}/{len(records)} failed after "
                f"{record.get('attempts', 0)} attempts: {record.get('error_summary')}"
            )
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate H3-native chained story clips")
    parser.add_argument(
        "--mode",
        choices=("native-chain", "continuous-chain"),
        default="native-chain",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-audit-checkpoint", type=Path)
    parser.add_argument("--character-cards", type=Path)
    parser.add_argument("--visual-memory", type=Path)
    parser.add_argument(
        "--performance-directions",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--scene-segments", type=Path, required=True)
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--keyframes-dir", type=Path)
    parser.add_argument("--shot-plan-output", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://172.31.102.189:8189")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--minimum-duration", type=int, default=5)
    parser.add_argument("--maximum-duration", type=int, default=10)
    parser.add_argument("--max-chain-length", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--job-timeout", type=float, default=14400.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-freeze-ratio", type=float, default=0.65)
    parser.add_argument("--max-black-ratio", type=float, default=0.20)
    parser.add_argument("--min-anchor-similarity", type=float, default=0.90)
    parser.add_argument("--composition-direction", default="")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise SystemExit("H3 width and height must be positive multiples of 16")
    if not 5 <= args.minimum_duration <= args.maximum_duration <= 15:
        raise SystemExit("H3 duration range must satisfy 5 <= minimum <= maximum <= 15")
    if args.max_chain_length < 1:
        raise SystemExit("--max-chain-length must be positive")
    if (
        not 0 <= args.max_freeze_ratio <= 1
        or not 0 <= args.max_black_ratio <= 1
        or not 0 <= args.min_anchor_similarity <= 1
    ):
        raise SystemExit("H3 quality ratios must be between 0 and 1")
    plan = load_plan(args.plan)
    audited_plan = load_audited_plan(plan, args.prompt_audit_checkpoint)
    timeline = build_timeline(
        audited_plan,
        novel_path=args.novel,
        labels_path=args.labels,
        segments_dir=args.segments_dir,
        audio_path=args.audio,
        ffprobe=args.ffprobe,
    )
    scenes = load_scene_segments(args.scene_segments)
    identity_contexts = build_identity_contexts(
        audited_plan,
        character_cards_path=args.character_cards,
        visual_memory_path=args.visual_memory,
    )
    LOGGER.info(
        "Loaded continuity-bible context for %d/%d story beats",
        len(identity_contexts),
        len(audited_plan),
    )
    records = build_records(
        audited_plan,
        timeline,
        scenes,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        limit=args.limit,
        mode=args.mode,
        max_chain_length=args.max_chain_length,
    )
    segment_count = len(discover_segment_files(args.segments_dir))
    entries = load_subtitle_entries(
        args.novel,
        args.labels,
        expected_segment_count=segment_count,
    )
    timestamps = build_segment_timestamps(entries, args.segments_dir)
    attach_audio_locked_cues(
        records,
        timeline,
        entries,
        timestamps,
        performance_paths=args.performance_directions,
    )
    scene_source_hash = sha256_file(args.scene_segments)
    if args.mode == "continuous-chain" and args.shot_plan_output is not None:
        write_shot_plan(
            args.shot_plan_output,
            records,
            scene_source_hash=scene_source_hash,
        )
    checkpoint = prepare_checkpoint(
        args.checkpoint,
        records,
        endpoint=args.endpoint,
        width=args.width,
        height=args.height,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        scene_source_hash=scene_source_hash,
        resume=args.resume,
        mode=args.mode,
        max_chain_length=args.max_chain_length,
        max_freeze_ratio=args.max_freeze_ratio,
        max_black_ratio=args.max_black_ratio,
        min_anchor_similarity=args.min_anchor_similarity,
    )
    save_checkpoint(args.checkpoint, checkpoint)
    resets = sum(
        1
        for index, record in enumerate(records)
        if bool(record.get("continuity_reset"))
    )
    short = sum(record["usable_duration"] < record["expected_duration"] for record in records)
    LOGGER.info(
        "Prepared %d %s H3 clips (%d chain resets, %d clips trimmed at audio boundaries)",
        len(records),
        args.mode,
        resets,
        short,
    )
    if args.dry_run:
        return 0
    if args.mode == "continuous-chain" and args.legacy_checkpoint is not None:
        migrate_legacy_records(
            checkpoint,
            args.legacy_checkpoint,
            checkpoint_path=args.checkpoint,
            keyframes_dir=args.keyframes_dir or args.frames_dir.parent / "keyframes",
            width=args.width,
            height=args.height,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            max_freeze_ratio=args.max_freeze_ratio,
            max_black_ratio=args.max_black_ratio,
        )
    run_generation(
        audited_plan,
        scenes,
        checkpoint,
        checkpoint_path=args.checkpoint,
        endpoint=args.endpoint,
        width=args.width,
        height=args.height,
        request_timeout=args.request_timeout,
        poll_seconds=args.poll_seconds,
        job_timeout=args.job_timeout,
        max_attempts=args.max_attempts,
        max_freeze_ratio=args.max_freeze_ratio,
        max_black_ratio=args.max_black_ratio,
        min_anchor_similarity=args.min_anchor_similarity,
        composition_direction=args.composition_direction,
        identity_contexts=identity_contexts,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    LOGGER.info("All H3-native clips are complete: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
