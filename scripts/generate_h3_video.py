"""Compose MiniMax H3 story clips with the existing production audio.

Each illustration interval is rendered as a frame-exact H.264 segment.  The
interval holds its source illustration and, when available, plays an H3 FL2VA
clip at the end so the generated clip lands on the following illustration.
H3's generated audio is always discarded; VoxCPM speech, BGM, and subtitles
remain the authoritative production timeline.  ``continuous-chain`` mode
requires dynamic clips to cover every output frame and contains no still-frame
padding path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.subtitles import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_LINES,
    build_segment_timestamps,
    discover_segment_files,
    load_subtitle_entries,
    write_srt,
)
from scripts.generate_h3_clips import (  # noqa: E402
    CHECKPOINT_VERSION as H3_CHECKPOINT_VERSION,
    find_image,
    load_plan,
    sha256_file,
    sha256_text,
)
from scripts.generate_h3_native_clips import CONTINUOUS_CHECKPOINT_VERSION  # noqa: E402
from scripts.generate_video import (  # noqa: E402
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    build_illustration_timeline,
    build_source_line_timeline,
    build_subtitle_filter,
    ffmpeg_supports_filter,
    probe_media_duration,
    validate_audio_timeline,
)


LOGGER = logging.getLogger("h3_video")
RENDER_CHECKPOINT_VERSION = 1


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def ffconcat_quote(path: Path) -> str:
    value = path.resolve().as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError(f"Media path cannot contain a newline: {path}")
    return "'" + value.replace("'", "'\\''") + "'"


def scale_filter(width: int, height: int, fps: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps},settb=expr=1/{fps}"
    )


def run_command(command: list[str], *, timeout: float | None = None) -> None:
    result = subprocess.run(command, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed ({result.returncode}): {' '.join(command[:8])}")


def load_h3_checkpoint(path: Path, expected_transitions: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid H3 checkpoint: {path}")
    mode = str(payload.get("mode", "illustration-bridge"))
    expected_version = (
        CONTINUOUS_CHECKPOINT_VERSION
        if mode == "continuous-chain"
        else H3_CHECKPOINT_VERSION
    )
    if payload.get("version") != expected_version:
        raise ValueError(f"Invalid H3 checkpoint version: {path}")
    records = payload.get("clips")
    if not isinstance(records, list):
        raise ValueError("H3 checkpoint clips must be a list")
    if mode != "continuous-chain" and len(records) != expected_transitions:
        raise ValueError(
            f"H3 checkpoint must contain {expected_transitions} transition records"
        )
    if mode == "continuous-chain" and not records:
        raise ValueError("Continuous H3 checkpoint is empty")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("index") != index:
            raise ValueError(f"H3 checkpoint record {index} is invalid or misordered")
        if record.get("status") not in {"success", "skipped"}:
            raise ValueError(f"H3 transition {index + 1} is not complete")
        if record.get("status") == "success":
            output = record.get("output_file")
            if not output or not Path(str(output)).is_file():
                raise ValueError(f"H3 transition {index + 1} output is missing")
    if mode == "continuous-chain":
        groups: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            if record.get("status") != "success":
                raise ValueError(f"Continuous H3 clip {int(record['index']) + 1} is not complete")
            beat_index = int(record.get("beat_index", -1))
            if not 0 <= beat_index < expected_transitions:
                raise ValueError(f"Continuous H3 clip has invalid beat_index={beat_index}")
            groups.setdefault(beat_index, []).append(record)
        if set(groups) != set(range(expected_transitions)):
            missing = sorted(set(range(expected_transitions)) - set(groups))
            raise ValueError(f"Continuous H3 checkpoint misses beats: {missing[:10]}")
        for beat_index, group in groups.items():
            ordered = sorted(group, key=lambda item: int(item.get("part_index", -1)))
            if [int(item.get("part_index", -1)) for item in ordered] != list(range(len(ordered))):
                raise ValueError(f"Continuous H3 beat {beat_index + 1} parts are misordered")
            if any(int(item.get("part_count", -1)) != len(ordered) for item in ordered):
                raise ValueError(f"Continuous H3 beat {beat_index + 1} part_count is invalid")
            cursor = 0.0
            for item in ordered:
                start = float(item.get("coverage_start_seconds", -1))
                end = float(item.get("coverage_end_seconds", -1))
                if abs(start - cursor) > 1e-5 or end <= start:
                    raise ValueError(
                        f"Continuous H3 beat {beat_index + 1} has a coverage gap at {cursor:.6f}s"
                    )
                cursor = end
            interval = float(ordered[0].get("interval_duration", 0))
            if abs(cursor - interval) > 1e-5:
                raise ValueError(
                    f"Continuous H3 beat {beat_index + 1} covers {cursor:.6f}s, "
                    f"expected {interval:.6f}s"
                )
    return payload, records


def build_timeline_and_subtitles(
    plan: Sequence[dict[str, Any]],
    *,
    novel_path: Path,
    labels_path: Path,
    segments_dir: Path,
    audio_path: Path,
    subtitle_path: Path,
    subtitle_label_mode: str,
    max_subtitle_chars: int,
    max_subtitle_lines: int,
    ffprobe: str,
) -> tuple[list[dict[str, Any]], int]:
    segment_count = len(discover_segment_files(segments_dir))
    if segment_count <= 0:
        raise ValueError(f"No TTS WAV segments found: {segments_dir}")
    entries = load_subtitle_entries(
        novel_path,
        labels_path,
        label_mode=subtitle_label_mode,
        expected_segment_count=segment_count,
    )
    timestamps = build_segment_timestamps(entries, segments_dir)
    write_srt(
        subtitle_path,
        entries,
        timestamps,
        max_chars=max_subtitle_chars,
        max_lines=max_subtitle_lines,
    )
    total_duration_ms = int(timestamps[-1]["end_ms"])
    audio_duration = probe_media_duration(audio_path, ffprobe)
    validate_audio_timeline(audio_duration, total_duration_ms)
    timeline = build_illustration_timeline(
        [dict(item) for item in plan],
        [int(item["start_ms"]) for item in timestamps],
        source_line_timeline=build_source_line_timeline(timestamps),
        total_duration_ms=total_duration_ms,
    )
    return timeline, total_duration_ms


def segment_specs(
    plan: Sequence[dict[str, Any]],
    timeline: Sequence[dict[str, Any]],
    h3_records: Sequence[dict[str, Any]],
    *,
    h3_mode: str,
    images_dir: Path | None,
    segments_output_dir: Path,
    fps: int,
) -> list[dict[str, Any]]:
    if len(plan) != len(timeline):
        raise ValueError("Illustration plan and timeline counts do not match")
    specs: list[dict[str, Any]] = []
    continuous_groups: dict[int, list[dict[str, Any]]] = {}
    if h3_mode == "continuous-chain":
        for record in h3_records:
            continuous_groups.setdefault(int(record.get("beat_index", -1)), []).append(record)
    for index, item in enumerate(timeline):
        start_frame = round(int(item["start_ms"]) * fps / 1000)
        end_frame = round(int(item["end_ms"]) * fps / 1000)
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            raise ValueError(f"Illustration beat {index + 1} has no timeline frames")
        clip: Path | None = None
        clip_duration: float | None = None
        placement = "end"
        record = h3_records[index] if index < len(h3_records) else None
        if h3_mode == "continuous-chain":
            group = sorted(
                continuous_groups.get(index, []),
                key=lambda value: int(value.get("part_index", -1)),
            )
            if not group:
                raise ValueError(f"Continuous H3 beat {index + 1} has no dynamic clips")
            clips: list[dict[str, Any]] = []
            interval_duration = float(group[0].get("interval_duration", 0))
            if interval_duration <= 0:
                raise ValueError(f"Continuous H3 beat {index + 1} has invalid duration")
            allocated_frames = 0
            for part_index, part in enumerate(group):
                if part.get("status") != "success":
                    raise ValueError(
                        f"Continuous H3 beat {index + 1} part {part_index + 1} is incomplete"
                    )
                part_start = round(
                    float(part["coverage_start_seconds"]) / interval_duration * frame_count
                )
                part_end = (
                    frame_count
                    if part_index == len(group) - 1
                    else round(
                        float(part["coverage_end_seconds"])
                        / interval_duration
                        * frame_count
                    )
                )
                target_frames = part_end - part_start
                if part_start != allocated_frames or target_frames <= 0:
                    raise ValueError(
                        f"Continuous H3 beat {index + 1} has a frame coverage gap at "
                        f"frame {allocated_frames}"
                    )
                path = Path(str(part.get("output_file", ""))).resolve()
                if not path.is_file():
                    raise ValueError(
                        f"Continuous H3 beat {index + 1} part {part_index + 1} is missing"
                    )
                duration = float(part.get("duration_seconds") or 0)
                available_frames = int(duration * fps + 1e-6)
                if available_frames < target_frames:
                    raise ValueError(
                        f"Continuous H3 beat {index + 1} part {part_index + 1} has only "
                        f"{available_frames} dynamic frames for {target_frames} required frames"
                    )
                clips.append(
                    {
                        "record_index": int(part["index"]),
                        "path": path,
                        "duration": duration,
                        "target_frames": target_frames,
                        "sha256": sha256_file(path),
                    }
                )
                allocated_frames = part_end
            if allocated_frames != frame_count:
                raise ValueError(
                    f"Continuous H3 beat {index + 1} covers {allocated_frames}/{frame_count} frames"
                )
            image = None
            output = segments_output_dir / f"{index + 1:04d}.mp4"
            fingerprint = sha256_text(
                {
                    "mode": h3_mode,
                    "index": index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": fps,
                    "clips": [
                        {
                            "record_index": value["record_index"],
                            "path": str(value["path"]),
                            "duration": value["duration"],
                            "target_frames": value["target_frames"],
                            "sha256": value["sha256"],
                        }
                        for value in clips
                    ],
                }
            )
            specs.append(
                {
                    "index": index,
                    "title": str(plan[index].get("title", "")),
                    "image": image,
                    "clip": None,
                    "clips": clips,
                    "clip_duration": None,
                    "placement": "continuous",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_count": frame_count,
                    "output": output.resolve(),
                    "fingerprint": fingerprint,
                }
            )
            continue
        if h3_mode == "native-chain":
            if record is None or record.get("status") != "success":
                raise ValueError(f"Native H3 clip {index + 1} is not complete")
            clip = Path(str(record["output_file"])).resolve()
            clip_duration = float(record["duration_seconds"])
            continuation = record.get("continuation_frame")
            if not continuation or not Path(str(continuation)).is_file():
                raise ValueError(f"Native H3 clip {index + 1} has no continuation frame")
            image = Path(str(continuation)).resolve()
            placement = "start"
        else:
            if images_dir is None:
                raise ValueError("Illustration-bridge mode requires --images-dir")
            image = find_image(images_dir, index)
        if h3_mode != "native-chain" and record is not None and record.get("status") == "success":
            clip = Path(str(record["output_file"])).resolve()
            clip_duration = float(record["duration_seconds"])
        output = segments_output_dir / f"{index + 1:04d}.mp4"
        fingerprint = sha256_text(
            {
                "index": index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "image": {
                    "path": str(image),
                    "size": image.stat().st_size,
                    "sha256": sha256_file(image),
                },
                "clip": (
                    {
                        "path": str(clip),
                        "size": clip.stat().st_size,
                        "sha256": sha256_file(clip),
                        "duration": clip_duration,
                    }
                    if clip is not None
                    else None
                ),
                "placement": placement,
            }
        )
        specs.append(
            {
                "index": index,
                "title": str(plan[index].get("title", "")),
                "image": image,
                "clip": clip,
                "clip_duration": clip_duration,
                "placement": placement,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "output": output.resolve(),
                "fingerprint": fingerprint,
            }
        )
    return specs


def prepare_render_checkpoint(
    path: Path,
    specs: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    resume: bool,
) -> dict[str, Any]:
    source_hash = sha256_text(
        {
            "specs": [item["fingerprint"] for item in specs],
            "width": width,
            "height": height,
            "fps": fps,
            "crf": crf,
            "preset": preset,
        }
    )
    fresh = {
        "version": RENDER_CHECKPOINT_VERSION,
        "source_hash": source_hash,
        "width": width,
        "height": height,
        "fps": fps,
        "crf": crf,
        "preset": preset,
        "updated_at": utc_now(),
        "segments": [
            {
                "index": item["index"],
                "status": "skipped" if item["frame_count"] <= 0 else "pending",
                "fingerprint": item["fingerprint"],
                "frame_count": item["frame_count"],
                "output_file": None,
                "error_summary": None,
            }
            for item in specs
        ],
    }
    if not resume or not path.is_file():
        return fresh
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fresh
    compatible = (
        isinstance(old, dict)
        and old.get("version") == RENDER_CHECKPOINT_VERSION
        and old.get("source_hash") == source_hash
        and isinstance(old.get("segments"), list)
        and len(old["segments"]) == len(specs)
    )
    if not compatible:
        LOGGER.warning("Ignoring incompatible H3 render checkpoint: %s", path)
        return fresh
    for current, previous in zip(fresh["segments"], old["segments"]):
        if not isinstance(previous, dict):
            continue
        if previous.get("fingerprint") != current["fingerprint"]:
            continue
        current.update(previous)
        if current.get("status") == "success":
            output = current.get("output_file")
            if not output or not Path(str(output)).is_file():
                current.update(status="pending", output_file=None, error_summary="Output missing")
    return fresh


def save_render_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(path, checkpoint)


def render_segment(
    spec: dict[str, Any],
    *,
    output: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    ffmpeg: str,
) -> None:
    total_frames = int(spec["frame_count"])
    if total_frames <= 0:
        raise ValueError("Cannot render a zero-frame segment")
    clip = spec.get("clip")
    clip_frames = 0
    if clip is not None:
        clip_frames = min(total_frames, max(1, round(float(spec["clip_duration"]) * fps)))
    hold_frames = total_frames - clip_frames
    common_output = [
        "-r",
        str(fps),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-g",
        str(fps * 2),
        "-sc_threshold",
        "0",
        "-an",
        "-frames:v",
        str(total_frames),
        "-movflags",
        "+faststart",
    ]
    base_filter = scale_filter(width, height, fps)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.rendering{output.suffix}")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    continuous_clips = spec.get("clips")
    if isinstance(continuous_clips, list) and continuous_clips:
        target_sum = sum(int(item["target_frames"]) for item in continuous_clips)
        if target_sum != total_frames:
            raise ValueError(
                f"Continuous segment covers {target_sum}/{total_frames} frames"
            )
        inputs: list[str] = []
        for item in continuous_clips:
            inputs.extend(["-i", str(Path(item["path"]))])
        if len(continuous_clips) == 1:
            target = int(continuous_clips[0]["target_frames"])
            command = [
                ffmpeg,
                "-y",
                "-nostdin",
                *inputs,
                "-vf",
                f"{base_filter},trim=end_frame={target},setpts=N/({fps}*TB)",
                *common_output,
                str(temporary),
            ]
        else:
            filters = []
            labels = []
            for input_index, item in enumerate(continuous_clips):
                label = f"v{input_index}"
                labels.append(f"[{label}]")
                filters.append(
                    f"[{input_index}:v]{base_filter},"
                    f"trim=end_frame={int(item['target_frames'])},"
                    f"setpts=PTS-STARTPTS[{label}]"
                )
            filters.append(
                "".join(labels)
                + f"concat=n={len(labels)}:v=1:a=0[joined]"
            )
            filters.append(
                f"[joined]fps={fps},settb=expr=1/{fps},"
                f"trim=end_frame={total_frames},setpts=N/({fps}*TB)[outv]"
            )
            command = [
                ffmpeg,
                "-y",
                "-nostdin",
                *inputs,
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[outv]",
                *common_output,
                str(temporary),
            ]
    elif clip is None or clip_frames <= 0:
        command = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(Path(spec["image"])),
            "-vf",
            f"{base_filter},trim=end_frame={total_frames},setpts=PTS-STARTPTS",
            *common_output,
            str(temporary),
        ]
    elif hold_frames <= 0:
        command = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-i",
            str(Path(clip)),
            "-vf",
            f"{base_filter},trim=end_frame={total_frames},setpts=PTS-STARTPTS",
            *common_output,
            str(temporary),
        ]
    else:
        if spec.get("placement") == "start":
            filter_complex = (
                f"[1:v]{base_filter},trim=end_frame={clip_frames},setpts=PTS-STARTPTS[motion];"
                f"[0:v]{base_filter},trim=end_frame={hold_frames},setpts=PTS-STARTPTS[still];"
                "[motion][still]concat=n=2:v=1:a=0[joined];"
                f"[joined]fps={fps},settb=expr=1/{fps},"
                f"setpts=N/({fps}*TB)[outv]"
            )
        else:
            filter_complex = (
                f"[0:v]{base_filter},trim=end_frame={hold_frames},setpts=PTS-STARTPTS[still];"
                f"[1:v]{base_filter},trim=end_frame={clip_frames},setpts=PTS-STARTPTS[motion];"
                "[still][motion]concat=n=2:v=1:a=0[joined];"
                f"[joined]fps={fps},settb=expr=1/{fps},"
                f"setpts=N/({fps}*TB)[outv]"
            )
        command = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(Path(spec["image"])),
            "-i",
            str(Path(clip)),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            *common_output,
            str(temporary),
        ]
    try:
        run_command(command)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def render_all_segments(
    specs: Sequence[dict[str, Any]],
    *,
    checkpoint_path: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    ffmpeg: str,
    ffprobe: str,
    resume: bool,
) -> tuple[dict[str, Any], list[Path]]:
    checkpoint = prepare_render_checkpoint(
        checkpoint_path,
        specs,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
        resume=resume,
    )
    save_render_checkpoint(checkpoint_path, checkpoint)
    rendered: list[Path] = []
    total = len(specs)
    for spec, record in zip(specs, checkpoint["segments"]):
        if record.get("status") == "skipped":
            continue
        output = Path(spec["output"])
        expected_duration = int(spec["frame_count"]) / fps
        if record.get("status") == "success" and output.is_file():
            try:
                duration = probe_media_duration(output, ffprobe)
                if abs(duration - expected_duration) <= 2 / fps:
                    rendered.append(output)
                    continue
            except (OSError, RuntimeError, ValueError):
                pass
        LOGGER.info(
            "[%d/%d] rendering hybrid segment: %s",
            int(spec["index"]) + 1,
            total,
            spec["title"],
        )
        record.update(status="running", output_file=None, error_summary=None)
        save_render_checkpoint(checkpoint_path, checkpoint)
        try:
            render_segment(
                spec,
                output=output,
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                preset=preset,
                ffmpeg=ffmpeg,
            )
            duration = probe_media_duration(output, ffprobe)
            if abs(duration - expected_duration) > 2 / fps:
                raise RuntimeError(
                    f"segment duration {duration:.3f}s != {expected_duration:.3f}s"
                )
            record.update(
                status="success",
                output_file=str(output),
                duration_seconds=round(duration, 6),
                error_summary=None,
            )
            save_render_checkpoint(checkpoint_path, checkpoint)
            rendered.append(output)
        except KeyboardInterrupt:
            save_render_checkpoint(checkpoint_path, checkpoint)
            raise
        except Exception as exc:
            record.update(status="failed", output_file=None, error_summary=str(exc))
            save_render_checkpoint(checkpoint_path, checkpoint)
            raise
    return checkpoint, rendered


def compose_final_video(
    segment_files: Sequence[Path],
    *,
    concat_path: Path,
    audio_path: Path,
    subtitle_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    audio_bitrate: str,
    subtitle_font: str,
    subtitle_font_size: int,
    subtitle_fonts_dir: Path | None,
    total_duration_ms: int,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    if not segment_files:
        raise ValueError("No rendered hybrid segments are available")
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path.write_text(
        "\n".join(f"file {ffconcat_quote(path)}" for path in segment_files) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not ffmpeg_supports_filter(ffmpeg, "subtitles"):
        raise RuntimeError("Configured FFmpeg does not include the subtitles/libass filter")
    subtitle_filter = build_subtitle_filter(
        subtitle_path,
        font_name=subtitle_font,
        font_size=subtitle_font_size,
        fonts_dir=subtitle_fonts_dir,
        video_width=width,
        video_height=height,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.rendering{output_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-i",
        str(audio_path.resolve()),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        "-shortest",
        "-t",
        f"{total_duration_ms / 1000:.3f}",
        str(temporary),
    ]
    started = time.perf_counter()
    try:
        run_command(command)
        rendered_duration = probe_media_duration(temporary, ffprobe)
        expected_duration = total_duration_ms / 1000
        if abs(rendered_duration - expected_duration) > 0.25:
            raise RuntimeError(
                f"Final video duration {rendered_duration:.3f}s != {expected_duration:.3f}s"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    LOGGER.info(
        "Completed H3 hybrid video in %.1fs: %s",
        time.perf_counter() - started,
        output_path,
    )


def video_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must use WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("video dimensions must be positive")
    return width, height


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose an H3 hybrid illustrated audiobook")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Legacy illustrations used only by illustration-bridge mode",
    )
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--segments-output-dir", type=Path, required=True)
    parser.add_argument("--render-checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=video_size, default=(DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT))
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--subtitle-output", type=Path, required=True)
    parser.add_argument("--subtitle-font", default="SimHei")
    parser.add_argument("--subtitle-font-size", type=positive_int, default=42)
    parser.add_argument("--subtitle-fonts-dir", type=Path)
    parser.add_argument(
        "--subtitle-label-mode",
        choices=("auto", "line", "parsed-line", "dialogue"),
        default="auto",
    )
    parser.add_argument("--max-subtitle-chars", type=positive_int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-subtitle-lines", type=positive_int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--fps", type=positive_int, default=25)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--audio-bitrate", default="256k")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    width, height = args.size
    plan = load_plan(args.plan)
    raw_h3 = json.loads(args.h3_checkpoint.read_text(encoding="utf-8"))
    h3_mode = str(raw_h3.get("mode", "illustration-bridge")) if isinstance(raw_h3, dict) else ""
    expected_records = (
        len(plan)
        if h3_mode in {"native-chain", "continuous-chain"}
        else len(plan) - 1
    )
    h3_checkpoint, h3_records = load_h3_checkpoint(args.h3_checkpoint, expected_records)
    timeline, total_duration_ms = build_timeline_and_subtitles(
        plan,
        novel_path=args.novel,
        labels_path=args.labels,
        segments_dir=args.segments_dir,
        audio_path=args.audio,
        subtitle_path=args.subtitle_output,
        subtitle_label_mode=args.subtitle_label_mode,
        max_subtitle_chars=args.max_subtitle_chars,
        max_subtitle_lines=args.max_subtitle_lines,
        ffprobe=args.ffprobe,
    )
    specs = segment_specs(
        plan,
        timeline,
        h3_records,
        h3_mode=h3_mode,
        images_dir=args.images_dir,
        segments_output_dir=args.segments_output_dir,
        fps=args.fps,
    )
    _, rendered = render_all_segments(
        specs,
        checkpoint_path=args.render_checkpoint,
        width=width,
        height=height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        resume=args.resume,
    )
    dependencies = [
        args.plan,
        args.audio,
        args.h3_checkpoint,
        args.render_checkpoint,
        args.subtitle_output,
        *rendered,
    ]
    cached = (
        not args.force_output
        and args.output.is_file()
        and args.output.stat().st_size > 0
        and all(args.output.stat().st_mtime_ns >= path.stat().st_mtime_ns for path in dependencies)
        and abs(probe_media_duration(args.output, args.ffprobe) - total_duration_ms / 1000) <= 0.25
    )
    if cached:
        LOGGER.info("Using cached H3 hybrid video: %s", args.output)
        return 0
    concat_path = args.render_checkpoint.with_name(f"{args.render_checkpoint.stem}.ffconcat")
    compose_final_video(
        rendered,
        concat_path=concat_path,
        audio_path=args.audio,
        subtitle_path=args.subtitle_output,
        output_path=args.output,
        width=width,
        height=height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
        subtitle_font=args.subtitle_font,
        subtitle_font_size=args.subtitle_font_size,
        subtitle_fonts_dir=args.subtitle_fonts_dir,
        total_duration_ms=total_duration_ms,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    LOGGER.info("H3 source checkpoint: %s", h3_checkpoint.get("source_hash"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
