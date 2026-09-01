"""Generate an H3-native illustrated audiobook without anchoring every old image.

The first beat of each coarse story/BGM scene uses text-to-video.  Subsequent
beats in the same scene use the prior H3 clip's continuation frame as their
first-frame reference.  This preserves local visual continuity while allowing
H3 to establish higher-quality compositions instead of inheriting all legacy
illustrations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


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
    safe_title,
    sha256_file,
    sha256_text,
    utc_now,
    validate_downloaded_clip,
    wait_for_job,
)


LOGGER = logging.getLogger("h3_native")


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


def build_native_prompt(
    item: Mapping[str, Any],
    scene: Mapping[str, Any],
    *,
    duration_seconds: int,
    uses_first_frame: bool,
) -> str:
    visual = str(item.get("prompt", "")).strip()
    description = str(item.get("description", "")).strip()
    title = str(item.get("title", "story beat")).strip()
    scene_title = str(scene.get("title", "story scene")).strip()
    scene_description = str(scene.get("description", "")).strip()
    reference = (
        "For the target video, at 0.00 seconds, Picture 1 is fully referenced.\n\n"
        if uses_first_frame
        else ""
    )
    opening = (
        "Continue naturally from Picture 1, preserving the exact same character faces, "
        "hair, clothing, age, body proportions, objects, palette, linework, lighting, "
        "camera geography, and environment. "
        if uses_first_frame
        else "Establish a polished original composition for this story scene. "
    )
    return (
        reference
        + "integrated_multimodal_description: [Shot 1] High-end 2D anime feature-film "
        f"look, coherent cinematic direction. {opening}"
        "Depict only literal visible story elements. Use subtle breathing, natural blinking, "
        "cloth and hair motion, physically plausible object movement, restrained acting, "
        "stable anatomy, and one motivated camera move. Do not invent extra people, animals, "
        "props, costumes, or supernatural effects. Avoid face drift, body morphing, sudden "
        "wardrobe changes, or unmotivated cuts. "
        f"Scene ({scene_title}): {scene_description} "
        f"Current beat ({title}): {description} Visual target: {visual} "
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
) -> list[dict[str, Any]]:
    count = len(plan) if limit is None else min(len(plan), max(0, limit))
    records = []
    for index in range(count):
        item = plan[index]
        interval = int(timeline[index]["duration_ms"]) / 1000
        requested, expected, usable = requested_duration_for_interval(
            interval,
            minimum_duration=minimum_duration,
            maximum_duration=maximum_duration,
        )
        scene_index, scene = scene_for_line(int(item.get("start_line", 1)), scenes)
        title = safe_title(item.get("title"), f"clip_{index + 1:04d}")
        static_identity = sha256_text(
            {
                "index": index,
                "item": item,
                "timeline": timeline[index],
                "scene_index": scene_index,
                "scene": scene,
                "requested_duration": requested,
                "expected_duration": expected,
                "usable_duration": usable,
            }
        )
        records.append(
            {
                "index": index,
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
                "interval_duration": interval,
                "uses_first_frame": None,
                "input_frame": None,
                "input_frame_sha256": None,
                "static_identity": static_identity,
                "fingerprint": None,
                "error_summary": None,
                "output_target": str(
                    (output_dir / f"{index + 1:04d}_{title}.mp4").resolve()
                ),
                "frame_target": str(
                    (frames_dir / f"{index + 1:04d}_{title}_continuation.png").resolve()
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
) -> dict[str, Any]:
    source_hash = sha256_text(
        {
            "records": [record["static_identity"] for record in records],
            "scene_source_hash": scene_source_hash,
            "width": width,
            "height": height,
            "minimum_duration": minimum_duration,
            "maximum_duration": maximum_duration,
        }
    )
    fresh = {
        "version": CHECKPOINT_VERSION,
        "mode": "native-chain",
        "endpoint": endpoint.rstrip("/"),
        "width": width,
        "height": height,
        "minimum_duration": minimum_duration,
        "maximum_duration": maximum_duration,
        "scene_source_hash": scene_source_hash,
        "source_hash": source_hash,
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
    compatible = (
        isinstance(old, dict)
        and old.get("version") == CHECKPOINT_VERSION
        and old.get("mode") == "native-chain"
        and old.get("endpoint") == endpoint.rstrip("/")
        and old.get("source_hash") == source_hash
        and isinstance(old.get("clips"), list)
        and len(old["clips"]) == len(records)
    )
    if not compatible:
        LOGGER.warning("Ignoring incompatible H3 native checkpoint: %s", path)
        return fresh
    for current, previous in zip(fresh["clips"], old["clips"]):
        if not isinstance(previous, dict):
            continue
        if previous.get("static_identity") != current["static_identity"]:
            continue
        output_target = current["output_target"]
        frame_target = current["frame_target"]
        current.update(previous)
        current["output_target"] = output_target
        current["frame_target"] = frame_target
    return fresh


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(path, checkpoint)


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
    if record.get("status") != "success" or record.get("fingerprint") != fingerprint:
        return False
    output = record.get("output_file")
    frame = record.get("continuation_frame")
    if not output or not frame:
        return False
    output_path = Path(str(output))
    frame_path = Path(str(frame))
    if not output_path.is_file() or not frame_path.is_file():
        return False
    validate_downloaded_clip(
        output_path,
        width=width,
        height=height,
        expected_duration=float(record["expected_duration"]),
        ffprobe=ffprobe,
    )
    return True


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
        if scene_index != previous_scene:
            previous_frame = None
        input_frame = previous_frame if previous_frame and previous_frame.is_file() else None
        input_hash = sha256_file(input_frame) if input_frame is not None else None
        uses_first_frame = input_frame is not None
        _, scene = scene_for_line(int(plan[index].get("start_line", 1)), scenes)
        prompt = build_native_prompt(
            plan[index],
            scene,
            duration_seconds=int(record["requested_duration"]),
            uses_first_frame=uses_first_frame,
        )
        fingerprint = sha256_text(
            {
                "static_identity": record["static_identity"],
                "prompt": prompt,
                "input_frame_sha256": input_hash,
                "uses_first_frame": uses_first_frame,
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
                previous_scene = scene_index
                previous_frame = Path(str(record["continuation_frame"]))
                LOGGER.info("[%d/%d] already complete: %s", index + 1, len(records), record["title"])
                continue
        except (OSError, ValueError, H3GenerationError):
            pass
        if record.get("fingerprint") != fingerprint:
            reset_record(record)
        record.update(
            fingerprint=fingerprint,
            uses_first_frame=uses_first_frame,
            input_frame=str(input_frame) if input_frame is not None else None,
            input_frame_sha256=input_hash,
        )
        save_checkpoint(checkpoint_path, checkpoint)

        while (
            str(record.get("job_id") or "").strip()
            or int(record.get("attempts") or 0) < max_attempts
        ):
            job_id = str(record.get("job_id") or "").strip()
            if not job_id:
                attempt = int(record.get("attempts") or 0) + 1
                mode = "I2VA" if uses_first_frame else "T2VA"
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
                record.update(
                    status="postprocessing",
                    output_file=str(output_path),
                    output_sha256=sha256_file(output_path),
                    duration_seconds=round(float(metadata["duration"]), 6),
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-audit-checkpoint", type=Path)
    parser.add_argument("--scene-segments", type=Path, required=True)
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://172.31.102.189:8189")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--minimum-duration", type=int, default=5)
    parser.add_argument("--maximum-duration", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--job-timeout", type=float, default=14400.0)
    parser.add_argument("--max-attempts", type=int, default=3)
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
    records = build_records(
        audited_plan,
        timeline,
        scenes,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        limit=args.limit,
    )
    checkpoint = prepare_checkpoint(
        args.checkpoint,
        records,
        endpoint=args.endpoint,
        width=args.width,
        height=args.height,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        scene_source_hash=sha256_file(args.scene_segments),
        resume=args.resume,
    )
    save_checkpoint(args.checkpoint, checkpoint)
    resets = sum(
        1
        for index, record in enumerate(records)
        if index == 0 or record["scene_index"] != records[index - 1]["scene_index"]
    )
    short = sum(record["usable_duration"] < record["expected_duration"] for record in records)
    LOGGER.info(
        "Prepared %d native H3 clips (%d T2VA scene resets, %d short beats trimmed)",
        len(records),
        resets,
        short,
    )
    if args.dry_run:
        return 0
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
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    LOGGER.info("All H3-native clips are complete: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
