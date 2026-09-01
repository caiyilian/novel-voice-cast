"""Generate resumable MiniMax H3 first/last-frame transition clips.

The H3 service is intentionally treated as a remote job queue: a submitted job
id is written to the checkpoint before polling begins, so Ctrl+C, a desktop stop
request, or a temporary LAN outage can be resumed without submitting duplicate
GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.subtitles import (  # noqa: E402
    build_segment_timestamps,
    discover_segment_files,
    load_subtitle_entries,
)
from scripts.generate_illustrations import apply_audited_prompts  # noqa: E402
from scripts.generate_video import (  # noqa: E402
    build_illustration_timeline,
    build_source_line_timeline,
    probe_media_duration,
    validate_audio_timeline,
)


LOGGER = logging.getLogger("h3_clips")
CHECKPOINT_VERSION = 1
DEFAULT_ENDPOINT = "http://172.31.102.189:8189"
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class H3GenerationError(RuntimeError):
    """A remote H3 job failed or returned an invalid artifact."""


class H3JobResumeRequired(H3GenerationError):
    """Stop locally while retaining a still-valid remote job id for resume."""


@dataclass(frozen=True)
class ClipSpec:
    index: int
    title: str
    first_frame: Path
    last_frame: Path
    output_path: Path
    prompt: str
    requested_duration: int | None
    expected_duration: float | None
    interval_duration: float
    fingerprint: str


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
    # On Windows, a read-only monitor, antivirus scanner, or indexer may briefly
    # open the destination without FILE_SHARE_DELETE.  os.replace then raises
    # WinError 5/32 even though both files are healthy.  Retry the atomic rename
    # instead of killing a multi-week generation run after a completed clip.
    deadline = time.monotonic() + 30.0
    delay = 0.05
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(1.0, delay * 2)


def sha256_text(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_title(value: object, fallback: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", str(value)).strip(" .")
    return cleaned[:60] or fallback


def h3_frame_count(duration_seconds: float, fps: int = 24) -> int:
    """Use the same 17k+5 frame rule as the deployed ComfyUI workflow."""

    frames = max(5, round(float(duration_seconds) * fps))
    return frames + (5 - frames % 17) % 17


def choose_duration(
    interval_seconds: float,
    *,
    minimum_seconds: int,
    maximum_seconds: int,
    fps: int = 24,
) -> tuple[int | None, float | None]:
    """Pick the longest integer H3 request whose actual frames fit the scene."""

    for requested in range(maximum_seconds, minimum_seconds - 1, -1):
        actual = h3_frame_count(requested, fps=fps) / fps
        if actual <= interval_seconds + 1e-6:
            return requested, actual
    return None, None


def build_transition_prompt(
    current: Mapping[str, Any],
    following: Mapping[str, Any],
    duration_seconds: int,
) -> str:
    current_prompt = str(current.get("prompt", "")).strip()
    following_prompt = str(following.get("prompt", "")).strip()
    current_title = str(current.get("title", "current story beat")).strip()
    following_title = str(following.get("title", "following story beat")).strip()
    return (
        "How the reference pictures align with the target video — Picture 1 "
        f"aligns with 0.00 seconds; Picture 2 aligns with {duration_seconds:.2f} seconds.\n\n"
        "integrated_multimodal_description: [Shot 1] High-quality 2D anime cinematic "
        "adaptation. Begin fully matched to Picture 1, preserving its exact character "
        "designs, faces, clothing, objects, palette, linework, lighting, camera position, "
        "and spatial layout. Use natural secondary motion, restrained character acting, "
        "stable anatomy, and a motivated camera move. Progress continuously toward the "
        "next visible story beat. If the two compositions differ substantially, use a "
        "natural foreground occlusion or camera movement instead of morphing bodies or "
        "objects. End fully matched to Picture 2, preserving its composition and all "
        f"identity details. Current beat ({current_title}): {current_prompt} "
        f"Following beat ({following_title}): {following_prompt}\n\n"
        "overall_soundscape: Natural scene ambience only. No spoken dialogue and no "
        "voiceover; the production audio will be supplied separately.\n\n"
        "non_diegetic_music: N/A\n\n"
        "Single coherent transition, no montage, no captions, no subtitles, no written "
        "text, no logos, no watermarks. Keep eyes, hands, teeth, faces, and character "
        "identity stable throughout."
    )


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("illustrations")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Illustration plan is missing or empty: {path}")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Every illustration plan item must be an object")
    return [dict(item) for item in payload]


def load_audited_plan(plan: Sequence[dict[str, Any]], audit_path: Path | None) -> list[dict[str, Any]]:
    if audit_path is None:
        return [dict(item) for item in plan]
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else None
    completed = payload.get("completed_indices") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != len(plan):
        raise ValueError("Visual prompt audit does not exactly cover the illustration plan")
    if completed != list(range(len(plan))):
        raise ValueError("Visual prompt audit checkpoint is incomplete")
    return apply_audited_prompts(plan, results)


def find_image(directory: Path, index: int) -> Path:
    matches = [path for path in directory.glob(f"{index + 1:04d}_*.png") if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one image for illustration {index + 1:04d} in {directory}; "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def build_timeline(
    plan: Sequence[dict[str, Any]],
    *,
    novel_path: Path,
    labels_path: Path,
    segments_dir: Path,
    audio_path: Path,
    ffprobe: str,
) -> list[dict[str, Any]]:
    segment_count = len(discover_segment_files(segments_dir))
    if segment_count <= 0:
        raise ValueError(f"No TTS WAV segments found: {segments_dir}")
    entries = load_subtitle_entries(
        novel_path,
        labels_path,
        expected_segment_count=segment_count,
    )
    timestamps = build_segment_timestamps(entries, segments_dir)
    total_duration_ms = int(timestamps[-1]["end_ms"])
    audio_duration = probe_media_duration(audio_path, ffprobe)
    validate_audio_timeline(audio_duration, total_duration_ms)
    return build_illustration_timeline(
        [dict(item) for item in plan],
        [int(item["start_ms"]) for item in timestamps],
        source_line_timeline=build_source_line_timeline(timestamps),
        total_duration_ms=total_duration_ms,
    )


def build_specs(
    plan: Sequence[dict[str, Any]],
    timeline: Sequence[dict[str, Any]],
    *,
    images_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    minimum_duration: int,
    maximum_duration: int,
    limit: int | None = None,
) -> list[ClipSpec]:
    if len(plan) != len(timeline):
        raise ValueError("Illustration plan and timeline counts do not match")
    image_hashes: dict[Path, str] = {}

    def image_identity(path: Path) -> dict[str, Any]:
        if path not in image_hashes:
            image_hashes[path] = sha256_file(path)
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "sha256": image_hashes[path]}

    count = max(0, len(plan) - 1)
    if limit is not None:
        count = min(count, max(0, limit))
    specs: list[ClipSpec] = []
    for index in range(count):
        first = find_image(images_dir, index)
        last = find_image(images_dir, index + 1)
        interval = int(timeline[index]["duration_ms"]) / 1000.0
        requested, expected = choose_duration(
            interval,
            minimum_seconds=minimum_duration,
            maximum_seconds=maximum_duration,
        )
        prompt = (
            build_transition_prompt(plan[index], plan[index + 1], requested)
            if requested is not None
            else ""
        )
        title = safe_title(plan[index].get("title"), f"clip_{index + 1:04d}")
        output = output_dir / f"{index + 1:04d}_to_{index + 2:04d}_{title}.mp4"
        fingerprint = sha256_text(
            {
                "index": index,
                "first": image_identity(first),
                "last": image_identity(last),
                "prompt": prompt,
                "requested_duration": requested,
                "expected_duration": expected,
                "interval_duration": round(interval, 6),
                "width": width,
                "height": height,
            }
        )
        specs.append(
            ClipSpec(
                index=index,
                title=title,
                first_frame=first,
                last_frame=last,
                output_path=output.resolve(),
                prompt=prompt,
                requested_duration=requested,
                expected_duration=expected,
                interval_duration=interval,
                fingerprint=fingerprint,
            )
        )
    return specs


def new_record(spec: ClipSpec) -> dict[str, Any]:
    skipped = spec.requested_duration is None
    return {
        "index": spec.index,
        "title": spec.title,
        "status": "skipped" if skipped else "pending",
        "reason": "scene interval is shorter than the minimum H3 clip" if skipped else None,
        "attempts": 0,
        "job_id": None,
        "submitted_at": None,
        "completed_at": None,
        "output_file": None,
        "output_sha256": None,
        "duration_seconds": None,
        "error_summary": None,
        "requested_duration": spec.requested_duration,
        "expected_duration": spec.expected_duration,
        "interval_duration": round(spec.interval_duration, 6),
        "fingerprint": spec.fingerprint,
    }


def prepare_checkpoint(
    path: Path,
    specs: Sequence[ClipSpec],
    *,
    endpoint: str,
    width: int,
    height: int,
    minimum_duration: int,
    maximum_duration: int,
    resume: bool,
) -> dict[str, Any]:
    source_hash = sha256_text([spec.fingerprint for spec in specs])
    fresh = {
        "version": CHECKPOINT_VERSION,
        "endpoint": endpoint.rstrip("/"),
        "width": width,
        "height": height,
        "minimum_duration": minimum_duration,
        "maximum_duration": maximum_duration,
        "source_hash": source_hash,
        "updated_at": utc_now(),
        "clips": [new_record(spec) for spec in specs],
    }
    if not resume or not path.is_file():
        return fresh
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring unreadable H3 checkpoint: %s", path)
        return fresh
    compatible = (
        isinstance(old, dict)
        and old.get("version") == CHECKPOINT_VERSION
        and old.get("endpoint") == endpoint.rstrip("/")
        and old.get("width") == width
        and old.get("height") == height
        and old.get("minimum_duration") == minimum_duration
        and old.get("maximum_duration") == maximum_duration
        and old.get("source_hash") == source_hash
        and isinstance(old.get("clips"), list)
        and len(old["clips"]) == len(specs)
    )
    if not compatible:
        LOGGER.warning("Ignoring incompatible H3 checkpoint: %s", path)
        return fresh
    for spec, record, old_record in zip(specs, fresh["clips"], old["clips"]):
        if not isinstance(old_record, dict) or old_record.get("fingerprint") != spec.fingerprint:
            continue
        record.update(old_record)
        record["index"] = spec.index
        record["fingerprint"] = spec.fingerprint
        if record.get("status") == "success":
            output = record.get("output_file")
            if not output or not Path(str(output)).is_file():
                record.update(
                    status="pending",
                    output_file=None,
                    output_sha256=None,
                    duration_seconds=None,
                    error_summary="Downloaded clip is missing; recovering",
                )
    return fresh


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(path, checkpoint)


class H3Client:
    def __init__(self, endpoint: str, *, timeout: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = max(1.0, float(timeout))
        self.session = requests.Session()
        self.session.trust_env = False

    def _url(self, suffix: str) -> str:
        return f"{self.endpoint}/{suffix.lstrip('/')}"

    @staticmethod
    def _json(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise H3GenerationError(
                f"H3 service returned non-JSON HTTP {response.status_code}"
            ) from exc
        if not isinstance(payload, dict):
            raise H3GenerationError("H3 service returned a non-object JSON response")
        if response.status_code >= 400:
            raise H3GenerationError(
                f"H3 service HTTP {response.status_code}: {payload.get('error', 'unknown error')}"
            )
        return payload

    def health(self) -> dict[str, Any]:
        response = self.session.get(self._url("api/health"), timeout=min(self.timeout, 15))
        payload = self._json(response)
        if payload.get("status") != "ok" or not payload.get("comfyui"):
            raise H3GenerationError(f"H3 service is not ready: {payload}")
        return payload

    def submit_request(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        duration: int,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
    ) -> str:
        values = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration": duration,
        }
        handles = []
        try:
            files: dict[str, tuple[str, Any, str]] = {}
            if first_frame is not None:
                first_handle = first_frame.open("rb")
                handles.append(first_handle)
                files["first_frame"] = (first_frame.name, first_handle, "image/png")
            if last_frame is not None:
                last_handle = last_frame.open("rb")
                handles.append(last_handle)
                files["last_frame"] = (last_frame.name, last_handle, "image/png")
            if files:
                response = self.session.post(
                    self._url("api/generate"),
                    data={key: str(value) for key, value in values.items()},
                    files=files,
                    timeout=self.timeout,
                )
            else:
                response = self.session.post(
                    self._url("api/generate"),
                    json=values,
                    timeout=self.timeout,
                )
        finally:
            for handle in handles:
                handle.close()
        payload = self._json(response)
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            raise H3GenerationError("H3 submission response is missing job_id")
        return job_id

    def submit(self, spec: ClipSpec, *, width: int, height: int) -> str:
        if spec.requested_duration is None:
            raise ValueError("Cannot submit a skipped clip")
        return self.submit_request(
            prompt=spec.prompt,
            width=width,
            height=height,
            duration=spec.requested_duration,
            first_frame=spec.first_frame,
            last_frame=spec.last_frame,
        )

    def status(self, job_id: str) -> dict[str, Any]:
        response = self.session.get(
            self._url(f"api/status/{job_id}"),
            timeout=min(self.timeout, 30),
        )
        return self._json(response)

    def download(self, job_id: str, output_path: Path) -> None:
        response = self.session.get(
            self._url(f"api/download/{job_id}"),
            timeout=max(self.timeout, 300),
            stream=True,
        )
        if response.status_code >= 400:
            self._json(response)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.download")
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise H3GenerationError("H3 download is empty")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise H3GenerationError(f"ffprobe rejected H3 clip: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise H3GenerationError("H3 clip has no video stream")
    return {
        "width": int(streams[0]["width"]),
        "height": int(streams[0]["height"]),
        "duration": float(payload["format"]["duration"]),
    }


def validate_downloaded_clip(
    path: Path,
    *,
    width: int,
    height: int,
    expected_duration: float,
    ffprobe: str,
) -> dict[str, Any]:
    metadata = probe_video(path, ffprobe)
    if metadata["width"] != width or metadata["height"] != height:
        raise H3GenerationError(
            "H3 clip dimensions do not match the request: "
            f"{metadata['width']}x{metadata['height']} != {width}x{height}"
        )
    if abs(metadata["duration"] - expected_duration) > 1.0:
        raise H3GenerationError(
            "H3 clip duration is outside tolerance: "
            f"{metadata['duration']:.3f}s != {expected_duration:.3f}s"
        )
    return metadata


def wait_for_job(
    client: H3Client,
    job_id: str,
    *,
    poll_seconds: float,
    job_timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    last_status = ""
    while True:
        if time.monotonic() - started > job_timeout:
            raise H3JobResumeRequired(
                f"H3 job {job_id} did not finish within {job_timeout / 3600:.1f} hours"
            )
        try:
            payload = client.status(job_id)
        except (requests.RequestException, H3GenerationError) as exc:
            if "HTTP 404" in str(exc):
                raise H3GenerationError(f"H3 job {job_id} is no longer known by the server") from exc
            # The timeout protects against a remote job that remains queued/running
            # for too long.  Time spent unable to reach the server is different: the
            # remote machine may simply be powered off, and expiring locally would
            # leave a resumable long-running pipeline stopped.  Give the job a fresh
            # timeout window after every transient connectivity failure.  Once the
            # server returns, a lost job is still detected via HTTP 404 and is safely
            # re-submitted by the checkpointed generation loop.
            started = time.monotonic()
            LOGGER.warning("H3 status request failed; keeping job id for resume: %s", exc)
            time.sleep(poll_seconds)
            continue
        status = str(payload.get("status", "")).lower()
        if status != last_status:
            LOGGER.info(
                "H3 job %s status=%s progress=%s",
                job_id,
                status or "unknown",
                payload.get("progress", "?"),
            )
            last_status = status
        if status == "completed":
            return payload
        if status == "failed":
            raise H3GenerationError(str(payload.get("error", "remote H3 job failed")))
        if status not in {"queued", "running"}:
            LOGGER.warning("H3 job %s returned unknown status %r", job_id, status)
        time.sleep(poll_seconds)


def download_completed_job(client: H3Client, job_id: str, output_path: Path) -> None:
    """Download without turning a local/network hiccup into an expensive re-generation."""

    try:
        client.download(job_id, output_path)
    except H3GenerationError as exc:
        if "HTTP 404" in str(exc):
            raise
        raise H3JobResumeRequired(
            f"H3 job {job_id} completed but its download is temporarily unavailable: {exc}"
        ) from exc
    except (OSError, requests.RequestException) as exc:
        raise H3JobResumeRequired(
            f"H3 job {job_id} completed but local download failed: {exc}"
        ) from exc


def run_generation(
    specs: Sequence[ClipSpec],
    *,
    checkpoint_path: Path,
    endpoint: str,
    width: int,
    height: int,
    minimum_duration: int,
    maximum_duration: int,
    request_timeout: float,
    poll_seconds: float,
    job_timeout: float,
    max_attempts: int,
    ffprobe: str,
    resume: bool,
) -> dict[str, Any]:
    checkpoint = prepare_checkpoint(
        checkpoint_path,
        specs,
        endpoint=endpoint,
        width=width,
        height=height,
        minimum_duration=minimum_duration,
        maximum_duration=maximum_duration,
        resume=resume,
    )
    save_checkpoint(checkpoint_path, checkpoint)
    client = H3Client(endpoint, timeout=request_timeout)
    client.health()
    total = len(specs)
    for spec, record in zip(specs, checkpoint["clips"]):
        if record.get("status") == "skipped":
            LOGGER.info(
                "[%d/%d] skipped; scene %.3fs is shorter than a valid H3 clip",
                spec.index + 1,
                total,
                spec.interval_duration,
            )
            continue
        if record.get("status") == "success":
            try:
                validate_downloaded_clip(
                    Path(str(record["output_file"])),
                    width=width,
                    height=height,
                    expected_duration=float(spec.expected_duration),
                    ffprobe=ffprobe,
                )
                LOGGER.info("[%d/%d] already complete: %s", spec.index + 1, total, spec.title)
                continue
            except (OSError, ValueError, H3GenerationError) as exc:
                record.update(
                    status="pending",
                    output_file=None,
                    output_sha256=None,
                    duration_seconds=None,
                    error_summary=f"Cached clip is invalid: {exc}",
                )
                save_checkpoint(checkpoint_path, checkpoint)

        while (
            str(record.get("job_id") or "").strip()
            or int(record.get("attempts") or 0) < max_attempts
        ):
            job_id = str(record.get("job_id") or "").strip()
            if not job_id:
                attempt = int(record.get("attempts") or 0) + 1
                LOGGER.info(
                    "[%d/%d] submitting %s (%ss, attempt %d/%d)",
                    spec.index + 1,
                    total,
                    spec.title,
                    spec.requested_duration,
                    attempt,
                    max_attempts,
                )
                try:
                    job_id = client.submit(spec, width=width, height=height)
                except (requests.RequestException, H3GenerationError) as exc:
                    record.update(
                        attempts=attempt,
                        status="pending",
                        error_summary=str(exc),
                    )
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
                    completed_at=None,
                    error_summary=None,
                )
                save_checkpoint(checkpoint_path, checkpoint)

            try:
                record["status"] = "running"
                save_checkpoint(checkpoint_path, checkpoint)
                wait_for_job(
                    client,
                    job_id,
                    poll_seconds=poll_seconds,
                    job_timeout=job_timeout,
                )
                download_completed_job(client, job_id, spec.output_path)
                try:
                    metadata = validate_downloaded_clip(
                        spec.output_path,
                        width=width,
                        height=height,
                        expected_duration=float(spec.expected_duration),
                        ffprobe=ffprobe,
                    )
                except OSError as exc:
                    raise H3JobResumeRequired(
                        f"H3 clip was downloaded but local validation could not run: {exc}"
                    ) from exc
                record.update(
                    status="success",
                    job_id=job_id,
                    completed_at=utc_now(),
                    output_file=str(spec.output_path),
                    output_sha256=sha256_file(spec.output_path),
                    duration_seconds=round(float(metadata["duration"]), 6),
                    error_summary=None,
                )
                save_checkpoint(checkpoint_path, checkpoint)
                LOGGER.info(
                    "[%d/%d] completed: %s (%.3fs)",
                    spec.index + 1,
                    total,
                    spec.output_path,
                    metadata["duration"],
                )
                break
            except KeyboardInterrupt:
                save_checkpoint(checkpoint_path, checkpoint)
                raise
            except H3JobResumeRequired as exc:
                record.update(status="queued", error_summary=str(exc))
                save_checkpoint(checkpoint_path, checkpoint)
                raise
            except (OSError, ValueError, requests.RequestException, H3GenerationError) as exc:
                LOGGER.error("[%d/%d] H3 attempt failed: %s", spec.index + 1, total, exc)
                record.update(
                    status="pending",
                    job_id=None,
                    completed_at=utc_now(),
                    output_file=None,
                    output_sha256=None,
                    duration_seconds=None,
                    error_summary=str(exc),
                )
                save_checkpoint(checkpoint_path, checkpoint)
                if int(record.get("attempts") or 0) < max_attempts:
                    time.sleep(min(60.0, poll_seconds * int(record["attempts"])))

        if record.get("status") != "success":
            raise H3GenerationError(
                f"H3 transition {spec.index + 1}/{total} failed after "
                f"{record.get('attempts', 0)} attempts: {record.get('error_summary')}"
            )
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate resumable MiniMax H3 transition clips")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-audit-checkpoint", type=Path)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--minimum-duration", type=int, default=5)
    parser.add_argument("--maximum-duration", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--job-timeout", type=float, default=14400.0)
    parser.add_argument("--max-attempts", type=int, default=3)
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
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")
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
    specs = build_specs(
        audited_plan,
        timeline,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        limit=args.limit,
    )
    skipped = sum(spec.requested_duration is None for spec in specs)
    LOGGER.info(
        "Prepared %d H3 transitions (%d generation jobs, %d short intervals skipped)",
        len(specs),
        len(specs) - skipped,
        skipped,
    )
    if args.dry_run:
        return 0
    run_generation(
        specs,
        checkpoint_path=args.checkpoint,
        endpoint=args.endpoint,
        width=args.width,
        height=args.height,
        minimum_duration=args.minimum_duration,
        maximum_duration=args.maximum_duration,
        request_timeout=args.request_timeout,
        poll_seconds=args.poll_seconds,
        job_timeout=args.job_timeout,
        max_attempts=args.max_attempts,
        ffprobe=args.ffprobe,
        resume=args.resume,
    )
    LOGGER.info("All H3 transition clips are complete: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
