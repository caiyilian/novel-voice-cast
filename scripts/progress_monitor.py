"""Serve a read-only web dashboard for the long-running full pipeline.

The monitor deliberately does not communicate with the pipeline process.  It
only reads the atomically-written manifest/checkpoints and output directory, so
it can be started or stopped without affecting resume behaviour.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STAGES = (
    "parse",
    "gender",
    "emotion",
    "performance",
    "tts",
    "splice",
    "bgm-segment",
    "bgm-label",
    "bgm-generate",
    "bgm-mix",
    "illustration-plan",
    "illustrations",
    "video",
)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def _resolve(root: Path, value: object, default: str) -> Path:
    candidate = Path(str(value or default)).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _modified_at(path: Path) -> str | None:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _size_info(value: object) -> dict[str, Any]:
    text = str(value or "")
    try:
        width_text, height_text = text.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"raw": text, "width": None, "height": None, "aspect_ratio": "unknown"}
    divisor = math.gcd(width, height)
    return {
        "raw": text,
        "width": width,
        "height": height,
        "aspect_ratio": f"{width // divisor}:{height // divisor}",
        "orientation": "portrait" if height > width else "landscape" if width > height else "square",
    }


def _tail_jsonl(path: Path, limit: int = 120) -> list[dict[str, Any]]:
    """Read a small JSONL tail without loading an indefinitely growing log."""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            chunks: deque[bytes] = deque()
            newline_count = 0
            while position > 0 and newline_count <= limit:
                amount = min(65536, position)
                position -= amount
                handle.seek(position)
                chunk = handle.read(amount)
                chunks.appendleft(chunk)
                newline_count += chunk.count(b"\n")
        lines = b"".join(chunks).decode("utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


class ProcessProbe:
    """Cache the relatively expensive operating-system process lookup."""

    def __init__(self, ttl_seconds: float = 8.0):
        self.ttl_seconds = ttl_seconds
        self._checked_at = 0.0
        self._value: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            if now - self._checked_at < self.ttl_seconds:
                return list(self._value)
            self._checked_at = now
            self._value = self._query()
            return list(self._value)

    @staticmethod
    def _query() -> list[dict[str, Any]]:
        if os.name == "nt":
            command = (
                "$ErrorActionPreference='SilentlyContinue';"
                "Get-CimInstance Win32_Process | "
                "Where-Object {$_.Name -match 'python' -and "
                "($_.CommandLine -match 'run_full.py|generate_illustrations.py|generate_h3_')} | "
                "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command],
                    check=False,
                    capture_output=True,
                    timeout=8,
                    creationflags=creationflags,
                )
                text = result.stdout.decode(errors="replace").strip()
                raw = json.loads(text) if text else []
                values = raw if isinstance(raw, list) else [raw]
                return [
                    {
                        "pid": item.get("ProcessId"),
                        "started_at": item.get("CreationDate"),
                        "command": item.get("CommandLine", ""),
                    }
                    for item in values
                    if isinstance(item, dict)
                ]
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
                return []

        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,lstart=,args="],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        values = []
        for line in result.stdout.splitlines():
            if not any(
                name in line
                for name in ("run_full.py", "generate_illustrations.py", "generate_h3_")
            ):
                continue
            parts = line.strip().split(maxsplit=1)
            values.append({"pid": int(parts[0]), "started_at": None, "command": parts[-1]})
        return values


class RateSampler:
    """Estimate throughput from checkpoint changes observed while serving."""

    def __init__(self):
        self._samples: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()

    def sample(self, key: str, completed: int, total: int) -> dict[str, float | None]:
        now = time.monotonic()
        with self._lock:
            history = self._samples.setdefault(key, deque(maxlen=240))
            if not history or history[-1][1] != completed or now - history[-1][0] >= 60:
                history.append((now, completed))
            useful = [item for item in history if item[1] != completed]
            if not useful:
                return {"items_per_hour": None, "eta_seconds": None}
            started_at, started_count = useful[0]
            elapsed = now - started_at
            delta = completed - started_count
            if elapsed <= 0 or delta <= 0:
                return {"items_per_hour": None, "eta_seconds": None}
            per_second = delta / elapsed
            return {
                "items_per_hour": round(per_second * 3600, 2),
                "eta_seconds": round(max(0, total - completed) / per_second, 1),
            }


class ProgressCollector:
    def __init__(
        self,
        root: Path = ROOT,
        config_path: Path = DEFAULT_CONFIG,
        *,
        process_probe: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.root = root.resolve()
        self.config_path = _resolve(self.root, config_path, str(DEFAULT_CONFIG))
        self.process_probe = process_probe or ProcessProbe()
        self.rates = RateSampler()

    def _config(self) -> dict[str, Any]:
        try:
            value = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            return {}
        return value if isinstance(value, dict) else {}

    def collect(self) -> dict[str, Any]:
        config = self._config()
        illustrations = config.get("illustrations", {})
        video_config = config.get("video", {})
        if not isinstance(illustrations, dict):
            illustrations = {}
        if not isinstance(video_config, dict):
            video_config = {}
        h3_config = video_config.get("h3", {})
        if not isinstance(h3_config, dict):
            h3_config = {}
        h3_enabled = bool(h3_config.get("enabled", False))
        h3_mode = str(h3_config.get("mode", "native-chain"))
        h3_root = _resolve(self.root, h3_config.get("output_dir"), "output/h3_video")

        manifest_path = self.root / "output/run_full_manifest.json"
        audit_path = _resolve(
            self.root,
            illustrations.get("prompt_audit_checkpoint_path"),
            "backend/data/visual_prompt_audit.checkpoint.json",
        )
        image_configs = [
            {
                "name": "portrait",
                "size": illustrations.get("size", "896x1152"),
                "checkpoint": _resolve(
                    self.root,
                    illustrations.get("checkpoint_path"),
                    "output/illustrations_checkpoint.json",
                ),
                "directory": _resolve(
                    self.root,
                    illustrations.get("output_dir"),
                    "output/illustrations",
                ),
            }
        ]
        landscape_images = illustrations.get("landscape", {})
        if isinstance(landscape_images, dict) and landscape_images.get("enabled", False):
            image_configs.append(
                {
                    "name": "landscape",
                    "size": landscape_images.get("size", "1280x720"),
                    "checkpoint": _resolve(
                        self.root,
                        landscape_images.get("checkpoint_path"),
                        "output/illustrations_local_landscape_16x9_checkpoint.json",
                    ),
                    "directory": _resolve(
                        self.root,
                        landscape_images.get("output_dir"),
                        "output/illustrations_local_landscape_16x9",
                    ),
                }
            )
        llm_log_path = self.root / "logs/illustration_prompt_audit_llm_calls.jsonl"

        manifest = _read_json(manifest_path, {})
        audit = self._audit_status(audit_path)
        image_variants = []
        for item in image_configs:
            status = self._image_status(
                item["checkpoint"],
                item["directory"],
                audit["total"],
                rate_key=f"images-{item['name']}",
            )
            status.update(name=item["name"], **_size_info(item["size"]))
            image_variants.append(status)
        image_generation = image_variants[0]
        llm = self._llm_status(llm_log_path)
        processes = self.process_probe()
        pipeline_running = bool(processes)
        video_configs = [
            {
                "name": "portrait",
                "path": _resolve(
                    self.root,
                    video_config.get("output_path"),
                    "output/illustration_video_agnes_subtitled.mp4",
                ),
                "subtitle": _resolve(
                    self.root,
                    video_config.get("subtitle_path"),
                    "output/illustration_video_agnes_subtitles.srt",
                ),
            }
        ]
        landscape_video = video_config.get("landscape", {})
        if (
            len(image_variants) > 1
            and isinstance(landscape_video, dict)
            and landscape_video.get("enabled", True)
        ):
            video_configs.append(
                {
                    "name": "landscape",
                    "path": _resolve(
                        self.root,
                        landscape_video.get("output_path"),
                        "output/illustration_video_local_landscape_16x9_subtitled.mp4",
                    ),
                    "subtitle": _resolve(
                        self.root,
                        landscape_video.get("subtitle_path"),
                        "output/illustration_video_local_landscape_16x9_subtitles.srt",
                    ),
                }
            )
        videos = []
        for item in video_configs:
            path, subtitle = item["path"], item["subtitle"]
            videos.append(
                {
                    "name": item["name"],
                    "exists": path.is_file() and path.stat().st_size > 0,
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                    "updated_at": _modified_at(path),
                    "subtitle_exists": subtitle.is_file() and subtitle.stat().st_size > 0,
                    "subtitle_path": str(subtitle),
                }
            )
        video = videos[0]

        h3_variants = []
        if h3_enabled:
            for item in video_configs:
                variant_root = h3_root / item["name"]
                h3_variants.append(
                    self._h3_status(
                        variant_root / "h3_clips.checkpoint.json",
                        variant_root / "h3_render.checkpoint.json",
                        fallback_total=audit["total"],
                        output_exists=next(
                            video_item["exists"]
                            for video_item in videos
                            if video_item["name"] == item["name"]
                        ),
                        name=item["name"],
                        mode=h3_mode,
                    )
                )

        phase = self._phase(
            pipeline_running,
            audit,
            image_variants,
            videos,
            h3_variants,
            manifest,
        )
        stages = self._stages(manifest, phase, pipeline_running)
        activity_paths = (
            manifest_path,
            audit_path,
            llm_log_path,
            *[item["checkpoint"] for item in image_configs],
            *[
                Path(path)
                for item in h3_variants
                for path in (item["checkpoint_path"], item["render_checkpoint_path"])
            ],
            *[item["path"] for item in video_configs],
        )
        ages = [value for value in (_age_seconds(path) for path in activity_paths) if value is not None]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_running": pipeline_running,
            "processes": processes,
            "phase": phase,
            "last_activity_seconds": min(ages) if ages else None,
            "manifest": {
                "path": str(manifest_path),
                "run_status": manifest.get("run_status", "unknown") if isinstance(manifest, dict) else "unknown",
                "run_started_at": manifest.get("run_started_at") if isinstance(manifest, dict) else None,
                "updated_at": manifest.get("updated_at") if isinstance(manifest, dict) else None,
            },
            "stages": stages,
            "audit": audit,
            "image_generation": image_generation,
            "image_variants": image_variants,
            "video": video,
            "videos": videos,
            "h3_enabled": h3_enabled,
            "h3_mode": h3_mode if h3_enabled else None,
            "h3_variants": h3_variants,
            "llm": llm,
            "image": {
                **_size_info(illustrations.get("size", "896x1152")),
                "steps": illustrations.get("steps"),
                "cfg": illustrations.get("cfg"),
                "provider": illustrations.get("provider"),
                "endpoint": illustrations.get("endpoint"),
            },
            "image_sizes": [
                {"name": item["name"], **_size_info(item["size"])} for item in image_configs
            ],
        }

    def _audit_status(self, path: Path) -> dict[str, Any]:
        raw = _read_json(path, {})
        total = int(raw.get("total_items") or 0) if isinstance(raw, dict) else 0
        completed_values = raw.get("completed_indices", []) if isinstance(raw, dict) else []
        completed_set = {
            int(value)
            for value in completed_values
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        }
        completed = len(completed_set)
        next_index = next((index + 1 for index in range(total) if index not in completed_set), None)
        errors = raw.get("errors", {}) if isinstance(raw, dict) else {}
        error_count = len(errors) if isinstance(errors, (dict, list)) else 0
        progress = round(completed * 100 / total, 2) if total else 0.0
        return {
            "exists": path.is_file(),
            "path": str(path),
            "model": raw.get("model") if isinstance(raw, dict) else None,
            "completed": completed,
            "total": total,
            "pending": max(0, total - completed),
            "percent": progress,
            "next_index": next_index,
            "error_count": error_count,
            "updated_at": _modified_at(path),
            "activity_age_seconds": _age_seconds(path),
            **self.rates.sample("audit", completed, total),
        }

    def _image_status(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        fallback_total: int,
        *,
        rate_key: str = "images",
    ) -> dict[str, Any]:
        raw = _read_json(checkpoint_path, {})
        images = raw.get("images", []) if isinstance(raw, dict) else []
        if not isinstance(images, list):
            images = []
        counts = Counter(
            str(item.get("status", "pending"))
            for item in images
            if isinstance(item, dict)
        )
        total = len(images) or fallback_total
        success = counts["success"]
        running = next(
            (item for item in images if isinstance(item, dict) and item.get("status") == "running"),
            None,
        )
        latest = None
        successful = [
            item
            for item in images
            if isinstance(item, dict) and item.get("status") == "success" and item.get("output_file")
        ]
        if successful:
            latest = successful[-1]
        try:
            output_count = sum(1 for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
        except OSError:
            output_count = 0
        return {
            "checkpoint_exists": checkpoint_path.is_file(),
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "total": total,
            "success": success,
            "failed": counts["failed"],
            "running": counts["running"],
            "pending": max(0, total - success - counts["failed"] - counts["running"]),
            "percent": round(success * 100 / total, 2) if total else 0.0,
            "attempts": sum(int(item.get("attempts") or 0) for item in images if isinstance(item, dict)),
            "output_count": output_count,
            "current": self._public_image_record(running),
            "latest": self._public_image_record(latest),
            "updated_at": _modified_at(checkpoint_path),
            "activity_age_seconds": _age_seconds(checkpoint_path),
            **self.rates.sample(rate_key, success, total),
        }

    def _h3_status(
        self,
        checkpoint_path: Path,
        render_checkpoint_path: Path,
        *,
        fallback_total: int,
        output_exists: bool,
        name: str,
        mode: str,
    ) -> dict[str, Any]:
        raw = _read_json(checkpoint_path, {})
        clips = raw.get("clips", []) if isinstance(raw, dict) else []
        if not isinstance(clips, list):
            clips = []
        render_raw = _read_json(render_checkpoint_path, {})
        segments = render_raw.get("segments", []) if isinstance(render_raw, dict) else []
        if not isinstance(segments, list):
            segments = []
        expected_clips = fallback_total if mode == "native-chain" else max(0, fallback_total - 1)
        clip_total = len(clips) or expected_clips
        render_total = len(segments) or fallback_total
        clip_counts = Counter(
            str(item.get("status", "pending"))
            for item in clips
            if isinstance(item, dict)
        )
        render_counts = Counter(
            str(item.get("status", "pending"))
            for item in segments
            if isinstance(item, dict)
        )
        active_statuses = {"queued", "running", "postprocessing", "postprocess_failed"}
        current = next(
            (
                item
                for item in clips
                if isinstance(item, dict) and str(item.get("status")) in active_statuses
            ),
            None,
        )
        clip_success = clip_counts["success"] + clip_counts["skipped"]
        render_success = render_counts["success"] + render_counts["skipped"]
        completed = clip_success + render_success + int(output_exists)
        total = clip_total + render_total + 1
        return {
            "name": name,
            "mode": mode,
            "checkpoint_exists": checkpoint_path.is_file(),
            "checkpoint_path": str(checkpoint_path),
            "render_checkpoint_path": str(render_checkpoint_path),
            "clip_total": clip_total,
            "clip_success": clip_success,
            "clip_running": sum(clip_counts[status] for status in active_statuses),
            "clip_failed": clip_counts["failed"],
            "render_total": render_total,
            "render_success": render_success,
            "output_exists": output_exists,
            "completed": completed,
            "total": total,
            "percent": round(completed * 100 / total, 2) if total else 0.0,
            "current": (
                {
                    key: current.get(key)
                    for key in (
                        "index",
                        "title",
                        "status",
                        "attempts",
                        "job_id",
                        "scene_title",
                        "requested_duration",
                    )
                }
                if current
                else None
            ),
            "updated_at": _modified_at(checkpoint_path),
            "activity_age_seconds": _age_seconds(checkpoint_path),
            **self.rates.sample(f"h3-{name}", clip_success, clip_total),
        }

    @staticmethod
    def _public_image_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not record:
            return None
        return {
            key: record.get(key)
            for key in ("index", "title", "status", "attempts", "started_at", "ended_at", "duration_seconds")
        }

    @staticmethod
    def _llm_status(path: Path) -> dict[str, Any]:
        records = _tail_jsonl(path)
        latest = records[-1] if records else None
        account_counts = Counter(
            str(record.get("account"))
            for record in records
            if record.get("account") is not None
        )
        recent = []
        for record in records[-12:]:
            recent.append(
                {
                    key: record.get(key)
                    for key in (
                        "timestamp",
                        "agent_role",
                        "model",
                        "account",
                        "status",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "elapsed_seconds",
                    )
                }
            )
        return {
            "path": str(path),
            "updated_at": _modified_at(path),
            "activity_age_seconds": _age_seconds(path),
            "latest": recent[-1] if latest else None,
            "recent": recent,
            "recent_account_counts": dict(sorted(account_counts.items())),
        }

    @staticmethod
    def _phase(
        running: bool,
        audit: Mapping[str, Any],
        images: Sequence[Mapping[str, Any]],
        videos: Sequence[Mapping[str, Any]],
        h3_variants: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
    ) -> dict[str, str]:
        if running:
            if audit.get("total") and audit.get("completed", 0) < audit.get("total", 0):
                return {"code": "prompt-audit", "label": "插图提示词审核"}
            video_stage = manifest.get("stages", {}).get("video", {}) if isinstance(manifest, Mapping) else {}
            video_stage_running = (
                isinstance(video_stage, Mapping) and video_stage.get("status") == "running"
            )
            if h3_variants and (
                video_stage_running or any(item.get("checkpoint_exists") for item in h3_variants)
            ):
                if any(
                    item.get("clip_success", 0) < item.get("clip_total", 0)
                    for item in h3_variants
                ):
                    return {"code": "h3-generation", "label": "MiniMax H3 动态镜头生成"}
                if any(
                    item.get("render_success", 0) < item.get("render_total", 0)
                    for item in h3_variants
                ):
                    return {"code": "h3-render", "label": "H3 镜头时间轴合成"}
            if any(
                image.get("total") and image.get("success", 0) < image.get("total", 0)
                for image in images
            ):
                return {"code": "image-generation", "label": "本地文生图"}
            if any(not video.get("exists") for video in videos):
                return {"code": "video", "label": "字幕视频合成"}
            return {"code": "finishing", "label": "流程收尾"}
        status = str(manifest.get("run_status", "unknown")) if isinstance(manifest, Mapping) else "unknown"
        return {
            "code": status,
            "label": {
                "complete": "已完成",
                "failed": "已失败",
                "interrupted": "已中断，可断点续跑",
            }.get(status, "未运行"),
        }

    @staticmethod
    def _stages(
        manifest: Mapping[str, Any],
        phase: Mapping[str, str],
        pipeline_running: bool,
    ) -> list[dict[str, Any]]:
        raw_stages = manifest.get("stages", {}) if isinstance(manifest, Mapping) else {}
        raw_stages = raw_stages if isinstance(raw_stages, dict) else {}
        values = []
        for stage in STAGES:
            entry = raw_stages.get(stage, {})
            entry = entry if isinstance(entry, dict) else {}
            status = entry.get("status", "pending")
            if pipeline_running and stage == "illustrations" and phase.get("code") in {
                "prompt-audit",
                "image-generation",
            }:
                status = "running"
            if pipeline_running and stage == "video" and phase.get("code") in {
                "video",
                "h3-generation",
                "h3-render",
            }:
                status = "running"
            values.append(
                {
                    "name": stage,
                    "status": status,
                    "elapsed_seconds": entry.get("elapsed_seconds"),
                    "finished_at": entry.get("finished_at"),
                    "error": entry.get("error") if status == "failed" else None,
                }
            )
        return values


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Novel Voice Cast · 进度监控</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e; --blue:#58a6ff; --green:#3fb950; --yellow:#d29922; --red:#f85149; }
    * { box-sizing:border-box } body { margin:0; background:radial-gradient(circle at 15% 0,#17233a 0,transparent 35%),var(--bg); color:var(--text); font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif }
    main { width:min(1180px,calc(100% - 32px)); margin:30px auto 60px }
    header { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin-bottom:20px }
    h1 { font-size:26px; margin:0 0 5px } h2 { font-size:17px; margin:0 0 14px } .muted { color:var(--muted) }
    .live { display:flex; align-items:center; gap:8px; white-space:nowrap; padding:7px 12px; border:1px solid var(--line); border-radius:999px; background:#11161d }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--red) } .dot.on { background:var(--green); box-shadow:0 0 12px var(--green) }
    .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px }.card { grid-column:span 4; background:color-mix(in srgb,var(--panel) 94%,transparent); border:1px solid var(--line); border-radius:12px; padding:17px; min-width:0 }
    .wide { grid-column:span 8 }.full { grid-column:1/-1 }.value { font-size:28px; font-weight:700; letter-spacing:-.5px }.sub { color:var(--muted); margin-top:3px; overflow-wrap:anywhere }
    .bar { height:9px; background:#252b33; border-radius:99px; overflow:hidden; margin:13px 0 7px }.fill { height:100%; background:linear-gradient(90deg,#1f6feb,var(--blue)); border-radius:inherit; transition:width .35s }
    .pair { display:flex; justify-content:space-between; gap:15px }.pill { display:inline-block; padding:2px 8px; border-radius:99px; background:#243044; color:#b8d8ff; margin:2px }
    .stages { display:grid; grid-template-columns:repeat(13,minmax(52px,1fr)); gap:6px }.stage { text-align:center; padding:9px 4px; border-radius:8px; background:#22272e; color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis }.stage.complete { color:#b6f0c2; background:#15321e }.stage.running { color:#cae5ff; background:#153557 }.stage.failed { color:#ffc2bd; background:#3c1c1c }
    table { width:100%; border-collapse:collapse; font-size:13px } th,td { padding:7px 8px; border-bottom:1px solid #262c34; text-align:left } th { color:var(--muted); font-weight:500 } tbody tr:last-child td { border-bottom:0 }
    code { color:#b8d8ff } .error { color:#ffb4ad }.nowrap { white-space:nowrap }
    @media(max-width:850px){.card,.wide{grid-column:1/-1}.stages{grid-template-columns:repeat(3,1fr)}header{display:block}.live{margin-top:12px;width:max-content}}
  </style>
</head>
<body><main>
  <header><div><h1>Novel Voice Cast</h1><div class="muted" id="phase">正在读取进度…</div></div><div class="live"><span class="dot" id="dot"></span><span id="liveText">检测中</span></div></header>
  <section class="grid">
    <article class="card"><h2>提示词审核</h2><div class="value" id="auditValue">—</div><div class="bar"><div class="fill" id="auditBar"></div></div><div class="pair muted"><span id="auditNext">—</span><span id="auditEta">—</span></div></article>
    <article class="card"><h2>本地文生图</h2><div class="value" id="imageValue">—</div><div class="bar"><div class="fill" id="imageBar"></div></div><div class="pair muted"><span id="imageState">—</span><span id="imageEta">—</span></div></article>
    <article class="card"><h2>MiniMax H3 镜头</h2><div class="value" id="h3Value">—</div><div class="bar"><div class="fill" id="h3Bar"></div></div><div class="pair muted"><span id="h3State">—</span><span id="h3Eta">—</span></div></article>
    <article class="card"><h2>视频输出</h2><div class="value" id="videoValue">—</div><div class="sub" id="videoPath">—</div></article>
    <article class="card"><h2>画面规格</h2><div class="value" id="ratioValue">—</div><div class="sub" id="sizeValue">—</div></article>
    <article class="card wide"><h2>SenseNova 最近状态</h2><div id="llmSummary" class="sub">—</div><div id="accounts" style="margin-top:10px"></div></article>
    <article class="card full"><h2>完整流水线</h2><div class="stages" id="stages"></div></article>
    <article class="card full"><h2>最近模型调用</h2><div style="overflow:auto"><table><thead><tr><th>时间</th><th>角色</th><th>账号</th><th>状态</th><th>Token</th><th>耗时</th></tr></thead><tbody id="calls"></tbody></table></div></article>
  </section>
</main>
<script>
const $=id=>document.getElementById(id); const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const duration=s=>{if(s==null)return '等待测速';s=Math.max(0,Number(s));if(s<60)return Math.round(s)+'秒';if(s<3600)return Math.round(s/60)+'分钟';return (s/3600).toFixed(1)+'小时'};
const localTime=s=>s?new Date(s).toLocaleTimeString('zh-CN',{hour12:false}):'—';
function progress(prefix,v){$(prefix+'Value').textContent=`${v.completed??v.success}/${v.total} · ${Number(v.percent||0).toFixed(1)}%`;$(prefix+'Bar').style.width=Math.min(100,Number(v.percent||0))+'%'}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);const d=await r.json();
  $('dot').className='dot '+(d.pipeline_running?'on':'');$('liveText').textContent=d.pipeline_running?`运行中 · ${d.processes.length} 个关联进程`:'未检测到流水线进程';$('phase').textContent=`当前阶段：${d.phase.label} · 最近活动 ${duration(d.last_activity_seconds)}前`;
  progress('audit',d.audit);$('auditNext').textContent=d.audit.next_index?`下一项 #${d.audit.next_index}`:'审核完成';$('auditEta').textContent=d.audit.eta_seconds==null?'页面观测到新进度后显示 ETA':`预计剩余 ${duration(d.audit.eta_seconds)}`;
  const iv=d.image_variants||[d.image_generation], done=iv.reduce((n,x)=>n+x.success,0), total=iv.reduce((n,x)=>n+x.total,0), pct=total?done*100/total:0;
  $('imageValue').textContent=iv.map(x=>`${x.name==='portrait'?'竖':'横'} ${x.success}/${x.total}`).join(' · ');$('imageBar').style.width=Math.min(100,pct)+'%';$('imageState').textContent=iv.some(x=>x.checkpoint_exists)?iv.map(x=>`${x.name==='portrait'?'竖':'横'}失败 ${x.failed}`).join(' / '):'等待首条审核结果';const imageEta=iv.map(x=>x.eta_seconds).filter(x=>x!=null);$('imageEta').textContent=imageEta.length?`较慢队列剩余 ${duration(Math.max(...imageEta))}`:'—';
  const hv=d.h3_variants||[], hdone=hv.reduce((n,x)=>n+x.completed,0), htotal=hv.reduce((n,x)=>n+x.total,0), hpct=htotal?hdone*100/htotal:0;
  $('h3Value').textContent=hv.length?hv.map(x=>`${x.name==='portrait'?'竖':'横'} 片段 ${x.clip_success}/${x.clip_total}`).join(' · '):'未启用';$('h3Bar').style.width=Math.min(100,hpct)+'%';$('h3State').textContent=hv.length?hv.map(x=>{const c=x.current;return `${x.name==='portrait'?'竖':'横'} ${c?`#${Number(c.index)+1} ${c.status}`:`编码 ${x.render_success}/${x.render_total}`}`}).join(' / '):'静态插图模式';const h3Eta=hv.map(x=>x.eta_seconds).filter(x=>x!=null);$('h3Eta').textContent=h3Eta.length?`较慢队列剩余 ${duration(Math.max(...h3Eta))}`:'等待生成两条后估算';
  const vv=d.videos||[d.video];$('videoValue').textContent=vv.map(x=>`${x.name==='portrait'?'竖':'横'}${x.exists?'已生成':'待生成'}`).join(' · ');$('videoPath').innerHTML=vv.map(x=>`<code>${esc(x.path)}</code>`).join('<br>');
  $('ratioValue').textContent=(d.image_sizes||[d.image]).map(x=>x.aspect_ratio).join(' + ');$('sizeValue').textContent=`${(d.image_sizes||[d.image]).map(x=>x.raw).join(' / ')} · steps ${d.image.steps} · CFG ${d.image.cfg}`;
  const latest=d.llm.latest;$('llmSummary').textContent=latest?`${latest.model} · account=${latest.account} · ${latest.status} · ${Number(latest.elapsed_seconds||0).toFixed(1)} 秒`:'尚无记录';
  $('accounts').innerHTML=Object.entries(d.llm.recent_account_counts).map(([a,n])=>`<span class="pill">账号 ${esc(a)}：${n} 次</span>`).join('');
  $('stages').innerHTML=d.stages.map(s=>`<div class="stage ${esc(s.status)}" title="${esc(s.name)} · ${esc(s.status)}">${esc(s.name)}</div>`).join('');
  $('calls').innerHTML=[...d.llm.recent].reverse().map(x=>`<tr><td class="nowrap">${esc(localTime(x.timestamp))}</td><td>${esc(x.agent_role)}</td><td>${esc(x.account)}</td><td>${esc(x.status)}</td><td>${esc(x.total_tokens)}</td><td>${Number(x.elapsed_seconds||0).toFixed(1)}s</td></tr>`).join('')||'<tr><td colspan="6" class="muted">暂无模型调用</td></tr>';
}catch(e){$('liveText').textContent='监控读取失败';$('phase').textContent=String(e)}}refresh();setInterval(refresh,3000);
</script></body></html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    collector: ProgressCollector

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            payload = json.dumps(self.collector.collect(), ensure_ascii=False).encode("utf-8")
            self._send(payload, "application/json; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[monitor] {self.address_string()} {fmt % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动小说流水线只读进度监控网页")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    collector = ProgressCollector(ROOT, args.config)
    MonitorHandler.collector = collector
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    address, port = server.server_address[:2]
    print(f"进度监控已启动：http://{address}:{port}")
    print("这是只读监控；按 Ctrl+C 只会关闭网页，不会停止主流水线。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n进度监控已关闭。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
