"""Run the complete novel voice, BGM, illustration, and video pipeline.

The pipeline is deliberately stage based. Every expensive stage validates its
inputs and outputs, supports resume where the underlying implementation allows
it, and records timing/artifacts in ``output/run_full_manifest.json``.
"""
from __future__ import annotations

import _thread
import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from app.core.bgm_mixer import mix_bgm
from app.core.bgm_generator import (
    ACE_STEP_MODEL,
    ACE_STEP_LM_MODEL,
    BGM_GENERATION_VERSION,
    build_bgm_seed,
    build_segment_bgm_prompt,
)
from app.core.bgm_segmenter import (
    BGM_SEGMENTATION_PIPELINE_VERSION,
    BGM_TYPE_PIPELINE_VERSION,
    DEFAULT_OUTPUT_PATH as BGM_SEGMENTS_PATH,
    bgm_source_hash,
    label_bgm_types,
    load_segments,
    save_segments,
    segment_novel_chunked,
    validate_segments,
)
from app.core.llm_client import LLMClient, SENSENOVA_FLASH_LITE_MODEL
from app.core.parser import parse
from app.core.splicer import AudioSplicer
from app.core.tts_quality import (
    TTS_CHUNKING_VERSION,
    TTS_CONTROL_VERSION,
    compact_performance_control,
    control_variants,
    duration_quality_bounds,
    split_tts_text,
)
from pydub import AudioSegment
from scripts.desktop_events import DesktopEventEmitter, DesktopEventLoggingHandler


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

NARRATOR_SPEAKER = "旁白"

STAGE_OPERATIONS = {
    "parse": "正在解析小说与角色标注",
    "gender": "正在识别角色性别",
    "emotion": "正在标注逐句情绪",
    "performance": "正在生成角色档案与逐句表演指导",
    "tts": "正在用 VoxCPM 生成语音",
    "splice": "正在拼接完整语音",
    "bgm-segment": "正在划分 BGM 场景",
    "bgm-label": "正在标注 BGM 类型与提示词",
    "bgm-generate": "正在生成 BGM 音频",
    "bgm-mix": "正在混合语音与 BGM",
    "illustration-plan": "正在规划插图",
    "illustrations": "正在审核提示词并生成插图",
    "video": "正在生成字幕与横竖版视频",
}

DESKTOP_EVENTS = DesktopEventEmitter()


class PipelineError(RuntimeError):
    """A stage failed or produced an invalid artifact."""


class StopFileWatcher:
    """Turn a desktop stop-file request into a main-thread KeyboardInterrupt."""

    def __init__(
        self,
        path: Path,
        *,
        interrupt: Callable[[], Any] = _thread.interrupt_main,
        poll_seconds: float = 0.2,
    ):
        self.path = path
        self.interrupt = interrupt
        self.poll_seconds = max(0.02, float(poll_seconds))
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def watch() -> None:
            while not self._stopped.wait(self.poll_seconds):
                if not self.path.is_file():
                    continue
                DESKTOP_EVENTS.log("WARNING", f"收到桌面停止请求：{self.path}")
                self.interrupt()
                return

        self._thread = threading.Thread(
            target=watch,
            name="desktop-stop-file-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.poll_seconds * 2))


class StageProgressMonitor:
    """Poll atomically-written checkpoints while a long stage is blocking."""

    def __init__(
        self,
        stage: str,
        probe: Callable[[], tuple[int, int, str] | None],
        *,
        interval_seconds: float = 1.0,
    ):
        self.stage = stage
        self.probe = probe
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def watch() -> None:
            last_value: tuple[int, int, str] | None = None
            while not self._stopped.is_set():
                try:
                    value = self.probe()
                    if value is not None and value != last_value:
                        current, total, operation = value
                        DESKTOP_EVENTS.progress(
                            self.stage,
                            current=max(0, current),
                            total=max(1, total),
                            operation=operation,
                        )
                        last_value = value
                except Exception:
                    pass
                self._stopped.wait(self.interval_seconds)

        self._thread = threading.Thread(
            target=watch,
            name=f"desktop-progress-{self.stage}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))


def _collection_count(value: Any) -> int:
    if isinstance(value, (dict, list, tuple, set)):
        return len(value)
    return 0


def checkpoint_progress_probe(
    entries: Iterable[tuple[Path, str, int]],
    operation: str,
) -> Callable[[], tuple[int, int, str]]:
    values = tuple(entries)

    def probe() -> tuple[int, int, str]:
        completed = 0
        total = 0
        for path, key, expected in values:
            payload = read_json(path, {})
            current = _collection_count(payload.get(key)) if isinstance(payload, dict) else 0
            completed += min(max(0, expected), current)
            total += max(0, expected)
        return completed, max(1, total), f"{operation}：{completed}/{max(1, total)}"

    return probe


def tts_progress_probe(
    config: dict[str, Any],
    total: int,
) -> Callable[[], tuple[int, int, str]]:
    checkpoint = streaming_tts_checkpoint_path(config)
    directory = output_dir(config) / "segments"

    def probe() -> tuple[int, int, str]:
        payload = read_json(checkpoint, {})
        checkpoint_count = (
            _collection_count(payload.get("segments")) if isinstance(payload, dict) else 0
        )
        wav_count = sum(1 for path in directory.glob("*.wav") if nonempty_file(path))
        completed = min(total, max(checkpoint_count, wav_count))
        return completed, max(1, total), f"已生成语音：{completed}/{max(1, total)}"

    return probe


def bgm_generation_progress_probe(
    config: dict[str, Any],
    segment_count: int,
) -> Callable[[], tuple[int, int, str]]:
    manifest = output_dir(config) / "bgm/bgm_manifest.json"
    total = segment_count * int(config.get("bgm", {}).get("clips_per_segment", 3))

    def probe() -> tuple[int, int, str]:
        completed = min(total, _bgm_checkpoint_clip_count(manifest))
        return completed, max(1, total), f"已生成 BGM：{completed}/{max(1, total)}"

    return probe


def illustration_progress_probe(
    config: dict[str, Any],
    plan_count: int,
) -> Callable[[], tuple[int, int, str]]:
    variants = illustration_variant_specs(config)
    audit_path = visual_prompt_checkpoint_path(config)
    total = plan_count * (1 + len(variants))

    def probe() -> tuple[int, int, str]:
        audit = read_json(audit_path, {})
        completed = min(
            plan_count,
            _collection_count(audit.get("completed_indices")) if isinstance(audit, dict) else 0,
        )
        for variant in variants:
            checkpoint = read_json(variant["checkpoint"], {})
            completed += min(
                plan_count,
                _collection_count(checkpoint.get("images"))
                if isinstance(checkpoint, dict)
                else 0,
            )
        return completed, max(1, total), f"提示词审核与生图：{completed}/{max(1, total)}"

    return probe


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | os.PathLike[str], base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def configure_logging(path: str = "logs/run_full.log") -> None:
    log_path = resolve_path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    if DESKTOP_EVENTS.enabled:
        handlers.append(DesktopEventLoggingHandler(DESKTOP_EVENTS))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def load_config(config_path: str) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    config.setdefault("output", {})["dir"] = str(resolve_path(config.get("output", {}).get("dir", "output")))
    return config


def apply_input_overrides(
    config: dict[str, Any],
    *,
    novel_path: str | None = None,
    labels_path: str | None = None,
) -> dict[str, Any]:
    """Override this run's inputs in memory without modifying the YAML file."""

    novel = config.setdefault("novel", {})
    if novel_path:
        novel["text_path"] = str(resolve_path(novel_path))
    if labels_path:
        novel["labels_path"] = str(resolve_path(labels_path))
    return config


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def media_duration_seconds(path: Path, ffprobe: str = "ffprobe") -> float:
    if not nonempty_file(path):
        return 0.0
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def output_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config["output"]["dir"])


def bgm_segments_path(config: dict[str, Any]) -> Path:
    configured = config.get("bgm", {}).get("segments_path", str(BGM_SEGMENTS_PATH))
    return resolve_path(configured)


def speech_output_path(config: dict[str, Any]) -> Path:
    out = config["output"]
    return output_dir(config) / f"{out['filename']}.{out.get('format', 'mp3')}"


def mixed_audio_path(config: dict[str, Any]) -> Path:
    return output_dir(config) / "full_volume_bgm.mp3"


def illustration_plan_path(config: dict[str, Any]) -> Path:
    value = config.get("illustrations", {}).get("plan_path")
    return resolve_path(value) if value else output_dir(config) / "illustration_plan.json"


def illustration_output_dir(config: dict[str, Any]) -> Path:
    value = config.get("illustrations", {}).get("output_dir")
    return resolve_path(value) if value else output_dir(config) / "illustrations_agnes"


def illustration_checkpoint_path(config: dict[str, Any]) -> Path:
    value = config.get("illustrations", {}).get("checkpoint_path")
    return resolve_path(value) if value else output_dir(config) / "illustrations_agnes_checkpoint.json"


def visual_prompt_checkpoint_path(config: dict[str, Any]) -> Path:
    value = config.get("illustrations", {}).get("prompt_audit_checkpoint_path")
    return resolve_path(value) if value else ROOT / "backend/data/visual_prompt_audit.checkpoint.json"


def character_cards_path(config: dict[str, Any]) -> Path:
    value = config.get("illustrations", {}).get("character_cards_path")
    return resolve_path(value) if value else ROOT / "docs/角色卡.md"


def illustration_generation_settings(
    config: dict[str, Any],
    *,
    size: str | None = None,
) -> dict[str, Any]:
    illustration_config = config.get("illustrations", {})
    provider = str(illustration_config.get("provider", "agnes"))
    endpoint_default = (
        "http://127.0.0.1:8000/generate"
        if provider == "local-http"
        else "https://apihub.agnes-ai.com/v1/images/generations"
    )
    settings: dict[str, Any] = {
        "model": str(illustration_config.get("model", "agnes-image-2.1-flash")),
        "endpoint": str(illustration_config.get("endpoint", endpoint_default)).rstrip("/"),
        "size": str(size or illustration_config.get("size", "896x1152")),
    }
    if provider == "local-http":
        from scripts.generate_illustrations import DEFAULT_NEGATIVE_PROMPT

        settings.update(
            steps=int(illustration_config.get("steps", 25)),
            cfg=float(illustration_config.get("cfg", 7.0)),
            negative_prompt=str(
                illustration_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
            ).strip(),
            seed=int(illustration_config.get("seed", -1)),
        )
    return settings


def illustration_variant_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the portrait target and optional independent landscape target."""

    illustration_config = config.get("illustrations", {})
    portrait = {
        "name": "portrait",
        "directory": illustration_output_dir(config),
        "checkpoint": illustration_checkpoint_path(config),
        "settings": illustration_generation_settings(config),
        "composition_suffix": str(
            illustration_config.get(
                "composition_suffix",
                "vertical 7:9 portrait frame, balanced full-height composition, "
                "keep important faces and details clear of the lower subtitle-safe area",
            )
        ),
    }
    values = [portrait]
    landscape = illustration_config.get("landscape", {})
    if isinstance(landscape, dict) and landscape.get("enabled", False):
        values.append(
            {
                "name": "landscape",
                "directory": resolve_path(
                    landscape.get("output_dir", "output/illustrations_local_landscape_16x9")
                ),
                "checkpoint": resolve_path(
                    landscape.get(
                        "checkpoint_path",
                        "output/illustrations_local_landscape_16x9_checkpoint.json",
                    )
                ),
                "settings": illustration_generation_settings(
                    config,
                    size=str(landscape.get("size", "1280x720")),
                ),
                "composition_suffix": str(
                    landscape.get(
                        "composition_suffix",
                        "cinematic 16:9 landscape frame, expand the environment horizontally, "
                        "keep important faces and details clear of the lower subtitle-safe area",
                    )
                ),
            }
        )
    return values


def illustration_validation_options(
    config: dict[str, Any],
    *,
    settings: dict[str, Any] | None = None,
    composition_suffix: str | None = None,
) -> dict[str, Any]:
    illustration_config = config.get("illustrations", {})
    settings = settings or illustration_generation_settings(config)
    return {
        "expected_provider": str(illustration_config.get("provider", "agnes")),
        "expected_model": settings["model"],
        "expected_size": settings["size"],
        "expected_endpoint": settings["endpoint"],
        "expected_generation_settings": settings,
        "prompt_audit_enabled": bool(
            illustration_config.get("prompt_audit_enabled", True)
        ),
        "novel_path": resolve_path(config["novel"]["text_path"]),
        "character_cards": character_cards_path(config),
        "composition_suffix": (
            str(illustration_config.get("composition_suffix", ""))
            if composition_suffix is None
            else composition_suffix
        ),
    }


def video_output_path(config: dict[str, Any]) -> Path:
    value = config.get("video", {}).get("output_path")
    return resolve_path(value) if value else output_dir(config) / "illustration_video_agnes_subtitled.mp4"


def video_subtitle_path(config: dict[str, Any]) -> Path:
    value = config.get("video", {}).get("subtitle_path")
    return resolve_path(value) if value else output_dir(config) / "illustration_video_agnes_subtitles.srt"


def video_variant_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    illustration_variants = {item["name"]: item for item in illustration_variant_specs(config)}
    video_config = config.get("video", {})
    values = [
        {
            "name": "portrait",
            "illustrations": illustration_variants["portrait"],
            "output": video_output_path(config),
            "subtitle": video_subtitle_path(config),
            "options": video_config,
        }
    ]
    landscape_images = illustration_variants.get("landscape")
    landscape_video = video_config.get("landscape", {})
    if (
        landscape_images
        and isinstance(landscape_video, dict)
        and landscape_video.get("enabled", True)
    ):
        values.append(
            {
                "name": "landscape",
                "illustrations": landscape_images,
                "output": resolve_path(
                    landscape_video.get(
                        "output_path",
                        "output/illustration_video_local_landscape_16x9_subtitled.mp4",
                    )
                ),
                "subtitle": resolve_path(
                    landscape_video.get(
                        "subtitle_path",
                        "output/illustration_video_local_landscape_16x9_subtitles.srt",
                    )
                ),
                "options": {**video_config, **landscape_video},
            }
        )
    return values


def h3_video_enabled(config: dict[str, Any]) -> bool:
    value = config.get("video", {}).get("h3", {})
    return isinstance(value, dict) and bool(value.get("enabled", False))


def h3_variant_spec(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Resolve independent H3 generation and render caches for one aspect ratio."""

    video_config = config.get("video", {})
    h3_config = video_config.get("h3", {})
    if not isinstance(h3_config, dict):
        h3_config = {}
    variant_overrides = h3_config.get(variant["name"], {})
    if not isinstance(variant_overrides, dict):
        variant_overrides = {}
    defaults = (
        {"width": 960, "height": 544}
        if variant["name"] == "landscape"
        else {"width": 672, "height": 864}
    )
    default_composition = (
        "cinematic 16:9 landscape staging; use environment and two-character spatial "
        "relationships, keeping important action clear of the lower subtitle area"
        if variant["name"] == "landscape"
        else "vertical 7:9 staging; prioritize faces, gestures, and depth layers while "
        "keeping important action clear of the lower subtitle area"
    )
    root = resolve_path(h3_config.get("output_dir", "output/h3_video")) / variant["name"]
    mode = str(h3_config.get("mode", "native-chain")).strip().lower()
    if mode not in {"native-chain", "continuous-chain", "illustration-bridge"}:
        raise PipelineError(
            "video.h3.mode must be 'native-chain', 'continuous-chain', or "
            "'illustration-bridge'"
        )
    width = int(variant_overrides.get("width", h3_config.get("width", defaults["width"])))
    height = int(
        variant_overrides.get("height", h3_config.get("height", defaults["height"]))
    )
    minimum_duration = int(h3_config.get("minimum_duration", 5))
    maximum_duration = int(h3_config.get("maximum_duration", 10))
    max_chain_length = int(h3_config.get("max_chain_length", 3))
    request_timeout = float(h3_config.get("request_timeout", 60.0))
    poll_seconds = float(h3_config.get("poll_seconds", 15.0))
    job_timeout = float(h3_config.get("job_timeout", 14400.0))
    max_attempts = int(h3_config.get("max_attempts", 3))
    max_freeze_ratio = float(h3_config.get("max_freeze_ratio", 0.65))
    max_black_ratio = float(h3_config.get("max_black_ratio", 0.20))
    generation_timeout = int(h3_config.get("generation_timeout", 15552000))
    render_timeout = int(h3_config.get("render_timeout", 604800))
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise PipelineError("H3 width and height must be positive multiples of 16")
    if not 5 <= minimum_duration <= maximum_duration <= 15:
        raise PipelineError("H3 duration range must satisfy 5 <= minimum <= maximum <= 15")
    if min(request_timeout, poll_seconds, job_timeout) <= 0:
        raise PipelineError("H3 request, poll, and job timeouts must be positive")
    if (
        max_attempts < 1
        or max_chain_length < 1
        or generation_timeout <= 0
        or render_timeout <= 0
    ):
        raise PipelineError("H3 attempts and pipeline timeouts must be positive")
    if not 0 <= max_freeze_ratio <= 1 or not 0 <= max_black_ratio <= 1:
        raise PipelineError("H3 quality ratios must be between 0 and 1")
    legacy_root_value = h3_config.get("reuse_output_dir")
    legacy_root = (
        resolve_path(legacy_root_value) / variant["name"]
        if legacy_root_value
        else None
    )
    return {
        "mode": mode,
        "endpoint": str(
            variant_overrides.get(
                "endpoint",
                h3_config.get("endpoint", "http://172.31.102.189:8189"),
            )
        ).rstrip("/"),
        "width": width,
        "height": height,
        "minimum_duration": minimum_duration,
        "maximum_duration": maximum_duration,
        "max_chain_length": max_chain_length,
        "request_timeout": request_timeout,
        "poll_seconds": poll_seconds,
        "job_timeout": job_timeout,
        "max_attempts": max_attempts,
        "max_freeze_ratio": max_freeze_ratio,
        "max_black_ratio": max_black_ratio,
        "composition_direction": str(
            variant_overrides.get("composition_direction", default_composition)
        ),
        "generation_timeout": generation_timeout,
        "render_timeout": render_timeout,
        "clips_dir": root / "clips",
        "frames_dir": root / "continuation_frames",
        "keyframes_dir": root / "keyframes",
        "checkpoint": root / "h3_clips.checkpoint.json",
        "segments_dir": root / "render_segments",
        "render_checkpoint": root / "h3_render.checkpoint.json",
        "shot_plan": resolve_path(
            h3_config.get("shot_plan_path", "backend/data/h3_shot_plan.json")
        ),
        "legacy_checkpoint": (
            legacy_root / "h3_clips.checkpoint.json" if legacy_root is not None else None
        ),
    }


def _completed_checkpoint_records(path: Path, key: str) -> tuple[int, int]:
    payload = read_json(path, {})
    records = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return 0, 0
    completed = sum(
        1
        for record in records
        if isinstance(record, dict) and record.get("status") in {"success", "skipped"}
    )
    return completed, len(records)


def h3_clip_checkpoint_complete(
    path: Path,
    *,
    mode: str,
    plan_count: int,
) -> tuple[bool, int, int, str]:
    payload = read_json(path, {})
    records = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return False, 0, 0, "H3 clip checkpoint is missing"
    completed = sum(
        1
        for record in records
        if isinstance(record, dict) and record.get("status") in {"success", "skipped"}
    )
    if mode == "continuous-chain":
        beats = {
            int(record.get("beat_index", -1))
            for record in records
            if isinstance(record, dict)
        }
        coverage = payload.get("coverage")
        valid = (
            payload.get("mode") == mode
            and bool(records)
            and completed == len(records)
            and all(
                isinstance(record, dict) and record.get("status") == "success"
                for record in records
            )
            and beats == set(range(plan_count))
            and isinstance(coverage, dict)
            and coverage.get("complete") is True
            and int(coverage.get("beat_count", -1)) == plan_count
            and abs(
                float(coverage.get("planned_seconds", 0))
                - float(coverage.get("required_seconds", -1))
            )
            <= 1e-3
        )
        return (
            valid,
            completed,
            len(records),
            "" if valid else "continuous H3 dynamic coverage is incomplete",
        )
    expected = plan_count if mode == "native-chain" else max(0, plan_count - 1)
    valid = len(records) == expected and completed == len(records)
    return valid, completed, len(records), "" if valid else "H3 clip checkpoint is incomplete"


def h3_video_progress_probe(
    config: dict[str, Any],
    plan_count: int,
) -> Callable[[], tuple[int, int, str]]:
    variants = video_variant_specs(config)
    h3_config = config.get("video", {}).get("h3", {})
    mode = str(h3_config.get("mode", "native-chain")) if isinstance(h3_config, dict) else ""
    native_modes = {"native-chain", "continuous-chain"}
    fallback_clip_count = plan_count if mode in native_modes else max(0, plan_count - 1)

    def probe() -> tuple[int, int, str]:
        completed = 0
        total = 0
        for variant in variants:
            spec = h3_variant_spec(config, variant)
            clip_done, clip_total = _completed_checkpoint_records(spec["checkpoint"], "clips")
            render_done, render_total = _completed_checkpoint_records(
                spec["render_checkpoint"], "segments"
            )
            clip_total = clip_total or fallback_clip_count
            render_total = render_total or plan_count
            completed += min(clip_total, clip_done)
            completed += min(plan_count, render_done)
            if nonempty_file(variant["output"]):
                completed += 1
            total += clip_total + render_total + 1
        total = max(1, total)
        return completed, total, f"H3 过渡与成片：{completed}/{total}"

    return probe


def stage_slice(from_stage: str | None, to_stage: str | None) -> tuple[str, ...]:
    start = STAGES.index(from_stage) if from_stage else 0
    end = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    if start > end:
        raise ValueError(f"--from-stage {STAGES[start]} comes after --to-stage {STAGES[end]}")
    return STAGES[start : end + 1]


class PipelineRecorder:
    """Persist the latest status, elapsed time, and artifacts for each stage."""

    def __init__(self, path: Path, selected: Iterable[str]):
        self.path = path
        previous = read_json(path, {})
        self.data = previous if isinstance(previous, dict) else {}
        self.data.update(
            {
                "version": 1,
                "root": str(ROOT),
                "selected_stages": list(selected),
                "run_started_at": utc_now(),
                "run_status": "running",
            }
        )
        self.data.pop("run_finished_at", None)
        self.data.pop("run_error", None)
        self.data.setdefault("stages", {})
        self.save()

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        write_json(self.path, self.data)

    def record(
        self,
        stage: str,
        status: str,
        elapsed: float,
        artifacts: Iterable[Path | str] = (),
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "artifacts": [str(item) for item in artifacts],
            "finished_at": utc_now(),
        }
        if error:
            entry["error"] = error
        self.data["stages"][stage] = entry
        if status in {"failed", "interrupted"}:
            self.data["run_status"] = status
        self.save()

    def finish(self) -> None:
        self.data["run_status"] = "complete"
        self.data["run_finished_at"] = utc_now()
        self.save()


def execute_stage(
    recorder: PipelineRecorder,
    stage: str,
    function: Callable[[], Any],
    artifacts: Callable[[Any], Iterable[Path | str]] | Iterable[Path | str] = (),
    progress_probe: Callable[[], tuple[int, int, str] | None] | None = None,
) -> Any:
    stage_index = STAGES.index(stage) + 1
    operation = STAGE_OPERATIONS[stage]
    print(f"\n[{stage_index}/{len(STAGES)}] {stage}")
    DESKTOP_EVENTS.stage(
        stage,
        index=stage_index,
        total=len(STAGES),
        status="running",
        operation=operation,
    )
    DESKTOP_EVENTS.progress(
        stage,
        current=0,
        total=1,
        operation=operation,
    )
    monitor = StageProgressMonitor(stage, progress_probe) if progress_probe else None
    if monitor is not None:
        monitor.start()
    started = time.monotonic()
    try:
        result = function()
        if monitor is not None:
            monitor.stop()
        produced = tuple(artifacts(result) if callable(artifacts) else artifacts)
        elapsed = time.monotonic() - started
        recorder.record(stage, "complete", elapsed, produced)
        DESKTOP_EVENTS.progress(
            stage,
            current=1,
            total=1,
            operation=f"{operation}完成",
            status="complete",
        )
        DESKTOP_EVENTS.stage(
            stage,
            index=stage_index,
            total=len(STAGES),
            status="complete",
            elapsed_seconds=elapsed,
            operation=f"{operation}完成",
            artifacts=[str(item) for item in produced],
        )
        return result
    except KeyboardInterrupt:
        if monitor is not None:
            monitor.stop()
        elapsed = time.monotonic() - started
        error = "interrupted by user"
        recorder.record(stage, "interrupted", elapsed, error=error)
        DESKTOP_EVENTS.stage(
            stage,
            index=stage_index,
            total=len(STAGES),
            status="interrupted",
            elapsed_seconds=elapsed,
            operation=f"{operation}已停止",
            error=error,
        )
        raise
    except Exception as exc:
        if monitor is not None:
            monitor.stop()
        elapsed = time.monotonic() - started
        DESKTOP_EVENTS.log("ERROR", str(exc), stage=stage)
        recorder.record(stage, "failed", elapsed, error=str(exc))
        DESKTOP_EVENTS.stage(
            stage,
            index=stage_index,
            total=len(STAGES),
            status="failed",
            elapsed_seconds=elapsed,
            operation=f"{operation}失败",
            error=str(exc),
        )
        raise


def record_skipped(recorder: PipelineRecorder, stage: str, reason: str) -> None:
    stage_index = STAGES.index(stage) + 1
    print(f"\n[{stage_index}/{len(STAGES)}] {stage}: skipped ({reason})")
    recorder.record(stage, "skipped", 0.0, error=reason)
    DESKTOP_EVENTS.progress(
        stage,
        current=1,
        total=1,
        operation=f"已跳过：{reason}",
        status="skipped",
    )
    DESKTOP_EVENTS.stage(
        stage,
        index=stage_index,
        total=len(STAGES),
        status="skipped",
        operation=f"已跳过：{reason}",
        error=reason,
    )


def step_parse(config: dict[str, Any]) -> tuple[list[dict], list[str], str]:
    novel_path = resolve_path(config["novel"]["text_path"])
    labels_path = resolve_path(config["novel"]["labels_path"])
    if not novel_path.is_file() or not labels_path.is_file():
        raise PipelineError(f"Novel inputs are missing: {novel_path}, {labels_path}")
    novel_text = novel_path.read_text(encoding="utf-8")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dialogues, characters = parse(novel_text, labels)
    if not dialogues:
        raise PipelineError("Parser produced no dialogue segments")
    print(f"  parsed {len(dialogues)} dialogues and {len(characters)} characters")
    DESKTOP_EVENTS.log(
        "INFO",
        f"解析完成：{len(dialogues)} 个语音片段，{len(characters)} 个角色",
        stage="parse",
    )
    return dialogues, characters, novel_text


def apply_dialogue_selection(
    dialogues: list[dict], characters: list[str], limit: int, range_value: str
) -> tuple[list[dict], list[str]]:
    if limit > 0:
        dialogues = dialogues[:limit]
    elif range_value:
        try:
            start_text, end_text = range_value.split("-", 1)
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise PipelineError("--range must use START-END integer syntax") from exc
        if start < 0 or end <= start:
            raise PipelineError("--range requires 0 <= START < END")
        dialogues = dialogues[start:end]
    if limit > 0 or range_value:
        characters = sorted({item.get("speaker", "") for item in dialogues if item.get("speaker")})
    if not dialogues:
        raise PipelineError("The selected dialogue range is empty")
    return dialogues, characters


def build_flash_lite_client(module_name: str) -> LLMClient:
    return LLMClient.for_flash_lite(module_name)


def gender_result_path() -> Path:
    return ROOT / "backend" / "data" / "gender_results.json"


def emotion_result_path() -> Path:
    return ROOT / "backend" / "data" / "emotion_results.json"


def performance_profile_result_path(config: dict[str, Any] | None = None) -> Path:
    configured = (config or {}).get("performance", {}).get("profile_output_path")
    return resolve_path(configured) if configured else ROOT / "backend" / "data" / "performance_profiles.json"


def performance_result_path(config: dict[str, Any] | None = None) -> Path:
    configured = (config or {}).get("performance", {}).get("output_path")
    return resolve_path(configured) if configured else ROOT / "backend" / "data" / "performance_directions.json"


def supplemental_performance_profile_result_path(config: dict[str, Any]) -> Path:
    configured = config.get("performance", {}).get("supplemental_profile_output_path")
    return (
        resolve_path(configured)
        if configured
        else ROOT / "backend" / "data" / "performance_profiles_supplemental.json"
    )


def supplemental_performance_result_path(config: dict[str, Any]) -> Path:
    configured = config.get("performance", {}).get("supplemental_output_path")
    return (
        resolve_path(configured)
        if configured
        else ROOT / "backend" / "data" / "performance_directions_supplemental.json"
    )


def step_gender(
    config: dict[str, Any], characters: list[str], dialogues: list[dict], novel_text: str
) -> dict[str, Any]:
    from app.core.gender_identifier import (
        GENDER_PIPELINE_VERSION,
        gender_source_hash,
        identify_all_genders,
    )

    path = gender_result_path()
    cached = read_json(path, {})
    names = [name for name in characters if name != "旁白"]
    source_hash = gender_source_hash(novel_text, names, dialogues)
    if gender_cache_valid(cached, names, source_hash):
        print(f"  using cached gender results: {path}")
        return cached

    client = build_flash_lite_client("gender")
    checkpoint_path = ROOT / "backend/data/gender_results.checkpoint.json"
    identified = identify_all_genders(
        names,
        novel_text,
        client=client,
        max_tool_steps=8,
        dialogues=dialogues,
        checkpoint_path=str(checkpoint_path),
        resume=True,
    )
    results = {item["character_name"]: item for item in identified}
    if "旁白" in characters:
        results["旁白"] = {
            "gender": "unknown",
            "confidence": 1.0,
            "evidence": "Narration role; gender identification is not applicable.",
        }
    checkpoint = read_json(checkpoint_path, {})
    results["_meta"] = {
        "model": SENSENOVA_FLASH_LITE_MODEL,
        "pipeline_version": GENDER_PIPELINE_VERSION,
        "source_hash": source_hash,
        "llm_usage": checkpoint.get("llm_usage", client.usage_summary()),
    }
    write_json(path, results)
    return results


def gender_cache_valid(
    results: dict[str, Any],
    character_names: list[str] | None = None,
    source_hash: str | None = None,
) -> bool:
    from app.core.gender_identifier import GENDER_PIPELINE_VERSION

    meta = results.get("_meta", {})
    if (
        meta.get("model") != SENSENOVA_FLASH_LITE_MODEL
        or meta.get("pipeline_version") != GENDER_PIPELINE_VERSION
        or not meta.get("source_hash")
    ):
        return False
    if source_hash is not None and meta.get("source_hash") != source_hash:
        return False
    if character_names is not None and not set(character_names).issubset(results):
        return False
    return True


def require_gender_results(
    characters: list[str], dialogues: list[dict], novel_text: str
) -> dict[str, Any]:
    from app.core.gender_identifier import gender_source_hash

    results = read_json(gender_result_path(), {})
    names = [name for name in characters if name != "旁白"]
    if not gender_cache_valid(results, names, gender_source_hash(novel_text, names, dialogues)):
        raise PipelineError("Valid gender cache is required before starting from the TTS stage")
    return results


def step_emotion(
    config: dict[str, Any], dialogues: list[dict], novel_text: str, force_reprocess: bool = False
) -> dict[str, Any]:
    from app.core.emotion_labeler import (
        EMOTION_PIPELINE_VERSION,
        emotion_source_hash,
        label_all_emotions,
    )

    path = emotion_result_path()
    cached = read_json(path, {})
    source_hash = emotion_source_hash(novel_text, dialogues)
    if not force_reprocess and emotion_cache_valid(cached, dialogues, source_hash):
        print(f"  using cached emotion results: {path}")
        return cached.get("results", {})

    client = build_flash_lite_client("emotion")
    checkpoint_path = ROOT / "backend/data/emotion_results.checkpoint.json"
    results = label_all_emotions(
        dialogues,
        novel_text,
        client=client,
        checkpoint_path=str(checkpoint_path),
        resume=not force_reprocess,
        max_tool_steps=6,
        item_retries=3,
    )
    checkpoint = read_json(checkpoint_path, {})
    write_json(
        path,
        {
            "meta": {
                "model": SENSENOVA_FLASH_LITE_MODEL,
                "pipeline_version": EMOTION_PIPELINE_VERSION,
                "source_hash": source_hash,
            },
            "results": results,
            "llm_usage": checkpoint.get("llm_usage", client.usage_summary()),
        },
    )
    return results


def emotion_cache_valid(
    payload: dict[str, Any],
    dialogues: list[dict] | None = None,
    source_hash: str | None = None,
) -> bool:
    from app.core.emotion_labeler import EMOTION_PIPELINE_VERSION

    meta = payload.get("meta", {})
    if (
        meta.get("model") != SENSENOVA_FLASH_LITE_MODEL
        or meta.get("pipeline_version") != EMOTION_PIPELINE_VERSION
        or not meta.get("source_hash")
    ):
        return False
    if source_hash is not None and meta.get("source_hash") != source_hash:
        return False
    results = payload.get("results")
    if not isinstance(results, dict):
        return False
    if dialogues is not None:
        expected = {
            str(index)
            for index, dialogue in enumerate(dialogues)
            if dialogue.get("speaker") and dialogue.get("speaker") not in {"旁白", "narrator", "Narrator"}
        }
        if set(results) != expected:
            return False
    return True


def require_emotion_results(dialogues: list[dict], novel_text: str) -> dict[str, Any]:
    from app.core.emotion_labeler import emotion_source_hash

    payload = read_json(emotion_result_path(), {})
    if not emotion_cache_valid(payload, dialogues, emotion_source_hash(novel_text, dialogues)):
        raise PipelineError("Valid emotion cache is required before starting from the TTS stage")
    return payload.get("results", {})


def build_emotion_prefix(emotion: str | None = None, tone: str | None = None) -> str:
    emotion_map = {
        "happy": "欢快活泼",
        "sad": "低落悲伤",
        "angry": "愤怒生气",
        "surprised": "惊讶震惊",
        "calm": "平静冷静",
        "nervous": "紧张焦虑",
        "cold": "冷漠淡漠",
    }
    tone_map = {
        "loud": "大声",
        "soft": "轻声",
        "whisper": "低语",
        "gentle": "温柔",
        "serious": "严肃",
        "sarcastic": "讽刺",
        "stutter": "结巴",
    }
    parts = [value for value in (emotion_map.get(emotion), tone_map.get(tone)) if value]
    return f"({'，'.join(parts)})" if parts else ""


def effective_speaker(speaker: Any) -> str:
    """Map parser-level non-character fragments to the narrator at output boundaries."""
    return str(speaker or "").strip() or NARRATOR_SPEAKER


def get_reference_audio(speaker: str, gender: str, config: dict[str, Any]) -> str:
    speaker = effective_speaker(speaker)
    characters = config.get("characters", {})
    if speaker in characters:
        return str(resolve_path(characters[speaker]))
    defaults = config.get("default_audio", {})
    fallback = defaults.get("female" if gender == "female" else "male")
    if fallback:
        return str(resolve_path(fallback))
    raise PipelineError(f"No VoxCPM reference audio configured for {speaker!r}")


def get_voice_assignment(speaker: str, gender: str, config: dict[str, Any]) -> dict[str, str]:
    """Select the configured TTS engine and voice for a speaker.

    ``force_all_characters`` makes every line use VoxCPM. Explicit character
    references still take precedence, while every other speaker uses the
    gender-specific default reference audio.
    """
    speaker = effective_speaker(speaker)
    voxcpm = config.get("voxcpm", {})
    if bool(voxcpm.get("force_all_characters", False)):
        return {
            "engine": "voxcpm",
            "reference_audio": get_reference_audio(speaker, gender, config),
        }

    configured = voxcpm.get("characters")
    if configured is None:
        clone_characters = set(config.get("characters", {}))
    elif isinstance(configured, dict):
        clone_characters = set(configured)
    else:
        clone_characters = set(configured)

    if speaker in clone_characters:
        override = configured.get(speaker) if isinstance(configured, dict) else None
        reference = override or config.get("characters", {}).get(speaker)
        if not reference:
            reference = get_reference_audio(speaker, gender, config)
        return {"engine": "voxcpm", "reference_audio": str(resolve_path(reference))}

    edge_config = config.get("edge_tts", {})
    voice_id = edge_config.get(
        "female_voice" if gender == "female" else "male_voice",
        "zh-CN-XiaoxiaoNeural" if gender == "female" else "zh-CN-YunxiNeural",
    )
    return {"engine": "edge-tts", "voice_id": voice_id}


def performance_target_indices(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
) -> list[int]:
    """Return dialogue indices that receive detailed LLM performance direction.

    This is the legacy/primary stream. Keeping its exact target set preserves
    the existing checkpoint; forced-clone speakers outside this set are handled
    by the disjoint supplemental performance stream.
    """
    configured = config.get("performance", {}).get("characters")
    if configured is None:
        target_speakers = set(config.get("characters", {}))
    elif isinstance(configured, dict):
        target_speakers = set(configured)
    else:
        target_speakers = set(configured)

    targets: list[int] = []
    for index, dialogue in enumerate(dialogues):
        speaker = str(dialogue.get("speaker", "")).strip()
        if speaker in target_speakers:
            targets.append(index)
    return targets


def all_performance_target_indices(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
) -> list[int]:
    if bool(config.get("voxcpm", {}).get("force_all_characters", False)):
        return list(range(len(dialogues)))
    return performance_target_indices(config, dialogues, gender_results)


def supplemental_performance_target_indices(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
) -> list[int]:
    primary = set(performance_target_indices(config, dialogues, gender_results))
    return [
        index
        for index in all_performance_target_indices(config, dialogues, gender_results)
        if index not in primary
    ]


def _performance_cards(config: dict[str, Any]) -> str:
    configured = config.get("performance", {}).get(
        "character_cards_path",
        config.get("illustrations", {}).get("character_cards_path", "docs/角色卡.md"),
    )
    path = resolve_path(configured)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _performance_inputs(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
) -> tuple[list[int], list[str]]:
    targets = performance_target_indices(config, dialogues, gender_results)
    speakers = list(
        dict.fromkeys(str(dialogues[index].get("speaker", "")).strip() for index in targets)
    )
    return targets, speakers


def _performance_groups(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
) -> list[dict[str, Any]]:
    performance_config = config.get("performance", {})
    primary_targets, primary_speakers = _performance_inputs(config, dialogues, gender_results)
    groups = [{
        "name": "primary",
        "targets": primary_targets,
        "speakers": primary_speakers,
        "dialogues": dialogues,
        "profile_path": performance_profile_result_path(config),
        "profile_checkpoint": resolve_path(
            performance_config.get(
                "profile_checkpoint_path",
                "backend/data/performance_profiles.checkpoint.json",
            )
        ),
        "output_path": performance_result_path(config),
        "direction_checkpoint": resolve_path(
            performance_config.get(
                "checkpoint_path",
                "backend/data/performance_directions.checkpoint.json",
            )
        ),
    }]
    supplemental_targets = supplemental_performance_target_indices(
        config,
        dialogues,
        gender_results,
    )
    if supplemental_targets:
        supplemental_dialogues = [dict(dialogue) for dialogue in dialogues]
        for index in supplemental_targets:
            if not str(supplemental_dialogues[index].get("speaker", "")).strip():
                supplemental_dialogues[index]["speaker"] = NARRATOR_SPEAKER
        supplemental_speakers = list(dict.fromkeys(
            str(supplemental_dialogues[index].get("speaker", "")).strip()
            for index in supplemental_targets
        ))
        groups.append({
            "name": "supplemental",
            "targets": supplemental_targets,
            "speakers": supplemental_speakers,
            "dialogues": supplemental_dialogues,
            "profile_path": supplemental_performance_profile_result_path(config),
            "profile_checkpoint": resolve_path(
                performance_config.get(
                    "supplemental_profile_checkpoint_path",
                    "backend/data/performance_profiles_supplemental.checkpoint.json",
                )
            ),
            "output_path": supplemental_performance_result_path(config),
            "direction_checkpoint": resolve_path(
                performance_config.get(
                    "supplemental_checkpoint_path",
                    "backend/data/performance_directions_supplemental.checkpoint.json",
                )
            ),
        })
    return groups


def _validate_performance_group_cache(
    config: dict[str, Any],
    group: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    emotion_results: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    from app.core.performance_director import validate_performance_payload, validate_profile_payload

    cards = _performance_cards(config)
    group_dialogues = group.get("dialogues", dialogues)
    profile_payload = read_json(group["profile_path"], {})
    profile_problems = validate_profile_payload(
        profile_payload,
        group["speakers"],
        group_dialogues,
        novel_text,
        character_cards_text=cards,
    )
    if profile_problems:
        return profile_problems, {}, {}
    profiles = profile_payload.get("profiles", {})
    performance_config = config.get("performance", {})
    output_payload = read_json(group["output_path"], {})
    problems = validate_performance_payload(
        output_payload,
        group["targets"],
        group_dialogues,
        novel_text,
        profiles,
        emotion_results,
        character_cards_text=cards,
        context_radius=int(performance_config.get("context_radius", 100)),
        min_control_chars=int(performance_config.get("min_control_chars", 18)),
        max_control_chars=int(performance_config.get("max_control_chars", 140)),
    )
    return problems, profiles, output_payload.get("results", {}) if not problems else {}


def performance_cache_valid(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
) -> tuple[bool, list[str], dict[str, dict[str, Any]]]:
    problems: list[str] = []
    profiles: dict[str, dict[str, Any]] = {}
    for group in _performance_groups(config, dialogues, gender_results):
        group_problems, group_profiles, _ = _validate_performance_group_cache(
            config,
            group,
            dialogues,
            novel_text,
            emotion_results,
        )
        problems.extend(f"{group['name']}: {problem}" for problem in group_problems)
        profiles.update(group_profiles)
    return not problems, problems, profiles


def _run_performance_group(
    config: dict[str, Any],
    group: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    emotion_results: dict[str, Any],
    client: LLMClient,
) -> dict[str, Any]:
    from app.core.performance_director import (
        PERFORMANCE_DIRECTION_PIPELINE_VERSION,
        PERFORMANCE_PROMPT_SIGNATURE,
        PERFORMANCE_PROFILE_PIPELINE_VERSION,
        PROFILE_PROMPT_SIGNATURE,
        build_performance_profiles,
        direct_all_performances,
        performance_direction_source_hash,
        performance_profile_source_hash,
        validate_performance_payload,
        validate_profile_payload,
    )

    performance_config = config.get("performance", {})
    group_dialogues = group.get("dialogues", dialogues)
    targets = group["targets"]
    speakers = group["speakers"]
    cards = _performance_cards(config)
    profile_path = group["profile_path"]
    profile_checkpoint = group["profile_checkpoint"]
    profile_payload = read_json(profile_path, {})
    profile_problems = validate_profile_payload(
        profile_payload,
        speakers,
        group_dialogues,
        novel_text,
        character_cards_text=cards,
    )
    force_profiles = bool(performance_config.get("force_profiles", False))
    if profile_problems or force_profiles:
        profiles = build_performance_profiles(
            speakers,
            novel_text,
            group_dialogues,
            character_cards_text=cards,
            client=client,
            checkpoint_path=profile_checkpoint,
            resume=not force_profiles,
            max_agent_rounds=int(performance_config.get("max_agent_rounds", 8)),
        )
        profile_checkpoint_payload = read_json(profile_checkpoint, {})
        profile_hash = performance_profile_source_hash(novel_text, speakers, group_dialogues, cards)
        profile_payload = {
            "meta": {
                "model": SENSENOVA_FLASH_LITE_MODEL,
                "pipeline_version": PERFORMANCE_PROFILE_PIPELINE_VERSION,
                "prompt_signature": PROFILE_PROMPT_SIGNATURE,
                "source_hash": profile_hash,
                "target_speakers": speakers,
            },
            "profiles": profiles,
            "llm_usage": profile_checkpoint_payload.get("llm_usage", {}),
        }
        write_json(profile_path, profile_payload)
    else:
        profiles = profile_payload["profiles"]
        print(f"  using cached performance profiles: {profile_path}")

    output_path = group["output_path"]
    output_payload = read_json(output_path, {})
    direction_problems = validate_performance_payload(
        output_payload,
        targets,
        group_dialogues,
        novel_text,
        profiles,
        emotion_results,
        character_cards_text=cards,
        context_radius=int(performance_config.get("context_radius", 100)),
        min_control_chars=int(performance_config.get("min_control_chars", 18)),
        max_control_chars=int(performance_config.get("max_control_chars", 140)),
    )
    force_directions = bool(performance_config.get("force_directions", False))
    if not direction_problems and not force_directions:
        print(f"  using cached performance directions: {output_path}")
        return output_payload["results"]

    direction_checkpoint = group["direction_checkpoint"]
    results = direct_all_performances(
        targets,
        group_dialogues,
        novel_text,
        profiles,
        emotion_results,
        character_cards_text=cards,
        client=client,
        checkpoint_path=direction_checkpoint,
        resume=not force_directions,
        context_radius=int(performance_config.get("context_radius", 100)),
        min_control_chars=int(performance_config.get("min_control_chars", 18)),
        max_control_chars=int(performance_config.get("max_control_chars", 140)),
        max_agent_rounds=int(performance_config.get("max_agent_rounds", 8)),
        item_retries=int(performance_config.get("item_retries", 3)),
    )
    direction_checkpoint_payload = read_json(direction_checkpoint, {})
    source_hash = performance_direction_source_hash(
        novel_text,
        targets,
        group_dialogues,
        profiles,
        emotion_results,
        cards,
        int(performance_config.get("context_radius", 100)),
        int(performance_config.get("min_control_chars", 18)),
        int(performance_config.get("max_control_chars", 140)),
    )
    output_payload = {
        "meta": {
            "model": SENSENOVA_FLASH_LITE_MODEL,
            "pipeline_version": PERFORMANCE_DIRECTION_PIPELINE_VERSION,
            "prompt_signature": PERFORMANCE_PROMPT_SIGNATURE,
            "source_hash": source_hash,
            "target_count": len(targets),
            "profile_source_hash": profile_payload["meta"]["source_hash"],
        },
        "results": results,
        "llm_usage": direction_checkpoint_payload.get("llm_usage", {}),
    }
    final_problems = validate_performance_payload(
        output_payload,
        targets,
        group_dialogues,
        novel_text,
        profiles,
        emotion_results,
        character_cards_text=cards,
        context_radius=int(performance_config.get("context_radius", 100)),
        min_control_chars=int(performance_config.get("min_control_chars", 18)),
        max_control_chars=int(performance_config.get("max_control_chars", 140)),
    )
    if final_problems:
        raise PipelineError(f"Performance direction output failed validation: {final_problems[:5]}")
    write_json(output_path, output_payload)
    return results


def step_performance(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
) -> dict[str, Any]:
    groups = _performance_groups(config, dialogues, gender_results)
    if not groups or not any(group["targets"] for group in groups):
        raise PipelineError("Performance direction is enabled but no VoxCPM target dialogues were found")
    client = build_flash_lite_client("performance")
    merged: dict[str, Any] = {}
    for group in groups:
        print(
            f"  performance group={group['name']} targets={len(group['targets'])} "
            f"speakers={len(group['speakers'])}",
            flush=True,
        )
        results = _run_performance_group(
            config,
            group,
            dialogues,
            novel_text,
            emotion_results,
            client,
        )
        overlap = set(merged).intersection(results)
        if overlap:
            raise PipelineError(f"Performance groups overlap at indices: {sorted(overlap)[:5]}")
        merged.update(results)
    expected = {str(index) for index in all_performance_target_indices(config, dialogues, gender_results)}
    if set(merged) != expected:
        raise PipelineError(
            f"Merged performance output is incomplete: expected={len(expected)} actual={len(merged)}"
        )
    return merged


def require_performance_results(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
) -> dict[str, Any]:
    valid, problems, _ = performance_cache_valid(
        config,
        dialogues,
        novel_text,
        gender_results,
        emotion_results,
    )
    if not valid:
        raise PipelineError(f"Valid performance directions are required before TTS: {problems[:5]}")
    merged: dict[str, Any] = {}
    for group in _performance_groups(config, dialogues, gender_results):
        merged.update(read_json(group["output_path"], {}).get("results", {}))
    return merged


TTS_PIPELINE_VERSION = 7
VOXCPM_BATCH_CHECKPOINT_VERSION = 5
STREAMING_TTS_CHECKPOINT_VERSION = 1


@lru_cache(maxsize=64)
def _file_content_hash(path_value: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_audio_identity(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": _file_content_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
    }


def voxcpm_generation_options(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("voxcpm", {})
    options = {
        "cfg_value": float(raw.get("cfg_value", 2.0)),
        "inference_timesteps": int(raw.get("inference_timesteps", 10)),
        "normalize": bool(raw.get("normalize", False)),
        "reuse_reference_cache": bool(raw.get("reuse_reference_cache", True)),
        "retry_badcase": bool(raw.get("retry_badcase", True)),
        "retry_badcase_max_times": int(raw.get("retry_badcase_max_times", 3)),
        "retry_badcase_ratio_threshold": float(raw.get("retry_badcase_ratio_threshold", 6.0)),
        "max_len": int(raw.get("max_len", 4096)),
        "task_attempts": int(raw.get("task_attempts", 3)),
        "chunk_max_chars": int(raw.get("chunk_max_chars", 48)),
        "chunk_min_chars": int(raw.get("chunk_min_chars", 8)),
        "control_max_chars": int(raw.get("control_max_chars", 32)),
        "quality_min_duration_ratio": float(raw.get("quality_min_duration_ratio", 0.92)),
        "quality_max_duration_ratio": float(raw.get("quality_max_duration_ratio", 2.5)),
        "inter_chunk_pause_ms": int(raw.get("inter_chunk_pause_ms", 100)),
    }
    if options["cfg_value"] <= 0:
        raise PipelineError("voxcpm.cfg_value must be positive")
    if options["inference_timesteps"] < 1:
        raise PipelineError("voxcpm.inference_timesteps must be at least 1")
    if options["retry_badcase"] and options["retry_badcase_max_times"] < 1:
        raise PipelineError("voxcpm.retry_badcase_max_times must be at least 1")
    if options["retry_badcase_ratio_threshold"] <= 1:
        raise PipelineError("voxcpm.retry_badcase_ratio_threshold must be greater than 1")
    if options["max_len"] < 100:
        raise PipelineError("voxcpm.max_len must be at least 100")
    if options["task_attempts"] < 1:
        raise PipelineError("voxcpm.task_attempts must be at least 1")
    if options["chunk_max_chars"] < 8:
        raise PipelineError("voxcpm.chunk_max_chars must be at least 8")
    if not 1 <= options["chunk_min_chars"] <= options["chunk_max_chars"] // 2:
        raise PipelineError("voxcpm.chunk_min_chars must be between 1 and half chunk_max_chars")
    if options["control_max_chars"] < 12:
        raise PipelineError("voxcpm.control_max_chars must be at least 12")
    if not 0.5 <= options["quality_min_duration_ratio"] <= 1.5:
        raise PipelineError("voxcpm.quality_min_duration_ratio must be between 0.5 and 1.5")
    if options["quality_max_duration_ratio"] <= 1.2:
        raise PipelineError("voxcpm.quality_max_duration_ratio must be greater than 1.2")
    if not 0 <= options["inter_chunk_pause_ms"] <= 1000:
        raise PipelineError("voxcpm.inter_chunk_pause_ms must be between 0 and 1000")
    return options


@lru_cache(maxsize=8)
def _runtime_tree_identity(path_value: str, mode: str) -> dict[str, Any]:
    root = Path(path_value)
    if not root.exists():
        return {"path": str(root), "missing": True}
    excluded = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "data",
        "logs",
        "outputs",
        "output",
    }
    source_suffixes = {".py", ".json", ".toml", ".yaml", ".yml", ".md"}
    entries: list[dict[str, Any]] = []
    paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        if any(part in excluded for part in Path(relative).parts):
            continue
        if mode == "source" and path.suffix.lower() not in source_suffixes:
            continue
        stat = path.stat()
        entry: dict[str, Any] = {
            "path": relative,
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
        if mode == "source" or (path.suffix.lower() in source_suffixes and stat.st_size <= 10 * 1024 * 1024):
            entry["sha256"] = _file_content_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        entries.append(entry)
    digest = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"path": str(root.resolve()), "files": len(entries), "sha256": digest}


def voxcpm_runtime_identity(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("voxcpm", {})
    model_path = resolve_path(raw.get("model_path", "backend/models/VoxCPM2"))
    module_path = resolve_path(raw.get("module_path", "backend"))
    python_path = resolve_voxcpm_python(config)
    python_stat = python_path.stat()
    return {
        "model": _runtime_tree_identity(str(model_path), "model"),
        "module": _runtime_tree_identity(str(module_path), "source"),
        "python": {
            "path": str(python_path),
            "size": python_stat.st_size,
            "modified_ns": python_stat.st_mtime_ns,
        },
    }


def tts_fingerprint(
    dialogue: dict[str, Any],
    assignment: dict[str, str],
    emotion_result: dict[str, Any],
    performance_result: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    engine_config: dict[str, Any] = {}
    if config is not None:
        section = "voxcpm" if assignment.get("engine") == "voxcpm" else "edge_tts"
        raw_config = config.get(section, {})
        if isinstance(raw_config, dict):
            if section == "voxcpm":
                engine_config = {
                    **voxcpm_generation_options(config),
                    "runtime": voxcpm_runtime_identity(config),
                }
            else:
                engine_config = {
                    key: value
                    for key, value in raw_config.items()
                    if key not in {"timeout", "workers"}
                }
    payload = {
        "pipeline_version": TTS_PIPELINE_VERSION,
        "text": dialogue.get("text", ""),
        "speaker": dialogue.get("speaker", ""),
        "chapter": dialogue.get("chapter", "unknown"),
        "assignment": assignment,
        "reference_audio": reference_audio_identity(assignment.get("reference_audio")),
        "engine_config": engine_config,
        "style_control": (
            str((performance_result or {}).get("performance_control", ""))
            if assignment.get("engine") == "voxcpm"
            else ""
        ),
        "legacy_emotion": (
            {
                "emotion": emotion_result.get("emotion"),
                "tone": emotion_result.get("tone"),
            }
            if assignment.get("engine") == "voxcpm" and not performance_result
            else {}
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tts_source_hash(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
    performance_results: dict[str, Any],
) -> str:
    fingerprints = []
    for index, dialogue in enumerate(dialogues):
        raw_speaker = dialogue.get("speaker", "")
        speaker = effective_speaker(raw_speaker)
        gender = gender_results.get(speaker, gender_results.get(raw_speaker, {})).get("gender", "male")
        if gender not in {"male", "female"}:
            gender = "male"
        assignment = get_voice_assignment(speaker, gender, config)
        emotion = emotion_results.get(str(index), {}) if raw_speaker and speaker != NARRATOR_SPEAKER else {}
        performance = performance_results.get(str(index), {})
        fingerprints.append(tts_fingerprint(dialogue, assignment, emotion, performance, config))
    payload = json.dumps(
        {"pipeline_version": TTS_PIPELINE_VERSION, "fingerprints": fingerprints},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def synthesize_edge_tts_sync(text: str, voice_id: str) -> bytes:
    import edge_tts

    async def collect() -> bytes:
        audio = io.BytesIO()
        async for chunk in edge_tts.Communicate(text, voice_id).stream():
            if chunk.get("type") == "audio":
                audio.write(chunk["data"])
        return audio.getvalue()

    data = asyncio.run(collect())
    if not data:
        raise PipelineError("edge-tts returned empty audio")
    return data


def synthesize_edge_tts_to_wav(text: str, voice_id: str, output_path: Path) -> None:
    mp3_data = synthesize_edge_tts_sync(text, voice_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3").export(output_path, format="wav")
    if not nonempty_file(output_path):
        raise PipelineError(f"edge-tts did not create {output_path}")


def resolve_voxcpm_python(config: dict[str, Any]) -> Path:
    configured = config.get("voxcpm", {}).get("python") or os.environ.get("VOXCPM_PYTHON")
    path = resolve_path(configured) if configured else Path(sys.executable).resolve()
    if not path.is_file():
        raise PipelineError(f"VoxCPM Python interpreter not found: {path}")
    return path


def voxcpm_batch_source_hash(tasks: list[dict[str, Any]], config: dict[str, Any]) -> str:
    payload = {
        "version": VOXCPM_BATCH_CHECKPOINT_VERSION,
        "generation_options": voxcpm_generation_options(config),
        "runtime": voxcpm_runtime_identity(config),
        "tasks": [
            {
                "task_key": task.get("task_key", str(task["index"])),
                "index": task["index"],
                "chunk_index": task.get("chunk_index", 0),
                "fingerprint": task["fingerprint"],
                "output_path": task["output_path"],
            }
            for task in tasks
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def voxcpm_generation_signature(config: dict[str, Any]) -> str:
    payload = {
        "version": VOXCPM_BATCH_CHECKPOINT_VERSION,
        "generation_options": voxcpm_generation_options(config),
        "runtime": voxcpm_runtime_identity(config),
        "chunking_version": TTS_CHUNKING_VERSION,
        "control_version": TTS_CONTROL_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_voxcpm_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(output_dir(config) / "voxcpm_results.json", {})
    return payload if isinstance(payload, dict) else {}


def inspect_generated_wav(path: Path) -> dict[str, Any] | None:
    if not nonempty_file(path):
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
    except (OSError, EOFError, wave.Error):
        return None
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        return None
    stat = path.stat()
    return {
        "wav_sha256": _file_content_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
        "wav_size": stat.st_size,
        "wav_frames": frames,
        "wav_sample_rate": sample_rate,
        "wav_channels": channels,
        "wav_duration_seconds": round(frames / sample_rate, 6),
    }


def valid_voxcpm_result(result: Any, task: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or not all(
        key in task for key in ("index", "fingerprint", "output_path")
    ):
        return False
    if (
        result.get("status") != "ok"
        or (
            "task_key" in task
            and result.get("task_key") != str(task["task_key"])
        )
        or result.get("index") != task["index"]
        or result.get("chunk_index", 0) != task.get("chunk_index", 0)
        or result.get("fingerprint") != task["fingerprint"]
        or result.get("output_path") != task["output_path"]
    ):
        return False
    actual = inspect_generated_wav(Path(task["output_path"]))
    if not actual or actual["wav_duration_seconds"] < 0.08:
        return False
    return all(result.get(key) == value for key, value in actual.items())


def create_voxcpm_script(tasks: list[dict[str, Any]], config: dict[str, Any]) -> str:
    model_path = resolve_path(config.get("voxcpm", {}).get("model_path", "backend/models/VoxCPM2"))
    module_path = resolve_path(config.get("voxcpm", {}).get("module_path", "backend"))
    results_path = output_dir(config) / "voxcpm_results.json"
    tasks_literal = repr(json.dumps(tasks, ensure_ascii=False))
    batch_source_hash = voxcpm_batch_source_hash(tasks, config)
    generation_signature = voxcpm_generation_signature(config)
    options = voxcpm_generation_options(config)
    cfg_value = options["cfg_value"]
    inference_timesteps = options["inference_timesteps"]
    normalize = options["normalize"]
    reuse_reference_cache = options["reuse_reference_cache"]
    retry_badcase = options["retry_badcase"]
    retry_badcase_max_times = options["retry_badcase_max_times"]
    retry_badcase_ratio_threshold = options["retry_badcase_ratio_threshold"]
    max_len = options["max_len"]
    task_attempts = options["task_attempts"]
    return f'''import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, {str(module_path)!r})
try:
    from models.voxcpm import VoxCPM
except ImportError:
    from voxcpm import VoxCPM
import soundfile as sf

tasks = json.loads({tasks_literal})
checkpoint_path = {str(results_path)!r}
checkpoint_version = {VOXCPM_BATCH_CHECKPOINT_VERSION!r}
source_hash = {batch_source_hash!r}
generation_signature = {generation_signature!r}
results = {{}}
try:
    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        prior = json.load(handle)
    if (
        prior.get("version") == checkpoint_version
        and prior.get("generation_signature") == generation_signature
        and isinstance(prior.get("results"), dict)
    ):
        results = prior["results"]
except (OSError, ValueError, TypeError):
    pass

def save_checkpoint():
    payload = {{
        "version": checkpoint_version,
        "generation_signature": generation_signature,
        "source_hash": source_hash,
        "expected": len(tasks),
        "completed": sum(
            1
            for task in tasks
            if results.get(str(task.get("task_key", task["index"])), {{}}).get("status") == "ok"
        ),
        "results": results,
    }}
    temporary = checkpoint_path + f".{{os.getpid()}}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    for attempt in range(6):
        try:
            os.replace(temporary, checkpoint_path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def inspect_wav(path):
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise RuntimeError("generated WAV has invalid stream metadata")
    duration = info.frames / info.samplerate
    if duration < 0.08:
        raise RuntimeError(f"generated WAV is implausibly short: {{duration:.3f}}s")
    return {{
        "wav_sha256": file_sha256(path),
        "wav_size": os.path.getsize(path),
        "wav_frames": int(info.frames),
        "wav_sample_rate": int(info.samplerate),
        "wav_channels": int(info.channels),
        "wav_duration_seconds": round(duration, 6),
    }}

def valid_result(result, task):
    if not isinstance(result, dict):
        return False
    key = str(task.get("task_key", task["index"]))
    if (
        result.get("status") != "ok"
        or result.get("task_key") != key
        or result.get("index") != task["index"]
        or result.get("chunk_index", 0) != task.get("chunk_index", 0)
        or result.get("fingerprint") != task["fingerprint"]
        or result.get("output_path") != task["output_path"]
    ):
        return False
    try:
        actual = inspect_wav(task["output_path"])
    except Exception:
        return False
    return all(result.get(name) == value for name, value in actual.items())

try:
    model = VoxCPM.from_pretrained({str(model_path)!r}, load_denoiser=False)
except Exception as exc:
    results["_model"] = {{"status": "error", "error": f"model load failed: {{exc}}"}}
    save_checkpoint()
    raise

prompt_caches = {{}}
supports_prompt_cache = bool(
    {reuse_reference_cache!r}
    and hasattr(model, "tts_model")
    and hasattr(model.tts_model, "build_prompt_cache")
    and hasattr(model.tts_model, "generate_with_prompt_cache")
)

def normalize_text(text):
    text = re.sub(r"\\s+", " ", str(text).replace("\\n", " ")).strip()
    if {normalize!r}:
        if getattr(model, "text_normalizer", None) is None:
            from voxcpm.utils.text_normalize import TextNormalizer
            model.text_normalizer = TextNormalizer()
        text = model.text_normalizer.normalize(text)
    return text

def generate(task, generation_attempt):
    variants = task.get("control_variants") or [task.get("style_control", "")]
    control = str(variants[min(generation_attempt - 1, len(variants) - 1)]).strip()
    text = str(task["text"]).replace("\\n", " ")
    final_text = f"({{control}}){{text}}" if control else text
    if not supports_prompt_cache:
        wav = model.generate(
            text=final_text,
            reference_wav_path=task["reference_audio"],
            cfg_value={cfg_value!r},
            inference_timesteps={inference_timesteps!r},
            normalize={normalize!r},
            retry_badcase={retry_badcase!r},
            retry_badcase_max_times={retry_badcase_max_times!r},
        )
        return wav, {{"prompt_cache": False, "used_control": control}}
    reference = task["reference_audio"]
    if reference not in prompt_caches:
        prompt_caches[reference] = model.tts_model.build_prompt_cache(reference_wav_path=reference)
    wav, target_tokens, audio_features = model.tts_model.generate_with_prompt_cache(
        target_text=normalize_text(final_text),
        prompt_cache=prompt_caches[reference],
        cfg_value={cfg_value!r},
        inference_timesteps={inference_timesteps!r},
        retry_badcase={retry_badcase!r},
        retry_badcase_max_times={retry_badcase_max_times!r},
        retry_badcase_ratio_threshold={retry_badcase_ratio_threshold!r},
        max_len={max_len!r},
    )
    token_count = max(1, int(target_tokens.numel()))
    feature_count = int(audio_features.shape[0])
    audio_text_ratio = feature_count / token_count
    if {retry_badcase!r} and audio_text_ratio >= {retry_badcase_ratio_threshold!r}:
        raise RuntimeError(
            f"badcase remained after retries: audio_text_ratio={{audio_text_ratio:.3f}}"
        )
    return wav.squeeze(0).cpu().numpy(), {{
        "prompt_cache": True,
        "target_token_count": token_count,
        "audio_feature_count": feature_count,
        "audio_text_ratio": round(audio_text_ratio, 6),
        "used_control": control,
    }}

save_checkpoint()
for position, task in enumerate(tasks, 1):
    key = str(task.get("task_key", task["index"]))
    if valid_result(results.get(key), task):
        print(
            f"VoxCPM [{{position}}/{{len(tasks)}}] key={{key}} status=cached",
            flush=True,
        )
        continue
    last_error = None
    for generation_attempt in range(1, {task_attempts!r} + 1):
        temporary_wav = task["output_path"] + f".{{os.getpid()}}.tmp.wav"
        try:
            wav, diagnostics = generate(task, generation_attempt)
            os.makedirs(os.path.dirname(task["output_path"]) or ".", exist_ok=True)
            sf.write(temporary_wav, wav, model.tts_model.sample_rate)
            wave_meta = inspect_wav(temporary_wav)
            duration = wave_meta["wav_duration_seconds"]
            minimum = float(task.get("min_duration_seconds", 0.0))
            maximum = float(task.get("max_duration_seconds", float("inf")))
            if duration < minimum:
                # Never repair a performance with post-generation time
                # stretching.  VoxCPM sampling is stochastic, so discard this
                # take and let the normal attempt loop sample another one.  If
                # every take is anomalously fast, fail with a resumable
                # checkpoint instead of silently changing the actor's timing.
                raise RuntimeError(
                    "audio is anomalously fast; retrying a fresh VoxCPM take: "
                    f"{{duration:.3f}}s < {{minimum:.3f}}s"
                )
            if duration > maximum:
                raise RuntimeError(
                    "audio is too slow or leaked control text: "
                    f"{{duration:.3f}}s > {{maximum:.3f}}s"
                )
            os.replace(temporary_wav, task["output_path"])
            results[key] = {{
                "task_key": key,
                "index": task["index"],
                "chunk_index": task.get("chunk_index", 0),
                "status": "ok",
                "fingerprint": task["fingerprint"],
                "output_path": task["output_path"],
                "generation_attempts": generation_attempt,
                **diagnostics,
                **wave_meta,
            }}
            break
        except Exception as exc:
            last_error = exc
            try:
                os.unlink(temporary_wav)
            except FileNotFoundError:
                pass
            if generation_attempt < {task_attempts!r}:
                time.sleep(min(10.0, 1.5 * generation_attempt))
    if results.get(key, {{}}).get("status") != "ok":
        results[key] = {{
            "task_key": key,
            "index": task["index"],
            "chunk_index": task.get("chunk_index", 0),
            "status": "error",
            "fingerprint": task["fingerprint"],
            "output_path": task["output_path"],
            "generation_attempts": {task_attempts!r},
            "error": str(last_error),
        }}
    save_checkpoint()
    print(
        f"VoxCPM [{{position}}/{{len(tasks)}}] key={{key}} status={{results[key]['status']}}",
        flush=True,
    )

if any(
    results.get(str(task.get("task_key", task["index"])), {{}}).get("status") != "ok"
    for task in tasks
):
    raise SystemExit(1)
'''


def run_voxcpm_tasks(tasks: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    for task in tasks:
        reference = Path(task["reference_audio"])
        if not nonempty_file(reference):
            raise PipelineError(f"VoxCPM reference audio is missing or empty: {reference}")

    out_dir = output_dir(config)
    script_path = out_dir / "_batch_voxcpm.py"
    results_path = out_dir / "voxcpm_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(create_voxcpm_script(tasks, config), encoding="utf-8")
    try:
        run_checked_subprocess(
            [str(resolve_voxcpm_python(config)), str(script_path)],
            timeout=int(config.get("voxcpm", {}).get("timeout", 7200)),
        )
    finally:
        script_path.unlink(missing_ok=True)

    checkpoint = read_json(results_path, {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    raw_results = checkpoint.get("results", {})
    if (
        checkpoint.get("version") != VOXCPM_BATCH_CHECKPOINT_VERSION
        or checkpoint.get("generation_signature") != voxcpm_generation_signature(config)
        or not isinstance(raw_results, dict)
    ):
        raw_results = {}
    task_by_key = {str(task.get("task_key", task["index"])): task for task in tasks}
    results = [raw_results.get(key, {}) for key in task_by_key]
    failures = []
    for key, item in zip(task_by_key, results):
        task = task_by_key[key]
        if not valid_voxcpm_result(item, task):
            failures.append(item)
    if failures or len(results) != len(tasks):
        detail = failures[:3] or "missing results"
        raise PipelineError(f"VoxCPM batch failed: {detail}")
    return results


def recover_voxcpm_tasks(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover atomically checkpointed WAVs from an interrupted batch."""
    if not tasks:
        return [], []
    checkpoint = read_voxcpm_checkpoint(config)
    if checkpoint.get("version") != VOXCPM_BATCH_CHECKPOINT_VERSION:
        return [], list(tasks)
    signature = checkpoint.get("generation_signature")
    if signature is not None:
        if signature != voxcpm_generation_signature(config):
            return [], list(tasks)
    elif checkpoint.get("source_hash") != voxcpm_batch_source_hash(tasks, config):
        # Compatibility for an interrupted v4 batch written before stable
        # per-chunk generation signatures were added.
        return [], list(tasks)
    raw_results = checkpoint.get("results", {})
    if not isinstance(raw_results, dict):
        return [], list(tasks)
    recovered: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        result = raw_results.get(str(task.get("task_key", task["index"])), {})
        if valid_voxcpm_result(result, task):
            recovered.append(result)
        else:
            pending.append(task)
    return recovered, pending


def segments_for_dialogues(config: dict[str, Any], dialogues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    directory = output_dir(config) / "segments"
    return [
        {
            "audio_path": str(directory / f"{index:05d}.wav"),
            "chapter": dialogue.get("chapter", "unknown"),
            "order": index,
            "speaker": effective_speaker(dialogue.get("speaker", "")),
        }
        for index, dialogue in enumerate(dialogues)
    ]


def streaming_tts_checkpoint_path(config: dict[str, Any]) -> Path:
    configured = config.get("streaming_tts", {}).get("checkpoint_path")
    return resolve_path(configured) if configured else output_dir(config) / "streaming_tts.checkpoint.json"


def make_tts_task(
    config: dict[str, Any],
    index: int,
    dialogue: dict[str, Any],
    gender: str,
    emotion: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    speaker = effective_speaker(dialogue.get("speaker", ""))
    assignment = get_voice_assignment(speaker, gender, config)
    fingerprint = tts_fingerprint(dialogue, assignment, emotion, performance, config)
    path = output_dir(config) / "segments" / f"{index:05d}.wav"
    entry: dict[str, Any] = {
        "index": index,
        "speaker": speaker,
        "engine": assignment["engine"],
        "fingerprint": fingerprint,
        "audio_path": str(path),
    }
    task: dict[str, Any] = {
        "index": index,
        "text": dialogue["text"],
        "output_path": str(path),
        "fingerprint": fingerprint,
        "entry": entry,
    }
    if assignment["engine"] == "voxcpm":
        raw_control = str(performance.get("performance_control", "")).strip()
        if not raw_control:
            legacy = build_emotion_prefix(emotion.get("emotion"), emotion.get("tone"))
            raw_control = legacy[1:-1] if legacy.startswith("(") and legacy.endswith(")") else legacy
        options = voxcpm_generation_options(config)
        control = compact_performance_control(
            raw_control,
            speaker=speaker,
            emotion=str(emotion.get("emotion", "")),
            pace_hint=str(performance.get("pace", "")),
            max_chars=options["control_max_chars"],
        )
        chunks = split_tts_text(
            str(dialogue["text"]),
            max_chars=options["chunk_max_chars"],
            min_chars=options["chunk_min_chars"],
        )
        entry["style_control"] = control
        entry["chunk_count"] = len(chunks)
        task["reference_audio"] = assignment["reference_audio"]
        task["style_control"] = control
        task["chunks"] = []
        chunk_directory = output_dir(config) / "segments" / ".voxcpm_chunks" / f"{index:05d}"
        for chunk_index, chunk_text in enumerate(chunks):
            bounds = duration_quality_bounds(
                chunk_text,
                control,
                min_ratio=options["quality_min_duration_ratio"],
                max_ratio=options["quality_max_duration_ratio"],
            )
            chunk_payload = {
                "parent_fingerprint": fingerprint,
                "chunking_version": TTS_CHUNKING_VERSION,
                "control_version": TTS_CONTROL_VERSION,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "style_control": control,
                "control_variants": control_variants(control),
                "quality_bounds": bounds,
            }
            chunk_fingerprint = hashlib.sha256(
                json.dumps(chunk_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            task["chunks"].append(
                {
                    "task_key": f"{index:05d}:{chunk_index:03d}",
                    "index": index,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "output_path": str(chunk_directory / f"{chunk_index:03d}.wav"),
                    "fingerprint": chunk_fingerprint,
                    "reference_audio": assignment["reference_audio"],
                    "style_control": control,
                    "control_variants": chunk_payload["control_variants"],
                    **bounds,
                }
            )
    else:
        task["voice_id"] = assignment["voice_id"]
    return task


def voxcpm_chunk_tasks(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for task in tasks:
        raw_chunks = task.get("chunks", [])
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise PipelineError(f"VoxCPM task {task.get('index')} has no semantic chunks")
        chunks.extend(raw_chunks)
    return chunks


def assemble_voxcpm_task(task: dict[str, Any], config: dict[str, Any]) -> None:
    """Atomically join checkpointed semantic chunks into one dialogue WAV."""
    chunks = task.get("chunks", [])
    if not chunks:
        raise PipelineError(f"VoxCPM task {task.get('index')} has no chunks to assemble")
    pause_ms = voxcpm_generation_options(config)["inter_chunk_pause_ms"]
    combined = AudioSegment.empty()
    for position, chunk in enumerate(chunks):
        chunk_path = Path(chunk["output_path"])
        if not inspect_generated_wav(chunk_path):
            raise PipelineError(f"VoxCPM chunk is missing or invalid: {chunk_path}")
        try:
            combined += AudioSegment.from_wav(chunk_path)
        except Exception as exc:
            raise PipelineError(f"Unable to read VoxCPM chunk {chunk_path}: {exc}") from exc
        if position + 1 < len(chunks) and pause_ms:
            combined += AudioSegment.silent(duration=pause_ms, frame_rate=combined.frame_rate)
    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.wav")
    try:
        combined.export(temporary, format="wav")
        if not inspect_generated_wav(temporary):
            raise PipelineError(f"Assembled VoxCPM WAV is invalid: {temporary}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def run_voxcpm_composite_tasks(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resume/generate per-chunk audio, then assemble completed dialogue files."""
    if not tasks:
        return []
    chunks = voxcpm_chunk_tasks(tasks)
    recovered, pending = recover_voxcpm_tasks(chunks, config)
    generated: list[dict[str, Any]] = []
    if pending:
        generated = run_voxcpm_tasks(pending, config)
    completed_keys = {
        str(item.get("task_key"))
        for item in recovered
        if isinstance(item, dict) and item.get("status") == "ok"
    }
    for chunk, item in zip(pending, generated):
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        if item.get("task_key") is not None:
            completed_keys.add(str(item["task_key"]))
        elif item.get("index") == chunk["index"]:
            # Keeps the low-level boundary easy to fake in unit tests and for
            # older local runners; the real child always returns task_key.
            completed_keys.add(chunk["task_key"])
    missing = [chunk for chunk in chunks if chunk["task_key"] not in completed_keys]
    if missing:
        missing_keys = [chunk["task_key"] for chunk in missing[:5]]
        raise PipelineError(f"VoxCPM chunks remain incomplete: {missing_keys}")
    results: list[dict[str, Any]] = []
    for task in tasks:
        expected_keys = {chunk["task_key"] for chunk in task["chunks"]}
        if not expected_keys.issubset(completed_keys):
            raise PipelineError(f"VoxCPM task {task['index']} has incomplete chunks")
        assemble_voxcpm_task(task, config)
        results.append(
            {
                "index": task["index"],
                "status": "ok",
                "fingerprint": task["fingerprint"],
                "output_path": task["output_path"],
                "chunk_count": len(task["chunks"]),
            }
        )
    return results


def completed_tts_entry(task: dict[str, Any]) -> dict[str, Any]:
    wav = inspect_generated_wav(Path(task["output_path"]))
    if not wav:
        raise PipelineError(f"Generated TTS WAV is invalid: {task['output_path']}")
    return {**task["entry"], "wav": wav}


def reusable_tts_entry(entry: Any, task: dict[str, Any]) -> bool:
    if not isinstance(entry, dict) or entry.get("fingerprint") != task["fingerprint"]:
        return False
    actual = inspect_generated_wav(Path(task["output_path"]))
    return bool(actual and entry.get("wav") == actual)


def load_partial_performance_results(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Read only compatible, atomically checkpointed performance results."""
    from app.core.performance_director import (
        performance_direction_source_hash,
        validate_profile_payload,
    )

    cards = _performance_cards(config)
    performance_config = config.get("performance", {})
    merged: dict[str, Any] = {}
    all_complete = True
    groups = _performance_groups(config, dialogues, gender_results)
    if not groups:
        return {}, True
    for group in groups:
        group_dialogues = group.get("dialogues", dialogues)
        profile_payload = read_json(group["profile_path"], {})
        profile_problems = validate_profile_payload(
            profile_payload,
            group["speakers"],
            group_dialogues,
            novel_text,
            character_cards_text=cards,
        )
        if profile_problems:
            all_complete = False
            continue
        profiles = profile_payload.get("profiles", {})
        expected_hash = performance_direction_source_hash(
            novel_text,
            group["targets"],
            group_dialogues,
            profiles,
            emotion_results,
            cards,
            int(performance_config.get("context_radius", 100)),
            int(performance_config.get("min_control_chars", 18)),
            int(performance_config.get("max_control_chars", 140)),
        )
        checkpoint = read_json(group["direction_checkpoint"], {})
        if not isinstance(checkpoint, dict) or (
            checkpoint.get("source_hash") != expected_hash
            or checkpoint.get("target_indices") != group["targets"]
        ):
            all_complete = False
            continue
        raw_results = checkpoint.get("results", {})
        expected = {str(index) for index in group["targets"]}
        if not isinstance(raw_results, dict) or not set(raw_results).issubset(expected):
            all_complete = False
            continue
        results = {key: value for key, value in raw_results.items() if isinstance(value, dict)}
        if set(merged).intersection(results):
            raise PipelineError("Primary and supplemental performance checkpoints overlap")
        merged.update(results)
        group_complete = (
            set(results) == expected
            and not checkpoint.get("errors")
            and not checkpoint.get("inflight")
        )
        all_complete = all_complete and group_complete
    return merged, all_complete


def run_streaming_tts(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Consume partial performance checkpoints without blocking their producer."""
    settings = config.get("streaming_tts", {})
    poll_interval = max(0.05, float(settings.get("poll_interval_seconds", 15)))
    minimum_batch = max(1, int(settings.get("minimum_batch_size", 16)))
    maximum_batch = max(minimum_batch, int(settings.get("maximum_batch_size", 4096)))
    maximum_wait = max(poll_interval, float(settings.get("maximum_batch_wait_seconds", 120)))
    checkpoint_path = streaming_tts_checkpoint_path(config)
    payload = read_json(checkpoint_path, {})
    if (
        not isinstance(payload, dict)
        or payload.get("version") != STREAMING_TTS_CHECKPOINT_VERSION
        or payload.get("tts_pipeline_version") != TTS_PIPELINE_VERSION
    ):
        payload = {}
    entries = payload.get("segments", {}) if payload.get("total_segments") == len(dialogues) else {}
    if not isinstance(entries, dict):
        entries = {}
    entries = {
        key: value
        for key, value in entries.items()
        if key.isdigit() and 0 <= int(key) < len(dialogues)
    }
    active_batch = payload.get("active_batch", [])
    if not isinstance(active_batch, list):
        active_batch = []

    performance_targets = set(all_performance_target_indices(config, dialogues, gender_results))
    wait_started = time.monotonic()
    last_status: tuple[int, int, bool] | None = None

    def checkpoint() -> None:
        write_json(
            checkpoint_path,
            {
                "version": STREAMING_TTS_CHECKPOINT_VERSION,
                "tts_pipeline_version": TTS_PIPELINE_VERSION,
                "total_segments": len(dialogues),
                "segments": entries,
                "active_batch": active_batch,
            },
        )

    while True:
        if config.get("features", {}).get("performance_direction", False):
            performance_results, performance_complete = load_partial_performance_results(
                config,
                dialogues,
                novel_text,
                gender_results,
                emotion_results,
            )
        else:
            performance_results, performance_complete = {}, True

        tasks_by_index: dict[int, dict[str, Any]] = {}
        completed_count = 0
        for index, dialogue in enumerate(dialogues):
            if index in performance_targets and str(index) not in performance_results:
                continue
            speaker = str(dialogue.get("speaker", ""))
            gender = gender_results.get(speaker, {}).get("gender", "male")
            if gender not in {"male", "female"}:
                gender = "male"
            emotion = emotion_results.get(str(index), {})
            performance = performance_results.get(str(index), {})
            task = make_tts_task(config, index, dialogue, gender, emotion, performance)
            if reusable_tts_entry(entries.get(str(index)), task):
                completed_count += 1
            else:
                tasks_by_index[index] = task

        batch: list[dict[str, Any]] = []
        if active_batch:
            for descriptor in active_batch:
                if not isinstance(descriptor, dict):
                    batch = []
                    break
                task = tasks_by_index.get(descriptor.get("index"))
                if not task or task["fingerprint"] != descriptor.get("fingerprint"):
                    batch = []
                    break
                batch.append(task)
            if len(batch) != len(active_batch):
                active_batch = []
                checkpoint()

        candidates = [tasks_by_index[index] for index in sorted(tasks_by_index)]
        if not batch and candidates:
            waited = time.monotonic() - wait_started
            if len(candidates) >= minimum_batch or performance_complete or waited >= maximum_wait:
                batch = candidates[:maximum_batch]
                active_batch = [
                    {"index": task["index"], "fingerprint": task["fingerprint"]}
                    for task in batch
                ]
                checkpoint()

        if batch:
            if any(task["entry"]["engine"] != "voxcpm" for task in batch):
                raise PipelineError("Streaming TTS requires every ready task to use VoxCPM")
            try:
                results = run_voxcpm_composite_tasks(batch, config)
            except BaseException:
                checkpoint()
                raise
            successful = {item["index"] for item in results}
            for task in batch:
                if task["index"] in successful:
                    entries[str(task["index"])] = completed_tts_entry(task)
            active_batch = []
            checkpoint()
            wait_started = time.monotonic()
            continue

        status = (completed_count, len(performance_results), performance_complete)
        if status != last_status:
            print(
                "Streaming TTS: "
                f"audio={completed_count}/{len(dialogues)}, "
                f"performance={len(performance_results)}/{len(performance_targets)}, "
                f"producer_complete={performance_complete}",
                flush=True,
            )
            last_status = status
        if performance_complete and completed_count == len(dialogues):
            checkpoint()
            return segments_for_dialogues(config, dialogues)
        time.sleep(poll_interval)


def step_tts(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]],
    gender_results: dict[str, Any],
    emotion_results: dict[str, Any],
    performance_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    directory = output_dir(config) / "segments"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "segments_manifest.json"
    old_manifest = read_json(manifest_path, {})
    performance_results = performance_results or {}
    source_hash = tts_source_hash(
        config,
        dialogues,
        gender_results,
        emotion_results,
        performance_results,
    )
    old_entries = (
        old_manifest.get("segments", {})
        if isinstance(old_manifest, dict)
        and old_manifest.get("version") == TTS_PIPELINE_VERSION
        and old_manifest.get("source_hash") == source_hash
        else {}
    )
    streaming_manifest = read_json(streaming_tts_checkpoint_path(config), {})
    streaming_entries = (
        streaming_manifest.get("segments", {})
        if isinstance(streaming_manifest, dict)
        and streaming_manifest.get("version") == STREAMING_TTS_CHECKPOINT_VERSION
        and streaming_manifest.get("tts_pipeline_version") == TTS_PIPELINE_VERSION
        and streaming_manifest.get("total_segments") == len(dialogues)
        else {}
    )
    if not isinstance(streaming_entries, dict):
        streaming_entries = {}
    reusable_entries = {**streaming_entries, **old_entries}
    new_entries: dict[str, Any] = {}
    edge_tasks: list[dict[str, Any]] = []
    voxcpm_tasks: list[dict[str, Any]] = []

    for index, dialogue in enumerate(dialogues):
        raw_speaker = dialogue.get("speaker", "")
        speaker = effective_speaker(raw_speaker)
        gender = gender_results.get(speaker, gender_results.get(raw_speaker, {})).get("gender", "male")
        if gender not in {"male", "female"}:
            gender = "male"
        emotion = emotion_results.get(str(index), {})
        performance = performance_results.get(str(index), {})
        task = make_tts_task(config, index, dialogue, gender, emotion, performance)
        previous = reusable_entries.get(str(index), {})
        if reusable_tts_entry(previous, task):
            new_entries[str(index)] = completed_tts_entry(task)
            continue
        if task["entry"]["engine"] == "voxcpm":
            voxcpm_tasks.append(task)
        else:
            edge_tasks.append(task)

    def checkpoint() -> None:
        write_json(
            manifest_path,
            {
                "version": TTS_PIPELINE_VERSION,
                "source_hash": source_hash,
                "total_segments": len(dialogues),
                "segments": new_entries,
            },
        )

    print(
        f"  resume={len(new_entries)}, VoxCPM={len(voxcpm_tasks)}, "
        f"edge-tts={len(edge_tasks)}"
    )

    for task in edge_tasks:
        try:
            synthesize_edge_tts_to_wav(task["text"], task["voice_id"], Path(task["output_path"]))
        except Exception as exc:
            checkpoint()
            raise PipelineError(f"edge-tts failed for segment {task['index']}: {exc}") from exc
        new_entries[str(task["index"])] = completed_tts_entry(task)
        checkpoint()

    if voxcpm_tasks:
        try:
            results = run_voxcpm_composite_tasks(voxcpm_tasks, config)
        except BaseException:
            checkpoint()
            raise
        successful = {item["index"] for item in results}
        for task in voxcpm_tasks:
            if task["index"] in successful and nonempty_file(Path(task["output_path"])):
                new_entries[str(task["index"])] = completed_tts_entry(task)
        checkpoint()

    expected = set(map(str, range(len(dialogues))))
    missing = sorted(expected - set(new_entries), key=int)
    empty = [
        entry["audio_path"]
        for entry in new_entries.values()
        if entry.get("wav") != inspect_generated_wav(Path(entry["audio_path"]))
    ]
    if missing or empty:
        raise PipelineError(f"TTS output is incomplete; missing={missing[:10]}, empty={empty[:3]}")
    checkpoint()
    return segments_for_dialogues(config, dialogues)


def validate_tts_manifest(
    config: dict[str, Any],
    dialogues: list[dict[str, Any]] | None = None,
    gender_results: dict[str, Any] | None = None,
    emotion_results: dict[str, Any] | None = None,
    performance_results: dict[str, Any] | None = None,
) -> list[str]:
    directory = output_dir(config) / "segments"
    manifest = read_json(directory / "segments_manifest.json", {})
    entries = manifest.get("segments", {}) if isinstance(manifest, dict) else {}
    total = manifest.get("total_segments", 0) if isinstance(manifest, dict) else 0
    problems: list[str] = []
    if manifest.get("version") != TTS_PIPELINE_VERSION or not manifest.get("source_hash"):
        problems.append("segment manifest version or source hash is incompatible")
    if dialogues is not None:
        if total != len(dialogues):
            problems.append("segment manifest count does not match current dialogues")
        performance_required = config.get("features", {}).get("performance_direction", False)
        if (
            gender_results is None
            or emotion_results is None
            or (performance_required and performance_results is None)
        ):
            problems.append("current gender, emotion, and enabled performance results are required for TTS validation")
        else:
            expected_hash = tts_source_hash(
                config,
                dialogues,
                gender_results,
                emotion_results,
                performance_results or {},
            )
            if manifest.get("source_hash") != expected_hash:
                problems.append("segment manifest does not belong to current TTS inputs")
    if not total or set(entries) != set(map(str, range(total))):
        problems.append("segment manifest does not exactly cover its declared segment count")
    for entry in entries.values():
        path = Path(entry.get("audio_path", ""))
        actual = inspect_generated_wav(path)
        if not actual:
            problems.append(f"missing or invalid audio: {path}")
        elif entry.get("wav") != actual:
            problems.append(f"audio identity does not match manifest: {path}")
    return problems


def validate_current_tts_cache(config: dict[str, Any]) -> list[str]:
    try:
        dialogues, characters, novel_text = step_parse(config)
        gender_results = require_gender_results(characters, dialogues, novel_text)
        emotion_results = (
            require_emotion_results(dialogues, novel_text)
            if config.get("features", {}).get("emotion_label", True)
            else {}
        )
        performance_results = (
            require_performance_results(
                config,
                dialogues,
                novel_text,
                gender_results,
                emotion_results,
            )
            if config.get("features", {}).get("performance_direction", False)
            else {}
        )
    except (OSError, ValueError, PipelineError) as exc:
        return [str(exc)]
    return validate_tts_manifest(
        config,
        dialogues,
        gender_results,
        emotion_results,
        performance_results,
    )


def step_splice(config: dict[str, Any], segments: list[dict[str, Any]]) -> tuple[str, float]:
    missing = [item["audio_path"] for item in segments if not nonempty_file(Path(item["audio_path"]))]
    if missing:
        raise PipelineError(f"Cannot splice; {len(missing)} TTS segments are missing")
    path = speech_output_path(config)
    splicer = AudioSplicer(output_bitrate=str(config.get("output", {}).get("bitrate", "96k")))
    final_audio = splicer.splice(segments, output_path=str(path))
    if not nonempty_file(path):
        raise PipelineError(f"Splicer did not create {path}")
    return str(path), len(final_audio) / 1000.0


def step_bgm_segmentation(config: dict[str, Any]) -> list[dict[str, Any]]:
    novel_path = resolve_path(config["novel"]["text_path"])
    labels_path = resolve_path(config["novel"]["labels_path"])
    novel_text = novel_path.read_text(encoding="utf-8")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dialogues, _ = parse(novel_text, labels)
    total_lines = len(novel_text.splitlines())
    bgm_config = config.get("bgm", {})
    chunks = int(bgm_config.get("segmentation_chunks", 6))
    path = bgm_segments_path(config)

    if path.exists() and not bgm_config.get("force_segmentation", False):
        segments = load_segments(path)
        problems = validate_segments(segments, total_lines, 5, max(20, chunks * 20))
        source_hash = bgm_source_hash(novel_text)
        compatible = all(
            item.get("segmentation_model") == SENSENOVA_FLASH_LITE_MODEL
            and item.get("segmentation_pipeline_version") == BGM_SEGMENTATION_PIPELINE_VERSION
            and item.get("segmentation_source_hash") == source_hash
            for item in segments
        )
        if not problems and compatible:
            print(f"  using cached BGM segmentation: {len(segments)} segments")
            return segments

    client = build_flash_lite_client("bgm_segmentation")
    segments = segment_novel_chunked(novel_text, dialogues=dialogues, client=client, num_chunks=chunks)
    problems = validate_segments(segments, total_lines, 5, max(20, chunks * 20))
    if problems:
        raise PipelineError(f"BGM segmentation is invalid: {problems}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_segments(path, segments)
    return segments


def step_bgm_labeling(
    config: dict[str, Any], segments: list[dict[str, Any]], novel_text: str = ""
) -> list[dict[str, Any]]:
    bgm_config = config.get("bgm", {})
    if not novel_text:
        novel_text = resolve_path(config["novel"]["text_path"]).read_text(encoding="utf-8")
    source_hash = bgm_source_hash(novel_text)
    if not bgm_config.get("force_label", False) and all(
        item.get("bgm_type")
        and item.get("bgm_model") == SENSENOVA_FLASH_LITE_MODEL
        and item.get("bgm_pipeline_version") == BGM_TYPE_PIPELINE_VERSION
        and item.get("bgm_source_hash") == source_hash
        for item in segments
    ):
        print("  using cached BGM labels")
        return segments
    client = build_flash_lite_client("bgm_classification")
    labeled = label_bgm_types(segments, client=client, novel_text=novel_text)
    if not all(item.get("bgm_type") for item in labeled):
        raise PipelineError("BGM labeling returned unlabeled segments")
    save_segments(bgm_segments_path(config), labeled)
    return labeled


def validate_bgm_cache(
    segments: list[dict[str, Any]],
    manifest_path: Path,
    bgm_dir: Path,
    clips_per_segment: int,
    *,
    clip_duration: float = 30,
    inference_steps: int = 8,
    model: str = ACE_STEP_MODEL,
    lm_model: str | None = None,
    thinking: bool | None = None,
    guidance_scale: float | None = None,
    require_quality_validation: bool = False,
) -> list[str]:
    """Validate manifest identity and every clip for the current segment set."""
    problems: list[str] = []
    manifest = read_json(manifest_path, {})
    segment_map = manifest.get("segments", {}) if isinstance(manifest, dict) else {}
    expected: dict[str, dict[str, Any]] = {}
    for position, segment in enumerate(segments, 1):
        index = segment.get("segment_index", position)
        expected[str(index)] = {
            "index": index,
            "bgm_type": segment.get("bgm_type", "unknown"),
            "prompt": build_segment_bgm_prompt(segment),
        }

    if not expected:
        return ["current BGM segment list is empty"]
    if set(segment_map) != set(expected):
        problems.append("manifest segment IDs do not exactly match current segments")
    if manifest.get("total_segments") != len(expected):
        problems.append("manifest total_segments does not match current segments")
    if manifest.get("clips_per_segment") != clips_per_segment:
        problems.append("manifest clips_per_segment does not match configuration")
    if manifest.get("generation_version") != BGM_GENERATION_VERSION:
        problems.append("manifest generation version is incompatible")
    if manifest.get("model") != model:
        problems.append("manifest ACE-Step model does not match")
    if thinking is not None and manifest.get("lm_model") != (lm_model if thinking else None):
        problems.append("manifest ACE-Step LM model does not match")
    if thinking is not None and manifest.get("thinking") != thinking:
        problems.append("manifest ACE-Step thinking mode does not match")
    if guidance_scale is not None and manifest.get("guidance_scale") != guidance_scale:
        problems.append("manifest ACE-Step guidance scale does not match")
    if manifest.get("duration_per_segment") != clip_duration:
        problems.append("manifest clip duration does not match configuration")
    if manifest.get("inference_steps") != inference_steps:
        problems.append("manifest inference steps do not match configuration")

    for key, current in expected.items():
        entry = segment_map.get(key, {})
        if entry.get("bgm_type") != current["bgm_type"]:
            problems.append(f"segment {key} bgm_type does not match")
        if entry.get("prompt") != current["prompt"]:
            problems.append(f"segment {key} prompt does not match")
        clips = entry.get("clips", []) if isinstance(entry, dict) else []
        by_index = {clip.get("clip_index"): clip for clip in clips if isinstance(clip, dict)}
        if set(by_index) != set(range(clips_per_segment)):
            problems.append(f"segment {key} clip indexes are incomplete")
            continue
        for clip_index, metadata in by_index.items():
            expected_name = f"{current['index']:03d}_{clip_index}.mp3"
            if Path(metadata.get("file", "")).name != expected_name:
                problems.append(f"segment {key} clip {clip_index} filename does not match")
            if not nonempty_file(bgm_dir / expected_name):
                problems.append(f"missing BGM clip: {expected_name}")
            elif require_quality_validation:
                if not metadata.get("quality_validated"):
                    problems.append(f"segment {key} clip {clip_index} lacks quality validation")
                else:
                    try:
                        actual_duration = media_duration_seconds(bgm_dir / expected_name)
                    except Exception:
                        actual_duration = 0.0
                    if abs(actual_duration - clip_duration) > max(2.0, clip_duration * 0.04):
                        problems.append(
                            f"segment {key} clip {clip_index} duration does not match"
                        )
            expected_seed = build_bgm_seed(current["index"], clip_index)
            if metadata.get("base_seed", metadata.get("seed")) != expected_seed:
                problems.append(f"segment {key} clip {clip_index} seed does not match")
    return problems


def step_bgm_generation(config: dict[str, Any]) -> str:
    bgm_config = config.get("bgm", {})
    bgm_dir = output_dir(config) / "bgm"
    manifest_path = bgm_dir / "bgm_manifest.json"
    segments = load_segments(bgm_segments_path(config))
    clips_per_segment = int(bgm_config.get("clips_per_segment", 3))
    clip_duration = float(bgm_config.get("clip_duration", 30))
    inference_steps = int(bgm_config.get("inference_steps", 8))
    model = str(bgm_config.get("model", ACE_STEP_MODEL))
    lm_model = str(bgm_config.get("lm_model", ACE_STEP_LM_MODEL))
    thinking = bool(bgm_config.get("thinking", False))
    guidance_scale = float(bgm_config.get("guidance_scale", 7.0))
    force = bool(bgm_config.get("force_generate", False))
    if not force:
        problems = validate_bgm_cache(
            segments,
            manifest_path,
            bgm_dir,
            clips_per_segment,
            clip_duration=clip_duration,
            inference_steps=inference_steps,
            model=model,
            lm_model=lm_model,
            thinking=thinking,
            guidance_scale=guidance_scale,
            require_quality_validation=True,
        )
        if not problems:
            print(f"  using exact BGM cache: {len(segments) * clips_per_segment} clips")
            return str(manifest_path)

    python_path = resolve_path(bgm_config.get("python", "ACE-Step-1.5/.venv/Scripts/python.exe"))
    script_path = ROOT / "scripts" / "run_bgm_generate.py"
    if not python_path.is_file():
        raise PipelineError(f"ACE-Step Python interpreter not found: {python_path}")
    if not script_path.is_file():
        raise PipelineError(f"BGM generator script not found: {script_path}")
    command = [
        str(python_path),
        str(script_path),
        "--segments",
        str(bgm_segments_path(config)),
        "--output-dir",
        str(bgm_dir),
        "--duration",
        str(clip_duration),
        "--inference-steps",
        str(inference_steps),
        "--clips-per-segment",
        str(clips_per_segment),
        "--model",
        model,
        "--lm-model",
        lm_model,
        "--lm-backend",
        str(bgm_config.get("lm_backend", "vllm")),
        "--cpu-offload" if bool(bgm_config.get("cpu_offload", True)) else "--no-cpu-offload",
        "--guidance-scale",
        str(guidance_scale),
        "--clip-attempts",
        str(int(bgm_config.get("clip_attempts", 3))),
        "--process-clip-limit",
        str(int(bgm_config.get("process_clip_limit", 16))),
        "--proxy",
        str(bgm_config.get("proxy", "http://127.0.0.1:7890")),
    ]
    if not thinking:
        command.append("--no-thinking")
    if force:
        command.append("--force")
    run_resumable_bgm_subprocess(
        command,
        timeout=int(bgm_config.get("generation_timeout", 7200)),
        manifest_path=manifest_path,
        max_restarts=int(bgm_config.get("process_max_restarts", 64)),
        native_no_progress_limit=int(bgm_config.get("native_no_progress_limit", 3)),
    )
    problems = validate_bgm_cache(
        segments,
        manifest_path,
        bgm_dir,
        clips_per_segment,
        clip_duration=clip_duration,
        inference_steps=inference_steps,
        model=model,
        lm_model=lm_model,
        thinking=thinking,
        guidance_scale=guidance_scale,
        require_quality_validation=True,
    )
    if problems:
        raise PipelineError(f"ACE-Step reported success but BGM artifacts are invalid: {problems[:5]}")
    return str(manifest_path)


def dependencies_are_older(output: Path, dependencies: Iterable[Path]) -> bool:
    if not nonempty_file(output):
        return False
    output_time = output.stat().st_mtime_ns
    return all(not dependency.exists() or dependency.stat().st_mtime_ns <= output_time for dependency in dependencies)


def step_bgm_mixing(config: dict[str, Any]) -> str:
    bgm_dir = output_dir(config) / "bgm"
    manifest = bgm_dir / "bgm_manifest.json"
    segments = bgm_segments_path(config)
    speech = speech_output_path(config)
    output = mixed_audio_path(config)
    dependencies = [manifest, segments, speech]
    missing = [str(path) for path in dependencies if not nonempty_file(path)]
    if missing:
        raise PipelineError(f"Cannot mix BGM; missing inputs: {missing}")
    tts_problems = validate_current_tts_cache(config)
    if tts_problems:
        raise PipelineError(f"Cannot mix BGM with stale TTS inputs: {tts_problems[:5]}")
    cached_duration_matches = (
        nonempty_file(output)
        and abs(media_duration_seconds(output) - media_duration_seconds(speech)) <= 1.0
    )
    if (
        not config.get("bgm", {}).get("force_mix", False)
        and dependencies_are_older(output, dependencies)
        and cached_duration_matches
    ):
        print(f"  using cached BGM mix: {output}")
        return str(output)
    mix_bgm(
        speech_path=speech,
        bgm_dir=bgm_dir,
        manifest_path=manifest,
        segments_path=segments,
        output_path=output,
        config_path=Path(config["_config_path"]),
    )
    if (
        not nonempty_file(output)
        or not dependencies_are_older(output, dependencies)
        or abs(media_duration_seconds(output) - media_duration_seconds(speech)) > 1.0
    ):
        raise PipelineError(f"BGM mixer did not create a fresh output: {output}")
    return str(output)


def load_illustration_plan(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, None)
    if isinstance(payload, dict):
        payload = payload.get("illustrations")
    return payload if isinstance(payload, list) else []


def validate_illustration_plan(path: Path) -> list[str]:
    plan = load_illustration_plan(path)
    if not plan:
        return ["illustration plan is missing or empty"]
    problems = []
    for index, item in enumerate(plan):
        if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
            problems.append(f"illustration {index + 1} has no prompt")
    return problems


def validate_visual_prompt_audit(
    plan_path: Path,
    checkpoint_path: Path,
    *,
    novel_path: Path,
    character_cards: Path,
    enabled: bool = True,
) -> list[str]:
    if not enabled:
        return []
    plan = load_illustration_plan(plan_path)
    if not plan:
        return ["illustration plan is missing or empty"]
    from app.core.visual_prompt_auditor import (
        VISUAL_PROMPT_PIPELINE_VERSION,
        visual_prompt_source_hash,
    )

    audit = read_json(checkpoint_path, None)
    if not isinstance(audit, dict):
        return ["visual prompt audit checkpoint is missing or invalid"]
    novel_text = novel_path.read_text(encoding="utf-8") if novel_path.is_file() else ""
    cards_text = character_cards.read_text(encoding="utf-8") if character_cards.is_file() else ""
    expected_hash = visual_prompt_source_hash(novel_text, plan, cards_text)
    results = audit.get("results")
    if (
        audit.get("pipeline_version") != VISUAL_PROMPT_PIPELINE_VERSION
        or audit.get("model") != SENSENOVA_FLASH_LITE_MODEL
        or audit.get("total_items") != len(plan)
        or audit.get("completed_indices") != list(range(len(plan)))
        or audit.get("errors")
        or audit.get("source_hash") != expected_hash
        or not isinstance(results, list)
        or len(results) != len(plan)
    ):
        return ["visual prompt audit checkpoint is incomplete or incompatible"]
    return []


def validate_illustrations(
    plan_path: Path,
    directory: Path,
    checkpoint_path: Path,
    prompt_audit_checkpoint: Path,
    *,
    expected_provider: str = "agnes",
    expected_model: str,
    expected_size: str,
    expected_endpoint: str,
    expected_generation_settings: dict[str, Any] | None = None,
    prompt_audit_enabled: bool = True,
    novel_path: Path,
    character_cards: Path,
    composition_suffix: str = "",
) -> list[str]:
    plan = load_illustration_plan(plan_path)
    if not plan:
        return ["illustration plan is missing or empty"]
    from scripts.generate_illustrations import (
        CHECKPOINT_VERSION,
        apply_audited_prompts,
        generation_prompt_hash,
        generation_source_hash,
        prompt_for_target,
    )

    problems = []
    for index in range(1, len(plan) + 1):
        candidates = list(directory.glob(f"{index:04d}_*.png"))
        if len([path for path in candidates if nonempty_file(path)]) != 1:
            problems.append(f"illustration {index:04d} is missing, empty, or ambiguous")
    checkpoint = read_json(checkpoint_path, None)
    if not isinstance(checkpoint, dict):
        problems.append("illustration checkpoint is missing or invalid")
    else:
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            problems.append(
                f"illustration checkpoint is not version {CHECKPOINT_VERSION}"
            )
        if checkpoint.get("provider") != expected_provider:
            problems.append("illustration checkpoint provider does not match configuration")
        if checkpoint.get("model") != expected_model:
            problems.append("illustration checkpoint model does not match configuration")
        if checkpoint.get("size") != expected_size:
            problems.append("illustration checkpoint size does not match configuration")
        if checkpoint.get("endpoint") != expected_endpoint.rstrip("/"):
            problems.append("illustration checkpoint endpoint does not match configuration")
        if (
            expected_generation_settings is not None
            and checkpoint.get("generation_settings") != expected_generation_settings
        ):
            problems.append("illustration checkpoint generation settings do not match configuration")
        records = checkpoint.get("images")
        if not isinstance(records, list) or len(records) != len(plan):
            problems.append("illustration checkpoint does not exactly cover the current plan")
        elif any(
            record.get("index") != index or record.get("status") != "success"
            for index, record in enumerate(records)
            if isinstance(record, dict)
        ) or not all(isinstance(record, dict) for record in records):
            problems.append("illustration checkpoint contains incomplete or misordered records")

    expected_prompt_plan = plan
    expected_audit_source_hash: str | None = None
    if prompt_audit_enabled:
        from app.core.visual_prompt_auditor import (
            VISUAL_PROMPT_PIPELINE_VERSION,
            visual_prompt_source_hash,
        )

        audit = read_json(prompt_audit_checkpoint, None)
        if not isinstance(audit, dict):
            problems.append("visual prompt audit checkpoint is missing or invalid")
        else:
            novel_text = novel_path.read_text(encoding="utf-8") if novel_path.is_file() else ""
            cards_text = character_cards.read_text(encoding="utf-8") if character_cards.is_file() else ""
            expected_audit_hash = visual_prompt_source_hash(novel_text, plan, cards_text)
            expected_audit_source_hash = expected_audit_hash
            if (
                audit.get("pipeline_version") != VISUAL_PROMPT_PIPELINE_VERSION
                or audit.get("model") != SENSENOVA_FLASH_LITE_MODEL
                or audit.get("total_items") != len(plan)
                or audit.get("completed_indices") != list(range(len(plan)))
                or audit.get("errors")
                or audit.get("source_hash") != expected_audit_hash
            ):
                problems.append("visual prompt audit checkpoint is incomplete or incompatible")
            else:
                expected_prompt_plan = apply_audited_prompts(plan, audit.get("results", []))

    if isinstance(checkpoint, dict):
        if checkpoint.get("source_hash") != generation_source_hash(plan):
            problems.append("illustration checkpoint does not belong to the current plan")
        if checkpoint.get("audit_source_hash") != expected_audit_source_hash:
            problems.append("illustration checkpoint audit source does not match")
        records = checkpoint.get("images")
        if isinstance(records, list) and len(records) == len(expected_prompt_plan):
            for index, (record, item) in enumerate(zip(records, expected_prompt_plan)):
                if not isinstance(record, dict):
                    continue
                prompt = prompt_for_target(str(item.get("prompt", "")), composition_suffix)
                if record.get("prompt_hash") != generation_prompt_hash(prompt):
                    problems.append(f"illustration {index + 1:04d} prompt fingerprint does not match")
                    break
    return problems


BGM_PROCESS_RESTART_EXIT_CODE = 75
WINDOWS_ACCESS_VIOLATION_EXIT_CODE = 0xC0000005


def run_checked_subprocess(
    command: list[str],
    timeout: int,
    *,
    allowed_returncodes: set[int] | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_tail: deque[str] = deque(maxlen=40)
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                line = output_queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            if line is None:
                break
            console_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            console_line = line.encode(console_encoding, errors="replace").decode(console_encoding)
            print(console_line, end="", flush=True)
            output_tail.append(line)
        returncode = process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise

    if returncode != 0 and returncode not in (allowed_returncodes or set()):
        detail = "".join(output_tail).strip() or "no error output"
        raise PipelineError(f"Subprocess failed ({returncode}): {detail[-2000:]}")
    return returncode


def _bgm_checkpoint_clip_count(manifest_path: Path) -> int:
    manifest = read_json(manifest_path, {})
    segments = manifest.get("segments", {}) if isinstance(manifest, dict) else {}
    if not isinstance(segments, dict):
        return 0
    return sum(
        len(entry.get("clips", []))
        for entry in segments.values()
        if isinstance(entry, dict) and isinstance(entry.get("clips", []), list)
    )


def run_resumable_bgm_subprocess(
    command: list[str],
    timeout: int,
    manifest_path: Path,
    *,
    max_restarts: int = 64,
    native_no_progress_limit: int = 3,
) -> None:
    """Restart ACE-Step after planned batches or native access violations."""
    progress_before = _bgm_checkpoint_clip_count(manifest_path)
    native_no_progress = 0
    restart_codes = {
        BGM_PROCESS_RESTART_EXIT_CODE,
        WINDOWS_ACCESS_VIOLATION_EXIT_CODE,
    }
    active_command = list(command)

    for restart_index in range(max_restarts + 1):
        returncode = run_checked_subprocess(
            active_command,
            timeout,
            allowed_returncodes=restart_codes,
        )
        if returncode == 0:
            return

        progress_after = _bgm_checkpoint_clip_count(manifest_path)
        made_progress = progress_after > progress_before
        if returncode == BGM_PROCESS_RESTART_EXIT_CODE and not made_progress:
            raise PipelineError(
                "ACE-Step requested a planned restart without checkpointing a new BGM clip"
            )

        if returncode == WINDOWS_ACCESS_VIOLATION_EXIT_CODE and not made_progress:
            native_no_progress += 1
            if native_no_progress >= native_no_progress_limit:
                raise PipelineError(
                    "ACE-Step repeatedly crashed with Windows access violation "
                    f"without checkpoint progress ({native_no_progress} attempts)"
                )
        else:
            native_no_progress = 0

        if restart_index >= max_restarts:
            raise PipelineError(
                f"ACE-Step exceeded the BGM process restart limit ({max_restarts})"
            )

        reason = (
            "planned process refresh"
            if returncode == BGM_PROCESS_RESTART_EXIT_CODE
            else "Windows access violation 0xC0000005"
        )
        print(
            f"  ACE-Step {reason}; checkpoint advanced "
            f"{progress_before} -> {progress_after}. Restarting cleanly "
            f"({restart_index + 1}/{max_restarts})...",
            flush=True,
        )
        # ``--force`` applies to the first process only. Once that process has
        # checkpointed progress, later clean processes must reuse those clips
        # or a planned refresh would repeatedly regenerate the first batch.
        if made_progress and "--force" in active_command:
            active_command = [argument for argument in active_command if argument != "--force"]
        progress_before = progress_after


def step_illustration_plan(config: dict[str, Any]) -> str:
    plan_path = illustration_plan_path(config)
    if not config.get("illustrations", {}).get("force_plan", False):
        if not validate_illustration_plan(plan_path):
            print(f"  using cached illustration plan: {plan_path}")
            return str(plan_path)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_illustration_plan.py"),
        "--novel",
        str(resolve_path(config["novel"]["text_path"])),
        "--labels",
        str(resolve_path(config["novel"]["labels_path"])),
        "--output",
        str(plan_path),
        "--resume",
    ]
    run_checked_subprocess(command, int(config.get("illustrations", {}).get("planning_timeout", 7200)))
    problems = validate_illustration_plan(plan_path)
    if problems:
        raise PipelineError(f"Illustration planner reported success but output is invalid: {problems[:5]}")
    return str(plan_path)


def step_illustrations(config: dict[str, Any]) -> str:
    plan_path = illustration_plan_path(config)
    illustration_config = config.get("illustrations", {})
    variants = illustration_variant_specs(config)
    portrait = variants[0]
    directory = portrait["directory"]
    checkpoint = portrait["checkpoint"]
    audit_checkpoint = visual_prompt_checkpoint_path(config)
    provider = str(illustration_config.get("provider", "agnes"))
    settings = portrait["settings"]
    model = str(settings["model"])
    size = str(settings["size"])
    if not config.get("illustrations", {}).get("force_generate", False):
        cached_problems = [
            validate_illustrations(
                plan_path,
                variant["directory"],
                variant["checkpoint"],
                audit_checkpoint,
                **illustration_validation_options(
                    config,
                    settings=variant["settings"],
                    composition_suffix=variant["composition_suffix"],
                ),
            )
            for variant in variants
        ]
        if all(not problems for problems in cached_problems):
            print(
                "  using cached illustration variants: "
                + ", ".join(str(variant["directory"]) for variant in variants)
            )
            return str(directory)
    command = [
        sys.executable,
        str(ROOT / "scripts/generate_illustrations.py"),
        "--plan", str(plan_path),
        "--novel", str(resolve_path(config["novel"]["text_path"])),
        "--character-cards", str(character_cards_path(config)),
        "--output-dir", str(directory),
        "--checkpoint", str(checkpoint),
        "--prompt-audit-checkpoint", str(audit_checkpoint),
        "--provider", provider,
        "--endpoint", str(settings["endpoint"]),
        "--model", model,
        "--size", size,
        "--composition-suffix", str(portrait["composition_suffix"]),
        "--timeout", str(illustration_config.get("request_timeout", 900.0)),
        "--max-attempts", str(illustration_config.get("max_attempts", 5)),
        "--interval-min", str(illustration_config.get("interval_min", 1.0)),
        "--interval-max", str(illustration_config.get("interval_max", 2.0)),
    ]
    if provider == "local-http":
        command.extend([
            "--steps", str(settings["steps"]),
            "--cfg", str(settings["cfg"]),
            "--negative-prompt", str(settings["negative_prompt"]),
            "--seed", str(settings["seed"]),
        ])
    if len(variants) > 1:
        landscape = variants[1]
        command.extend(
            [
                "--landscape-output-dir", str(landscape["directory"]),
                "--landscape-checkpoint", str(landscape["checkpoint"]),
                "--landscape-size", str(landscape["settings"]["size"]),
                "--landscape-composition-suffix", str(landscape["composition_suffix"]),
            ]
        )
    if not illustration_config.get("force_generate", False):
        command.append("--resume")
    if illustration_config.get("force_prompt_audit", False):
        command.append("--force-prompt-audit")
    if not illustration_config.get("prompt_audit_enabled", True):
        command.append("--skip-prompt-audit")
    run_checked_subprocess(command, int(config.get("illustrations", {}).get("generation_timeout", 86400)))
    problems = []
    for variant in variants:
        variant_problems = validate_illustrations(
            plan_path,
            variant["directory"],
            variant["checkpoint"],
            audit_checkpoint,
            **illustration_validation_options(
                config,
                settings=variant["settings"],
                composition_suffix=variant["composition_suffix"],
            ),
        )
        problems.extend(f"{variant['name']}: {problem}" for problem in variant_problems)
    if problems:
        raise PipelineError(f"Illustration generator reported success but output is invalid: {problems[:5]}")
    return str(directory)


def step_video(config: dict[str, Any]) -> str:
    plan_path = illustration_plan_path(config)
    audio = mixed_audio_path(config)
    video_config = config.get("video", {})
    ffmpeg_bin = str(video_config.get("ffmpeg", os.environ.get("FFMPEG_BIN", "ffmpeg")))
    ffprobe_bin = str(video_config.get("ffprobe", os.environ.get("FFPROBE_BIN", "ffprobe")))
    use_h3 = h3_video_enabled(config)
    if not nonempty_file(audio):
        raise PipelineError(f"Cannot generate video; mixed audio is missing: {audio}")
    outputs = []
    for variant in video_variant_specs(config):
        illustration = variant["illustrations"]
        directory = illustration["directory"]
        checkpoint = illustration["checkpoint"]
        settings = illustration["settings"]
        video = variant["output"]
        subtitle = variant["subtitle"]
        options = variant["options"]
        h3_spec = h3_variant_spec(config, variant) if use_h3 else None
        native_h3 = (
            h3_spec is not None
            and h3_spec["mode"] in {"native-chain", "continuous-chain"}
        )
        image_files = [] if native_h3 else sorted(directory.glob("*.png"))
        dependencies = [plan_path, audio, *image_files]
        if native_h3:
            audit_path = visual_prompt_checkpoint_path(config)
            illustration_problems = validate_visual_prompt_audit(
                plan_path,
                audit_path,
                novel_path=resolve_path(config["novel"]["text_path"]),
                character_cards=character_cards_path(config),
                enabled=config.get("illustrations", {}).get("prompt_audit_enabled", True),
            )
            dependencies.append(bgm_segments_path(config))
            if config.get("illustrations", {}).get("prompt_audit_enabled", True):
                dependencies.append(audit_path)
        else:
            illustration_problems = validate_illustrations(
                plan_path,
                directory,
                checkpoint,
                visual_prompt_checkpoint_path(config),
                **illustration_validation_options(
                    config,
                    settings=settings,
                    composition_suffix=illustration["composition_suffix"],
                ),
            )
        if illustration_problems:
            raise PipelineError(
                f"Cannot generate {variant['name']} video: {illustration_problems[:5]}"
            )
        if use_h3:
            assert h3_spec is not None
            h3_command = [
                sys.executable,
                str(
                    ROOT
                    / (
                        "scripts/generate_h3_native_clips.py"
                        if h3_spec["mode"] in {"native-chain", "continuous-chain"}
                        else "scripts/generate_h3_clips.py"
                    )
                ),
                "--plan", str(plan_path),
                "--novel", str(resolve_path(config["novel"]["text_path"])),
                "--labels", str(resolve_path(config["novel"]["labels_path"])),
                "--segments-dir", str(output_dir(config) / "segments"),
                "--audio", str(audio),
                "--output-dir", str(h3_spec["clips_dir"]),
                "--checkpoint", str(h3_spec["checkpoint"]),
                "--endpoint", str(h3_spec["endpoint"]),
                "--width", str(h3_spec["width"]),
                "--height", str(h3_spec["height"]),
                "--minimum-duration", str(h3_spec["minimum_duration"]),
                "--maximum-duration", str(h3_spec["maximum_duration"]),
                "--request-timeout", str(h3_spec["request_timeout"]),
                "--poll-seconds", str(h3_spec["poll_seconds"]),
                "--job-timeout", str(h3_spec["job_timeout"]),
                "--max-attempts", str(h3_spec["max_attempts"]),
                "--max-freeze-ratio", str(h3_spec["max_freeze_ratio"]),
                "--max-black-ratio", str(h3_spec["max_black_ratio"]),
                "--composition-direction", str(h3_spec["composition_direction"]),
                "--ffprobe", ffprobe_bin,
                "--resume",
            ]
            if h3_spec["mode"] in {"native-chain", "continuous-chain"}:
                h3_command.extend(
                    [
                        "--mode", str(h3_spec["mode"]),
                        "--scene-segments", str(bgm_segments_path(config)),
                        "--frames-dir", str(h3_spec["frames_dir"]),
                        "--max-chain-length", str(h3_spec["max_chain_length"]),
                        "--ffmpeg", ffmpeg_bin,
                    ]
                )
                if h3_spec["mode"] == "continuous-chain":
                    h3_command.extend(
                        [
                            "--keyframes-dir", str(h3_spec["keyframes_dir"]),
                            "--shot-plan-output", str(h3_spec["shot_plan"]),
                        ]
                    )
                    if h3_spec["legacy_checkpoint"] is not None:
                        h3_command.extend(
                            [
                                "--legacy-checkpoint",
                                str(h3_spec["legacy_checkpoint"]),
                            ]
                        )
            else:
                h3_command.extend(["--images-dir", str(directory)])
            if config.get("illustrations", {}).get("prompt_audit_enabled", True):
                h3_command.extend(
                    [
                        "--prompt-audit-checkpoint",
                        str(visual_prompt_checkpoint_path(config)),
                    ]
                )
            run_checked_subprocess(h3_command, int(h3_spec["generation_timeout"]))
            h3_clip_files = sorted(h3_spec["clips_dir"].glob("*.mp4"))
            dependencies.extend(
                [
                    h3_spec["checkpoint"],
                    *h3_clip_files,
                    h3_spec["render_checkpoint"],
                    *sorted(h3_spec["segments_dir"].glob("*.mp4")),
                ]
            )
        cached_duration_matches = (
            nonempty_file(video)
            and abs(
                media_duration_seconds(video, ffprobe_bin)
                - media_duration_seconds(audio, ffprobe_bin)
            ) <= 1.0
        )
        h3_cache_complete = True
        if h3_spec is not None:
            plan_count = len(load_illustration_plan(plan_path))
            clip_complete, clip_done, clip_total, _ = h3_clip_checkpoint_complete(
                h3_spec["checkpoint"],
                mode=h3_spec["mode"],
                plan_count=plan_count,
            )
            render_done, render_total = _completed_checkpoint_records(
                h3_spec["render_checkpoint"], "segments"
            )
            h3_cache_complete = (
                clip_complete
                and render_total == plan_count
                and render_done == render_total
            )
        if (
            not options.get("force", False)
            and h3_cache_complete
            and dependencies_are_older(video, dependencies)
            and cached_duration_matches
        ):
            print(f"  using cached {variant['name']} video: {video}")
            outputs.append(video)
            continue
        if use_h3:
            assert h3_spec is not None
            compose_command = [
                sys.executable,
                str(ROOT / "scripts/generate_h3_video.py"),
                "--plan", str(plan_path),
                "--h3-checkpoint", str(h3_spec["checkpoint"]),
                "--segments-output-dir", str(h3_spec["segments_dir"]),
                "--render-checkpoint", str(h3_spec["render_checkpoint"]),
                "--audio", str(audio),
                "--output", str(video),
                "--size", str(settings["size"]),
                "--novel", str(resolve_path(config["novel"]["text_path"])),
                "--labels", str(resolve_path(config["novel"]["labels_path"])),
                "--segments-dir", str(output_dir(config) / "segments"),
                "--subtitle-output", str(subtitle),
                "--subtitle-font", str(options.get("subtitle_font", "SimHei")),
                "--subtitle-font-size", str(options.get("subtitle_font_size", 42)),
                "--max-subtitle-chars", str(options.get("max_subtitle_chars", 16)),
                "--max-subtitle-lines", str(options.get("max_subtitle_lines", 2)),
                "--fps", str(options.get("fps", 25)),
                "--crf", str(options.get("crf", 18)),
                "--preset", str(options.get("preset", "slow")),
                "--audio-bitrate", str(options.get("audio_bitrate", "256k")),
                "--ffmpeg", ffmpeg_bin,
                "--ffprobe", ffprobe_bin,
                "--resume",
            ]
            if h3_spec["mode"] == "illustration-bridge":
                compose_command.extend(["--images-dir", str(directory)])
            run_checked_subprocess(compose_command, int(h3_spec["render_timeout"]))
            dependencies.extend(
                [
                    h3_spec["render_checkpoint"],
                    *sorted(h3_spec["segments_dir"].glob("*.mp4")),
                ]
            )
        else:
            run_checked_subprocess(
                [
                    sys.executable,
                    str(ROOT / "scripts/generate_video.py"),
                    "--plan", str(plan_path),
                    "--illustrations-dir", str(directory),
                    "--audio", str(audio),
                    "--output", str(video),
                    "--size", str(settings["size"]),
                    "--novel", str(resolve_path(config["novel"]["text_path"])),
                    "--labels", str(resolve_path(config["novel"]["labels_path"])),
                    "--segments-dir", str(output_dir(config) / "segments"),
                    "--subtitle-output", str(subtitle),
                    "--subtitle-font", str(options.get("subtitle_font", "SimHei")),
                    "--subtitle-font-size", str(options.get("subtitle_font_size", 42)),
                    "--max-subtitle-chars", str(options.get("max_subtitle_chars", 16)),
                    "--max-subtitle-lines", str(options.get("max_subtitle_lines", 2)),
                    "--fps", str(options.get("fps", 25)),
                    "--crf", str(options.get("crf", 18)),
                    "--preset", str(options.get("preset", "slow")),
                    "--audio-bitrate", str(options.get("audio_bitrate", "256k")),
                    "--ffmpeg", ffmpeg_bin,
                    "--ffprobe", ffprobe_bin,
                ],
                int(options.get("timeout", 86400)),
            )
        if (
            not nonempty_file(video)
            or not dependencies_are_older(video, dependencies)
            or abs(
                media_duration_seconds(video, ffprobe_bin)
                - media_duration_seconds(audio, ffprobe_bin)
            ) > 1.0
        ):
            raise PipelineError(
                f"Video generator did not create a fresh {variant['name']} output: {video}"
            )
        outputs.append(video)
    return str(outputs[0])


def stage_cache_status(stage: str, config: dict[str, Any]) -> tuple[bool, str]:
    if stage == "parse":
        paths = [resolve_path(config["novel"]["text_path"]), resolve_path(config["novel"]["labels_path"])]
        return all(path.is_file() for path in paths), ", ".join(map(str, paths))
    if stage == "gender":
        valid = gender_cache_valid(read_json(gender_result_path(), {}))
        return valid, str(gender_result_path())
    if stage == "emotion":
        valid = emotion_cache_valid(read_json(emotion_result_path(), {}))
        return valid, str(emotion_result_path())
    if stage == "performance":
        if not config.get("features", {}).get("performance_direction", False):
            return True, "disabled in config"
        try:
            dialogues, characters, novel_text = step_parse(config)
            gender_results = require_gender_results(characters, dialogues, novel_text)
            emotion_results = (
                require_emotion_results(dialogues, novel_text)
                if config.get("features", {}).get("emotion_label", True)
                else {}
            )
            valid, problems, _ = performance_cache_valid(
                config,
                dialogues,
                novel_text,
                gender_results,
                emotion_results,
            )
            return valid, "; ".join(problems) or str(performance_result_path(config))
        except (OSError, ValueError, PipelineError) as exc:
            return False, str(exc)
    if stage == "tts":
        problems = validate_current_tts_cache(config)
        return not problems, "; ".join(problems) or str(output_dir(config) / "segments")
    if stage == "splice":
        path = speech_output_path(config)
        tts_manifest = output_dir(config) / "segments" / "segments_manifest.json"
        problems = validate_current_tts_cache(config)
        valid = not problems and nonempty_file(path) and dependencies_are_older(path, [tts_manifest])
        return valid, "; ".join(problems) or str(path)
    if stage in {"bgm-segment", "bgm-label"}:
        path = bgm_segments_path(config)
        segments = load_segments(path) if path.exists() else []
        novel_text = resolve_path(config["novel"]["text_path"]).read_text(encoding="utf-8")
        novel_lines = len(novel_text.splitlines())
        source_hash = bgm_source_hash(novel_text)
        chunks = int(config.get("bgm", {}).get("segmentation_chunks", 10))
        problems = validate_segments(segments, novel_lines, 5, max(20, chunks * 20)) if segments else ["missing segments"]
        correct_segmentation_model = all(
            item.get("segmentation_model") == SENSENOVA_FLASH_LITE_MODEL
            and item.get("segmentation_pipeline_version") == BGM_SEGMENTATION_PIPELINE_VERSION
            and item.get("segmentation_source_hash") == source_hash
            for item in segments
        )
        correctly_labeled = stage != "bgm-label" or all(
            item.get("bgm_type")
            and item.get("bgm_model") == SENSENOVA_FLASH_LITE_MODEL
            and item.get("bgm_pipeline_version") == BGM_TYPE_PIPELINE_VERSION
            and item.get("bgm_source_hash") == source_hash
            for item in segments
        )
        valid = not problems and correct_segmentation_model and correctly_labeled
        return valid, "; ".join(problems) or str(path)
    if stage == "bgm-generate":
        path = bgm_segments_path(config)
        segments = load_segments(path) if path.exists() else []
        clips = int(config.get("bgm", {}).get("clips_per_segment", 3))
        clip_duration = float(config.get("bgm", {}).get("clip_duration", 30))
        inference_steps = int(config.get("bgm", {}).get("inference_steps", 8))
        directory = output_dir(config) / "bgm"
        problems = validate_bgm_cache(
            segments,
            directory / "bgm_manifest.json",
            directory,
            clips,
            clip_duration=clip_duration,
            inference_steps=inference_steps,
        )
        return not problems, "; ".join(problems) or str(directory)
    if stage == "bgm-mix":
        path = mixed_audio_path(config)
        speech = speech_output_path(config)
        bgm_directory = output_dir(config) / "bgm"
        valid = (
            not validate_current_tts_cache(config)
            and nonempty_file(path)
            and nonempty_file(speech)
            and dependencies_are_older(
                path,
                [speech, bgm_segments_path(config), bgm_directory / "bgm_manifest.json"],
            )
            and abs(media_duration_seconds(path) - media_duration_seconds(speech)) <= 1.0
        )
        return valid, str(path)
    if stage == "illustration-plan":
        path = illustration_plan_path(config)
        problems = validate_illustration_plan(path)
        return not problems, "; ".join(problems) or str(path)
    if stage == "illustrations":
        variants = illustration_variant_specs(config)
        problems = []
        for variant in variants:
            current = validate_illustrations(
                illustration_plan_path(config),
                variant["directory"],
                variant["checkpoint"],
                visual_prompt_checkpoint_path(config),
                **illustration_validation_options(
                    config,
                    settings=variant["settings"],
                    composition_suffix=variant["composition_suffix"],
                ),
            )
            problems.extend(f"{variant['name']}: {problem}" for problem in current)
        return not problems, "; ".join(problems) or ", ".join(
            str(variant["directory"]) for variant in variants
        )
    if stage == "video":
        audio = mixed_audio_path(config)
        plan_path = illustration_plan_path(config)
        plan_count = len(load_illustration_plan(plan_path))
        ffprobe_bin = str(
            config.get("video", {}).get("ffprobe", os.environ.get("FFPROBE_BIN", "ffprobe"))
        )
        problems = []
        for variant in video_variant_specs(config):
            illustration = variant["illustrations"]
            path = variant["output"]
            h3_spec = h3_variant_spec(config, variant) if h3_video_enabled(config) else None
            native_h3 = (
                h3_spec is not None
                and h3_spec["mode"] in {"native-chain", "continuous-chain"}
            )
            dependencies = [plan_path, audio]
            if native_h3:
                audit_path = visual_prompt_checkpoint_path(config)
                illustration_problems = validate_visual_prompt_audit(
                    plan_path,
                    audit_path,
                    novel_path=resolve_path(config["novel"]["text_path"]),
                    character_cards=character_cards_path(config),
                    enabled=config.get("illustrations", {}).get(
                        "prompt_audit_enabled", True
                    ),
                )
                dependencies.append(bgm_segments_path(config))
                if config.get("illustrations", {}).get("prompt_audit_enabled", True):
                    dependencies.append(audit_path)
            else:
                illustration_problems = validate_illustrations(
                    plan_path,
                    illustration["directory"],
                    illustration["checkpoint"],
                    visual_prompt_checkpoint_path(config),
                    **illustration_validation_options(
                        config,
                        settings=illustration["settings"],
                        composition_suffix=illustration["composition_suffix"],
                    ),
                )
                dependencies.extend(sorted(illustration["directory"].glob("*.png")))
            h3_problems: list[str] = []
            if h3_spec is not None:
                clip_complete, _, _, clip_reason = h3_clip_checkpoint_complete(
                    h3_spec["checkpoint"],
                    mode=h3_spec["mode"],
                    plan_count=plan_count,
                )
                render_done, render_total = _completed_checkpoint_records(
                    h3_spec["render_checkpoint"], "segments"
                )
                if not clip_complete:
                    h3_problems.append(clip_reason)
                if render_total != plan_count or render_done != render_total:
                    h3_problems.append("H3 render checkpoint is incomplete")
                dependencies.extend(
                    [
                        h3_spec["checkpoint"],
                        h3_spec["render_checkpoint"],
                        *sorted(h3_spec["clips_dir"].glob("*.mp4")),
                        *sorted(h3_spec["segments_dir"].glob("*.mp4")),
                    ]
                )
            valid = (
                not illustration_problems
                and not h3_problems
                and nonempty_file(path)
                and nonempty_file(audio)
                and dependencies_are_older(path, dependencies)
                and abs(
                    media_duration_seconds(path, ffprobe_bin)
                    - media_duration_seconds(audio, ffprobe_bin)
                ) <= 1.0
            )
            if not valid:
                problems.extend(
                    f"{variant['name']}: {problem}" for problem in illustration_problems
                )
                problems.extend(f"{variant['name']}: {problem}" for problem in h3_problems)
                if not illustration_problems and not h3_problems:
                    problems.append(f"{variant['name']}: video is missing, stale, or has wrong duration")
        return not problems, "; ".join(problems) or ", ".join(
            str(variant["output"]) for variant in video_variant_specs(config)
        )
    raise ValueError(stage)


def dry_run_report(config: dict[str, Any], selected: tuple[str, ...]) -> bool:
    print("Dry run: no stages will execute")
    first_index = STAGES.index(selected[0])
    prerequisites_ok = True
    if first_index:
        if selected[0] == "video" and h3_video_enabled(config):
            h3_modes = {
                h3_variant_spec(config, variant)["mode"]
                for variant in video_variant_specs(config)
            }
            if h3_modes and h3_modes <= {"native-chain", "continuous-chain"}:
                problems = validate_visual_prompt_audit(
                    illustration_plan_path(config),
                    visual_prompt_checkpoint_path(config),
                    novel_path=resolve_path(config["novel"]["text_path"]),
                    character_cards=character_cards_path(config),
                    enabled=config.get("illustrations", {}).get(
                        "prompt_audit_enabled", True
                    ),
                )
                prerequisites_ok = not problems
                detail = "; ".join(problems) or str(visual_prompt_checkpoint_path(config))
                print(
                    "  prerequisite visual-prompt-audit: "
                    f"{'READY' if prerequisites_ok else 'MISSING'} - {detail}"
                )
            else:
                prerequisite = STAGES[first_index - 1]
                valid, detail = stage_cache_status(prerequisite, config)
                print(
                    f"  prerequisite {prerequisite}: "
                    f"{'READY' if valid else 'MISSING'} - {detail}"
                )
                prerequisites_ok = valid
        else:
            prerequisite = STAGES[first_index - 1]
            valid, detail = stage_cache_status(prerequisite, config)
            print(f"  prerequisite {prerequisite}: {'READY' if valid else 'MISSING'} - {detail}")
            prerequisites_ok = valid
    for stage in selected:
        valid, detail = stage_cache_status(stage, config)
        if len(detail) > 500:
            detail = detail[:500] + "..."
        print(f"  {stage}: {'CACHED' if valid else 'WOULD RUN'} - {detail}")
    return prerequisites_ok


def start_streaming_tts_worker(
    config: dict[str, Any],
    *,
    limit: int = 0,
    range_value: str = "",
) -> subprocess.Popen:
    settings = config.get("streaming_tts", {})
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_streaming_tts.py"),
        "--config",
        str(config["_config_path"]),
        "--log",
        str(settings.get("log_path", "logs/streaming_tts.log")),
    ]
    if limit:
        command.extend(["--limit", str(limit)])
    if range_value:
        command.extend(["--range", range_value])
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=str(ROOT), creationflags=creationflags)
    print(f"  streaming TTS worker started (PID {process.pid})", flush=True)
    return process


def stop_streaming_tts_worker(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=15)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Novel Voice Cast full pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file")
    parser.add_argument("--novel", help="Override novel.text_path for this run only")
    parser.add_argument("--labels", help="Override novel.labels_path for this run only")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N dialogues")
    parser.add_argument("--range", dest="range_value", default="", help="Dialogue range, e.g. 100-200")
    parser.add_argument("--log", default="logs/run_full.log", help="UTF-8 runtime log path")
    parser.add_argument("--from-stage", choices=STAGES, help="First stage to execute")
    parser.add_argument("--to-stage", choices=STAGES, help="Last stage to execute")
    parser.add_argument(
        "--stream-tts",
        action="store_true",
        help="Run resumable VoxCPM synthesis concurrently with performance direction",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate stage range and caches without executing")
    parser.add_argument(
        "--desktop-events",
        action="store_true",
        help="Emit versioned [STAGE], [PROGRESS], and [LOG] JSON lines",
    )
    parser.add_argument(
        "--stop-file",
        help="Interrupt the main thread when this desktop-owned file appears",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    display_command = subprocess.list2cmdline(
        [sys.executable, "-u", str(Path(__file__).resolve()), *effective_argv]
    )
    DESKTOP_EVENTS.configure(args.desktop_events, stream=sys.stdout, command=display_command)
    configure_logging(args.log)
    try:
        selected = stage_slice(args.from_stage, args.to_stage)
        config = apply_input_overrides(
            load_config(args.config),
            novel_path=args.novel,
            labels_path=args.labels,
        )
    except KeyboardInterrupt:
        DESKTOP_EVENTS.log("WARNING", "启动阶段已被用户中断")
        print("\nStartup interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        DESKTOP_EVENTS.log("ERROR", f"启动失败：{exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("Novel Voice Cast - full pipeline")
    print(f"Root: {ROOT}")
    print(f"Stages: {selected[0]} -> {selected[-1]}")
    print("=" * 72)
    DESKTOP_EVENTS.log(
        "INFO",
        f"流水线启动：{selected[0]} -> {selected[-1]}",
    )
    if args.dry_run:
        return 0 if dry_run_report(config, selected) else 1

    recorder = PipelineRecorder(output_dir(config) / "run_full_manifest.json", selected)
    stop_watcher = StopFileWatcher(resolve_path(args.stop_file)) if args.stop_file else None
    if stop_watcher is not None:
        stop_watcher.start()
    total_started = time.monotonic()
    parsed: tuple[list[dict], list[str], str] | None = None
    gender_results: dict[str, Any] | None = None
    emotion_results: dict[str, Any] | None = None
    performance_results: dict[str, Any] | None = None
    streaming_tts_process: subprocess.Popen | None = None
    segments: list[dict[str, Any]] | None = None
    bgm_segments: list[dict[str, Any]] | None = None

    def ensure_parsed() -> tuple[list[dict], list[str], str]:
        nonlocal parsed
        if parsed is None:
            dialogues, characters, novel_text = step_parse(config)
            dialogues, characters = apply_dialogue_selection(
                dialogues, characters, args.limit, args.range_value
            )
            parsed = dialogues, characters, novel_text
        return parsed

    try:
        if "parse" in selected:
            parsed = execute_stage(
                recorder,
                "parse",
                lambda: apply_dialogue_selection(*step_parse(config)[:2], args.limit, args.range_value),
                [resolve_path(config["novel"]["text_path"]), resolve_path(config["novel"]["labels_path"])],
            ) + (resolve_path(config["novel"]["text_path"]).read_text(encoding="utf-8"),)

        if "gender" in selected:
            dialogues, characters, novel_text = ensure_parsed()
            gender_total = len([name for name in characters if name != "旁白"])
            gender_results = execute_stage(
                recorder,
                "gender",
                lambda: step_gender(config, characters, dialogues, novel_text),
                [gender_result_path()],
                progress_probe=checkpoint_progress_probe(
                    [(ROOT / "backend/data/gender_results.checkpoint.json", "results", gender_total)],
                    "已识别角色",
                ),
            )

        if "emotion" in selected:
            if not config.get("features", {}).get("emotion_label", True):
                emotion_results = {}
                record_skipped(recorder, "emotion", "disabled in config")
            else:
                dialogues, _, novel_text = ensure_parsed()
                emotion_total = sum(
                    1
                    for dialogue in dialogues
                    if dialogue.get("speaker")
                    and dialogue.get("speaker") not in {"旁白", "narrator", "Narrator"}
                )
                emotion_results = execute_stage(
                    recorder,
                    "emotion",
                    lambda: step_emotion(
                        config,
                        dialogues,
                        novel_text,
                        force_reprocess=config.get("features", {}).get("force_reprocess", False),
                    ),
                    [emotion_result_path()],
                    progress_probe=checkpoint_progress_probe(
                        [
                            (
                                ROOT / "backend/data/emotion_results.checkpoint.json",
                                "results",
                                emotion_total,
                            )
                        ],
                        "已标注情绪",
                    ),
                )

        if "performance" in selected:
            if not config.get("features", {}).get("performance_direction", False):
                performance_results = {}
                record_skipped(recorder, "performance", "disabled in config")
            else:
                dialogues, characters, novel_text = ensure_parsed()
                if gender_results is None:
                    gender_results = require_gender_results(characters, dialogues, novel_text)
                if emotion_results is None:
                    emotion_results = (
                        require_emotion_results(dialogues, novel_text)
                        if config.get("features", {}).get("emotion_label", True)
                        else {}
                    )
                streaming_enabled = bool(
                    args.stream_tts or config.get("streaming_tts", {}).get("enabled", False)
                )
                if streaming_enabled and "tts" in selected:
                    streaming_tts_process = start_streaming_tts_worker(
                        config,
                        limit=args.limit,
                        range_value=args.range_value,
                    )
                performance_groups = _performance_groups(
                    config,
                    dialogues,
                    gender_results or {},
                )
                performance_results = execute_stage(
                    recorder,
                    "performance",
                    lambda: step_performance(
                        config,
                        dialogues,
                        novel_text,
                        gender_results or {},
                        emotion_results or {},
                    ),
                    [
                        path
                        for group in performance_groups
                        for path in (group["profile_path"], group["output_path"])
                    ],
                    progress_probe=checkpoint_progress_probe(
                        [
                            entry
                            for group in performance_groups
                            for entry in (
                                (
                                    group["profile_checkpoint"],
                                    "completed_speakers",
                                    len(group["speakers"]),
                                ),
                                (
                                    group["direction_checkpoint"],
                                    "completed_indices",
                                    len(group["targets"]),
                                ),
                            )
                        ],
                        "角色档案与逐句导演",
                    ),
                )

        if "tts" in selected:
            dialogues, characters, novel_text = ensure_parsed()
            if gender_results is None:
                gender_results = require_gender_results(characters, dialogues, novel_text)
            if emotion_results is None:
                emotion_results = (
                    require_emotion_results(dialogues, novel_text)
                    if config.get("features", {}).get("emotion_label", True)
                    else {}
                )
            if performance_results is None:
                performance_results = (
                    require_performance_results(
                        config,
                        dialogues,
                        novel_text,
                        gender_results or {},
                        emotion_results or {},
                    )
                    if config.get("features", {}).get("performance_direction", False)
                    else {}
                )
            if streaming_tts_process is not None:
                print("  performance complete; waiting for streaming TTS to catch up", flush=True)
                streaming_returncode = streaming_tts_process.wait()
                streaming_tts_process = None
                if streaming_returncode != 0:
                    print(
                        "  streaming TTS stopped with an error; the TTS stage will resume "
                        "its successful WAV checkpoints and retry the remainder",
                        flush=True,
                    )
            segments = execute_stage(
                recorder,
                "tts",
                lambda: step_tts(
                    config,
                    dialogues,
                    gender_results or {},
                    emotion_results or {},
                    performance_results or {},
                ),
                [output_dir(config) / "segments", output_dir(config) / "segments/segments_manifest.json"],
                progress_probe=tts_progress_probe(config, len(dialogues)),
            )

        if "splice" in selected:
            dialogues, characters, novel_text = ensure_parsed()
            if segments is None:
                if gender_results is None:
                    gender_results = require_gender_results(characters, dialogues, novel_text)
                if emotion_results is None:
                    emotion_results = (
                        require_emotion_results(dialogues, novel_text)
                        if config.get("features", {}).get("emotion_label", True)
                        else {}
                    )
                if performance_results is None:
                    performance_results = (
                        require_performance_results(
                            config,
                            dialogues,
                            novel_text,
                            gender_results or {},
                            emotion_results or {},
                        )
                        if config.get("features", {}).get("performance_direction", False)
                        else {}
                    )
                problems = validate_tts_manifest(
                    config,
                    dialogues,
                    gender_results,
                    emotion_results,
                    performance_results,
                )
                if problems:
                    raise PipelineError(f"Valid TTS cache is required for splice: {problems[:5]}")
                segments = segments_for_dialogues(config, dialogues)
            execute_stage(
                recorder,
                "splice",
                lambda: step_splice(config, segments or []),
                lambda result: [result[0]],
            )

        bgm_stage_names = {"bgm-segment", "bgm-label", "bgm-generate", "bgm-mix"}
        for stage in selected:
            if stage in bgm_stage_names and not config.get("bgm", {}).get("enabled", False):
                record_skipped(recorder, stage, "BGM disabled in config")

        if config.get("bgm", {}).get("enabled", False):
            if "bgm-segment" in selected:
                bgm_segments = execute_stage(
                    recorder,
                    "bgm-segment",
                    lambda: step_bgm_segmentation(config),
                    [bgm_segments_path(config)],
                    progress_probe=checkpoint_progress_probe(
                        [
                            (
                                ROOT / "backend/data/bgm_segmentation.checkpoint.json",
                                "chunks",
                                int(config.get("bgm", {}).get("segmentation_chunks", 6)),
                            )
                        ],
                        "已复核章节块",
                    ),
                )
            if "bgm-label" in selected:
                if bgm_segments is None:
                    path = bgm_segments_path(config)
                    if not path.exists():
                        raise PipelineError("BGM segmentation cache is required before BGM labeling")
                    bgm_segments = load_segments(path)
                novel_text = parsed[2] if parsed else ""
                bgm_segments = execute_stage(
                    recorder,
                    "bgm-label",
                    lambda: step_bgm_labeling(config, bgm_segments or [], novel_text),
                    [bgm_segments_path(config)],
                    progress_probe=checkpoint_progress_probe(
                        [
                            (
                                ROOT / "backend/data/bgm_types.checkpoint.json",
                                "results",
                                len(bgm_segments or []),
                            )
                        ],
                        "已标注音乐场景",
                    ),
                )
            if "bgm-generate" in selected:
                if bgm_segments is None:
                    path = bgm_segments_path(config)
                    if not path.exists():
                        raise PipelineError("BGM segmentation cache is required before BGM generation")
                    bgm_segments = load_segments(path)
                execute_stage(
                    recorder,
                    "bgm-generate",
                    lambda: step_bgm_generation(config),
                    lambda result: [result, output_dir(config) / "bgm"],
                    progress_probe=bgm_generation_progress_probe(config, len(bgm_segments)),
                )
            if "bgm-mix" in selected:
                execute_stage(
                    recorder,
                    "bgm-mix",
                    lambda: step_bgm_mixing(config),
                    lambda result: [result],
                )

        if "illustration-plan" in selected:
            execute_stage(
                recorder,
                "illustration-plan",
                lambda: step_illustration_plan(config),
                lambda result: [result],
            )
        if "illustrations" in selected:
            plan_path = illustration_plan_path(config)
            if not plan_path.exists():
                raise PipelineError("Illustration plan cache is required before illustration generation")
            illustration_plan_count = len(load_illustration_plan(plan_path))
            execute_stage(
                recorder,
                "illustrations",
                lambda: step_illustrations(config),
                lambda result: [
                    result,
                    *[
                        path
                        for variant in illustration_variant_specs(config)
                        for path in (variant["directory"], variant["checkpoint"])
                    ],
                    visual_prompt_checkpoint_path(config),
                ],
                progress_probe=illustration_progress_probe(config, illustration_plan_count),
            )
        if "video" in selected:
            video_plan_count = len(load_illustration_plan(illustration_plan_path(config)))
            execute_stage(
                recorder,
                "video",
                lambda: step_video(config),
                lambda result: [
                    result,
                    *[variant["output"] for variant in video_variant_specs(config)],
                ],
                progress_probe=(
                    h3_video_progress_probe(config, video_plan_count)
                    if h3_video_enabled(config)
                    else None
                ),
            )
    except KeyboardInterrupt:
        if stop_watcher is not None:
            stop_watcher.stop()
        stop_streaming_tts_worker(streaming_tts_process)
        recorder.data["run_status"] = "interrupted"
        recorder.data["run_error"] = "interrupted by user"
        recorder.data["run_finished_at"] = utc_now()
        recorder.save()
        DESKTOP_EVENTS.log("WARNING", "流水线已停止，已完成的断点得到保留")
        print("\nPipeline interrupted; completed checkpoints were preserved.", file=sys.stderr)
        print(f"Run manifest: {recorder.path}")
        return 130
    except Exception as exc:
        if stop_watcher is not None:
            stop_watcher.stop()
        stop_streaming_tts_worker(streaming_tts_process)
        recorder.data["run_status"] = "failed"
        recorder.data["run_error"] = str(exc)
        recorder.data["run_finished_at"] = utc_now()
        recorder.save()
        DESKTOP_EVENTS.log("ERROR", f"流水线失败：{exc}")
        print("\n" + "=" * 72)
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        print(f"Run manifest: {recorder.path}")
        return 1

    if stop_watcher is not None:
        stop_watcher.stop()
    stop_streaming_tts_worker(streaming_tts_process)
    recorder.finish()
    DESKTOP_EVENTS.log(
        "INFO",
        f"流水线完成，总耗时 {format_time(time.monotonic() - total_started)}",
    )
    print("\n" + "=" * 72)
    print(f"Pipeline complete in {format_time(time.monotonic() - total_started)}")
    print(f"Run manifest: {recorder.path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
