"""Versioned stdout events consumed by the Electron desktop application."""
from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any, TextIO


EVENT_VERSION = 1
EVENT_PREFIXES = {
    "stage": "[STAGE]",
    "progress": "[PROGRESS]",
    "log": "[LOG]",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DesktopEventEmitter:
    """Write thread-safe, one-line JSON events without affecting normal CLI output."""

    def __init__(self, enabled: bool = False, stream: TextIO | None = None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.command = ""
        self.active_stage: str | None = None
        self._last_percent: dict[str, float] = {}
        self._lock = threading.Lock()

    def configure(
        self,
        enabled: bool,
        *,
        stream: TextIO | None = None,
        command: str = "",
    ) -> None:
        self.enabled = enabled
        if stream is not None:
            self.stream = stream
        self.command = command
        self.active_stage = None
        self._last_percent.clear()

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        value = {
            "version": EVENT_VERSION,
            "timestamp": _utc_now(),
            **payload,
        }
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            print(f"{EVENT_PREFIXES[kind]} {line}", file=self.stream, flush=True)

    def stage(
        self,
        stage: str,
        *,
        index: int,
        total: int,
        status: str,
        elapsed_seconds: float = 0.0,
        operation: str = "",
        artifacts: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        if status == "running":
            self.active_stage = stage
            self._last_percent[stage] = 0.0
        elif self.active_stage == stage:
            self.active_stage = None
        payload: dict[str, Any] = {
            "stage": stage,
            "index": index,
            "total": total,
            "status": status,
            "elapsed_seconds": round(max(0.0, float(elapsed_seconds)), 3),
            "operation": operation,
        }
        if self.command:
            payload["command"] = self.command
        if artifacts is not None:
            payload["artifacts"] = artifacts
        if error:
            payload["error"] = error
        self._emit("stage", payload)

    def progress(
        self,
        stage: str,
        *,
        current: int | float,
        total: int | float,
        operation: str,
        status: str = "running",
    ) -> None:
        numeric_total = max(0.0, float(total))
        numeric_current = max(0.0, float(current))
        calculated = (
            min(100.0, numeric_current * 100.0 / numeric_total)
            if numeric_total
            else 0.0
        )
        percent = max(self._last_percent.get(stage, 0.0), calculated)
        self._last_percent[stage] = percent
        payload: dict[str, Any] = {
            "stage": stage,
            "current": current,
            "total": total,
            "percent": round(percent, 2),
            "status": status,
            "operation": operation,
        }
        if self.command:
            payload["command"] = self.command
        self._emit("progress", payload)

    def log(self, level: str, message: str, *, stage: str | None = None) -> None:
        payload: dict[str, Any] = {
            "level": str(level).upper(),
            "message": str(message),
        }
        current_stage = stage or self.active_stage
        if current_stage:
            payload["stage"] = current_stage
        self._emit("log", payload)


class DesktopEventLoggingHandler(logging.Handler):
    """Mirror Python logging records into the structured desktop event stream."""

    def __init__(self, emitter: DesktopEventEmitter):
        super().__init__(level=logging.NOTSET)
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.log(record.levelname, record.getMessage())
        except Exception:
            self.handleError(record)
