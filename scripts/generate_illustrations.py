"""Generate illustration-plan images with Agnes or a local form-based API."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import os
import queue
import random
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests


LOGGER = logging.getLogger("generate_illustrations")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENDPOINT = "https://apihub.agnes-ai.com/v1/images/generations"
DEFAULT_MODEL = "agnes-image-2.1-flash"
DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8000/generate"
DEFAULT_LOCAL_MODEL = "local-image"
DEFAULT_PROXY = "http://127.0.0.1:7890"
DEFAULT_SIZE = "896x1152"
DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, nude, lowres, worst quality, low quality, blurry, bad anatomy, "
    "bad hands, extra fingers, missing fingers, deformed, duplicate, text, "
    "watermark, signature, logo"
)
SEMANTIC_GUARD = (
    "Render only entities that are physically present in the described scene. "
    "Anything introduced only as a metaphor, simile, resemblance, imagination, memory, "
    "or visual impression must not appear as a literal extra object or character. "
    "Do not add unmentioned people, animals, text, logos, or watermarks."
)

PLAN_PATH = Path("output/illustration_plan.json")
OUTPUT_DIR = Path("output/illustrations")
CHECKPOINT_PATH = Path("output/illustrations_checkpoint.json")
NOVEL_PATH = Path("novels/novel.txt")
CHARACTER_CARDS_PATH = Path("docs/角色卡.md")
PROMPT_AUDIT_CHECKPOINT_PATH = Path("backend/data/visual_prompt_audit.checkpoint.json")

MAX_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = {408, 409, 425, 429}
CHECKPOINT_VERSION = 5
ERROR_SUMMARY_LIMIT = 500
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
BEARER_TOKEN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


class AgnesImageError(RuntimeError):
    """A generation failure with retry metadata."""

    def __init__(self, message: str, *, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass(frozen=True)
class GenerationResult:
    content: bytes
    source: str


@dataclass(frozen=True)
class GenerationTarget:
    """One output aspect ratio sharing the same audited semantic prompt."""

    name: str
    client: "AgnesImageClient"
    output_dir: Path
    checkpoint_path: Path
    composition_suffix: str = ""


def build_generation_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    comparison_targets = [
        match.strip(" ,")
        for match in re.findall(
            r"(?i)\b(?:look(?:s|ed|ing)?\s+like|resembl(?:e|es|ed|ing)|as\s+if)\s+([^.;!?]+)",
            prompt,
        )
        if match.strip(" ,")
    ]
    comparison_guard = ""
    if comparison_targets:
        comparison_guard = (
            " The following are comparison targets, not physical scene contents, and must be absent: "
            + "; ".join(comparison_targets)
            + "."
        )
    return f"{prompt}\n\nSemantic fidelity requirements: {SEMANTIC_GUARD}{comparison_guard}"


def prompt_for_target(prompt: str, composition_suffix: str = "") -> str:
    """Append presentation-only framing without changing audited scene facts."""

    prompt = prompt.strip()
    suffix = composition_suffix.strip()
    return f"{prompt}\n\nComposition requirement: {suffix}" if suffix else prompt


def generation_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def apply_audited_prompts(
    plans: Sequence[dict[str, Any]], audits: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(plans) != len(audits):
        raise ValueError("visual prompt audit count does not match illustration plan")
    output: list[dict[str, Any]] = []
    for plan, audit in zip(plans, audits):
        audited_prompt = str(audit.get("audited_prompt", "")).strip()
        if not audited_prompt:
            raise ValueError("visual prompt audit returned an empty prompt")
        item = dict(plan)
        item["original_prompt"] = str(plan.get("prompt", ""))
        item["prompt"] = audited_prompt
        item["prompt_audit"] = audit
        output.append(item)
    return output


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan(path: Path = PLAN_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("illustrations", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Illustration plan must be a list: {path}")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Every illustration plan item must be an object: {path}")
    return items


def load_api_key(
    explicit_path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Load the key without logging or embedding it in exception messages."""
    env = os.environ if environ is None else environ
    inline_key = env.get("AGNES_API_KEY", "").strip()
    if inline_key:
        return inline_key

    configured_path = explicit_path or (
        Path(env["AGNES_API_KEY_FILE"]) if env.get("AGNES_API_KEY_FILE") else None
    )
    candidates = [configured_path] if configured_path else [
        PROJECT_ROOT / "config" / "agneskey.txt",
        PROJECT_ROOT / "config" / "agnes_api_key",
        PROJECT_ROOT / "config" / "agnes_api_key.txt",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        key = candidate.read_text(encoding="utf-8").strip()
        if key:
            return key

    if configured_path:
        raise RuntimeError(f"Agnes API key file is missing or empty: {configured_path}")
    raise RuntimeError(
        "Agnes API key is not configured; set AGNES_API_KEY, AGNES_API_KEY_FILE, "
        "or create config/agneskey.txt (config/agnes_api_key is also supported)"
    )


def sanitize_error(value: object, secrets: Sequence[str] = ()) -> str:
    message = str(value).replace("\r", " ").replace("\n", " ")
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = BEARER_TOKEN.sub(r"\1[REDACTED]", message)
    return message[:ERROR_SUMMARY_LIMIT]


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


class AgnesImageClient:
    provider = "agnes"

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        size: str = DEFAULT_SIZE,
        proxy: str | None = DEFAULT_PROXY,
        timeout: float = 180.0,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base: float = 2.0,
        interval_min: float = 1.0,
        interval_max: float = 2.0,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Agnes API key cannot be empty")
        if not 1 <= max_attempts <= MAX_ATTEMPTS:
            raise ValueError(f"max_attempts must be between 1 and {MAX_ATTEMPTS}")
        if interval_min < 0 or interval_max < interval_min:
            raise ValueError("Request interval must satisfy 0 <= min <= max")
        self._api_key = api_key.strip()
        self.endpoint = endpoint
        self.model = model
        self.size = size
        self.proxy = proxy or None
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = max(0.0, backoff_base)
        self.interval_min = interval_min
        self.interval_max = interval_max
        self._session = session or requests.Session()
        self._sleep = sleep_fn
        self._random = random_fn

    @property
    def generation_settings(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "endpoint": self.endpoint.rstrip("/"),
            "size": self.size,
        }

    @property
    def secret_values(self) -> tuple[str, ...]:
        return (self._api_key,)

    @property
    def _proxies(self) -> dict[str, str] | None:
        if not self.proxy:
            return None
        return {"http": self.proxy, "https": self.proxy}

    def generate(
        self,
        prompt: str,
        *,
        on_attempt: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "prompt": build_generation_prompt(prompt),
            "size": self.size,
            "extra_body": {"response_format": "b64_json"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: AgnesImageError | None = None
        for attempt in range(1, self.max_attempts + 1):
            if on_attempt:
                on_attempt(attempt)
            LOGGER.info("Agnes request attempt %d/%d", attempt, self.max_attempts)
            try:
                response = self._session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    proxies=self._proxies,
                    timeout=self.timeout,
                )
                result = self._parse_generation_response(response)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = AgnesImageError(
                    sanitize_error(exc, (self._api_key,)), retryable=True
                )
            except AgnesImageError as exc:
                last_error = exc
            else:
                self._sleep_request_interval()
                return result

            self._sleep_request_interval()
            if not last_error.retryable or attempt >= self.max_attempts:
                raise last_error

            exponential = self.backoff_base * (2 ** (attempt - 1))
            retry_delay = max(exponential, last_error.retry_after or 0.0)
            LOGGER.warning(
                "Retryable Agnes failure on attempt %d/%d: %s; backoff %.2fs",
                attempt,
                self.max_attempts,
                sanitize_error(last_error, (self._api_key,)),
                retry_delay,
            )
            self._sleep(retry_delay)

        raise last_error or AgnesImageError("Agnes request failed", retryable=False)

    def _sleep_request_interval(self) -> None:
        delay = self._random(self.interval_min, self.interval_max)
        if delay > 0:
            LOGGER.info("Request interval %.2fs", delay)
            self._sleep(delay)

    def _parse_generation_response(self, response: requests.Response) -> GenerationResult:
        if not 200 <= response.status_code < 300:
            summary = sanitize_error(response.text, (self._api_key,))
            retryable = (
                response.status_code in RETRYABLE_STATUS_CODES or response.status_code >= 500
            )
            raise AgnesImageError(
                f"HTTP {response.status_code}: {summary}",
                retryable=retryable,
                retry_after=_retry_after_seconds(response),
            )

        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise AgnesImageError("Agnes returned invalid JSON", retryable=True) from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise AgnesImageError("Agnes response has no image data", retryable=True)

        item = data[0]
        encoded = item.get("b64_json")
        if isinstance(encoded, str) and encoded:
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                content = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise AgnesImageError("Agnes returned invalid base64 image data", retryable=True) from exc
            if not content:
                raise AgnesImageError("Agnes returned an empty base64 image", retryable=True)
            return GenerationResult(content=content, source="base64")

        image_url = item.get("url")
        if isinstance(image_url, str) and image_url:
            try:
                image_response = self._session.get(
                    image_url,
                    proxies=self._proxies,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                raise AgnesImageError("Image URL download failed", retryable=True) from exc
            if not 200 <= image_response.status_code < 300:
                raise AgnesImageError(
                    f"Image URL download returned HTTP {image_response.status_code}",
                    retryable=image_response.status_code >= 500
                    or image_response.status_code in RETRYABLE_STATUS_CODES,
                    retry_after=_retry_after_seconds(image_response),
                )
            if not image_response.content:
                raise AgnesImageError("Image URL download returned an empty body", retryable=True)
            return GenerationResult(content=image_response.content, source="url")

        raise AgnesImageError("Agnes response contains neither url nor b64_json", retryable=True)


class LocalImageClient(AgnesImageClient):
    """Client for the local ``POST /generate`` form API returning raw PNG bytes."""

    provider = "local-http"

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_LOCAL_ENDPOINT,
        model: str = DEFAULT_LOCAL_MODEL,
        size: str = DEFAULT_SIZE,
        steps: int = 25,
        cfg: float = 7.0,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: int = -1,
        timeout: float = 900.0,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base: float = 2.0,
        interval_min: float = 0.0,
        interval_max: float = 0.0,
        session: requests.Session | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        try:
            width_text, height_text = size.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("size must use WIDTHxHEIGHT, for example 896x1152") from exc
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if cfg <= 0:
            raise ValueError("cfg must be positive")

        local_session = session or requests.Session()
        if session is None:
            # A localhost inference server must never be routed through the
            # optional Agnes/network proxy inherited from the environment.
            local_session.trust_env = False
        super().__init__(
            api_key="local-image-no-key",
            endpoint=endpoint.rstrip("/"),
            model=model,
            size=f"{width}x{height}",
            proxy=None,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_base=backoff_base,
            interval_min=interval_min,
            interval_max=interval_max,
            session=local_session,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )
        self.width = width
        self.height = height
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.negative_prompt = negative_prompt.strip()
        self.seed = int(seed)

    @property
    def generation_settings(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "size": self.size,
            "steps": self.steps,
            "cfg": self.cfg,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

    @property
    def secret_values(self) -> tuple[str, ...]:
        return ()

    def generate(
        self,
        prompt: str,
        *,
        on_attempt: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        payload = {
            "prompt": build_generation_prompt(prompt),
            "neg_prompt": self.negative_prompt,
            "steps": self.steps,
            "cfg": self.cfg,
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
        }
        last_error: AgnesImageError | None = None
        for attempt in range(1, self.max_attempts + 1):
            if on_attempt:
                on_attempt(attempt)
            LOGGER.info("Local image request attempt %d/%d", attempt, self.max_attempts)
            try:
                response = self._session.post(
                    self.endpoint,
                    data=payload,
                    timeout=self.timeout,
                )
                result = self._parse_local_response(response)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = AgnesImageError(sanitize_error(exc), retryable=True)
            except AgnesImageError as exc:
                last_error = exc
            else:
                self._sleep_request_interval()
                return result

            self._sleep_request_interval()
            if not last_error.retryable or attempt >= self.max_attempts:
                raise last_error
            retry_delay = max(
                self.backoff_base * (2 ** (attempt - 1)),
                last_error.retry_after or 0.0,
            )
            LOGGER.warning(
                "Retryable local image failure on attempt %d/%d: %s; backoff %.2fs",
                attempt,
                self.max_attempts,
                sanitize_error(last_error),
                retry_delay,
            )
            self._sleep(retry_delay)
        raise last_error or AgnesImageError("Local image request failed", retryable=False)

    def _parse_local_response(self, response: requests.Response) -> GenerationResult:
        if not 200 <= response.status_code < 300:
            raise AgnesImageError(
                f"HTTP {response.status_code}: {sanitize_error(response.text)}",
                retryable=(
                    response.status_code in RETRYABLE_STATUS_CODES
                    or response.status_code >= 500
                ),
                retry_after=_retry_after_seconds(response),
            )
        content = response.content
        if not content:
            raise AgnesImageError("Local image server returned an empty body", retryable=True)
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            content_type = response.headers.get("Content-Type", "unknown")
            raise AgnesImageError(
                f"Local image server did not return PNG bytes (Content-Type: {content_type})",
                retryable=True,
            )
        return GenerationResult(content=content, source="raw-png")


def _safe_title(title: object, index: int) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", str(title)).strip(" .")
    return cleaned[:80] or f"img_{index + 1:04d}"


def _output_path(output_dir: Path, index: int, item: Mapping[str, Any]) -> Path:
    title = _safe_title(item.get("title", ""), index)
    return output_dir / f"{index + 1:04d}_{title}.png"


def _new_record(index: int, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "title": str(item.get("title", f"img_{index + 1:04d}")),
        "status": "pending",
        "attempts": 0,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "output_file": None,
        "error_summary": None,
        "prompt_hash": None,
    }


def generation_source_hash(plan: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(list(plan), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prepare_checkpoint(
    plan: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    output_dir: Path,
    client: AgnesImageClient,
    *,
    resume: bool,
    identity_plan: Sequence[Mapping[str, Any]] | None = None,
    audit_source_hash: str | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "provider": client.provider,
        "model": client.model,
        "endpoint": client.endpoint.rstrip("/"),
        "size": client.size,
        "generation_settings": client.generation_settings,
        "source_hash": generation_source_hash(identity_plan or plan),
        "audit_source_hash": audit_source_hash,
        "updated_at": utc_now(),
        "images": [_new_record(index, item) for index, item in enumerate(plan)],
    }
    if not resume or not checkpoint_path.exists():
        return checkpoint

    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    compatible = (
        isinstance(raw, dict)
        and raw.get("version") == CHECKPOINT_VERSION
        and raw.get("provider") == client.provider
        and raw.get("model") == client.model
        and raw.get("endpoint") == client.endpoint.rstrip("/")
        and raw.get("size") == client.size
        and raw.get("generation_settings") == client.generation_settings
        and raw.get("source_hash") == checkpoint["source_hash"]
        and raw.get("audit_source_hash") == audit_source_hash
    )
    if not compatible:
        LOGGER.warning(
            "Ignoring incompatible or legacy illustration checkpoint: %s",
            checkpoint_path,
        )
        return checkpoint

    existing = raw.get("images")
    if not isinstance(existing, list):
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    for index, old_record in enumerate(existing[: len(plan)]):
        if not isinstance(old_record, dict):
            continue
        record = checkpoint["images"][index]
        for field in (
            "status",
            "attempts",
            "started_at",
            "ended_at",
            "duration_seconds",
            "output_file",
            "error_summary",
            "prompt_hash",
        ):
            if field in old_record:
                record[field] = old_record[field]
        if record["status"] == "success":
            output_file = record.get("output_file")
            if not output_file or not Path(output_file).exists():
                record.update(status="pending", output_file=None, error_summary="Output file missing")
    return checkpoint


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(path, checkpoint)


def _reset_record(record: dict[str, Any], *, error_summary: str | None = None) -> None:
    record.update(
        status="pending",
        attempts=0,
        started_at=None,
        ended_at=None,
        duration_seconds=None,
        output_file=None,
        error_summary=error_summary,
        prompt_hash=None,
    )


def _generate_one(
    item: Mapping[str, Any],
    index: int,
    total: int,
    *,
    client: AgnesImageClient,
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    resume: bool,
    composition_suffix: str = "",
    target_name: str = "default",
) -> None:
    record = checkpoint["images"][index]
    prompt = item.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        timestamp = utc_now()
        record.update(
            status="failed",
            started_at=timestamp,
            ended_at=timestamp,
            duration_seconds=0.0,
            output_file=None,
            error_summary="Empty illustration prompt",
            prompt_hash=None,
        )
        _save_checkpoint(checkpoint_path, checkpoint)
        LOGGER.error("[%s %d/%d] failed: empty prompt", target_name, index + 1, total)
        return

    effective_prompt = prompt_for_target(prompt, composition_suffix)
    effective_hash = generation_prompt_hash(effective_prompt)
    if resume and record.get("status") == "success":
        output_file = record.get("output_file")
        if (
            record.get("prompt_hash") == effective_hash
            and output_file
            and Path(str(output_file)).is_file()
        ):
            LOGGER.info(
                "[%s %d/%d] already complete: %s",
                target_name,
                index + 1,
                total,
                record["title"],
            )
            return
        _reset_record(record, error_summary="Prompt or output changed; regenerating")

    LOGGER.info("[%s %d/%d] generating: %s", target_name, index + 1, total, record["title"])
    started = time.perf_counter()
    record.update(
        status="running",
        started_at=utc_now(),
        ended_at=None,
        duration_seconds=None,
        output_file=None,
        error_summary=None,
        prompt_hash=effective_hash,
    )
    _save_checkpoint(checkpoint_path, checkpoint)

    def mark_attempt(_attempt: int) -> None:
        record["attempts"] = int(record.get("attempts") or 0) + 1
        record["status"] = "running"
        _save_checkpoint(checkpoint_path, checkpoint)

    try:
        result = client.generate(effective_prompt, on_attempt=mark_attempt)
        output_path = _output_path(output_dir, index, item)
        atomic_write_bytes(output_path, result.content)
        duration = time.perf_counter() - started
        record.update(
            status="success",
            ended_at=utc_now(),
            duration_seconds=round(duration, 3),
            output_file=str(output_path),
            error_summary=None,
        )
        _save_checkpoint(checkpoint_path, checkpoint)
        LOGGER.info(
            "[%s %d/%d] success: %s (%s, %.2fs, %.1f KiB)",
            target_name,
            index + 1,
            total,
            output_path,
            result.source,
            duration,
            len(result.content) / 1024,
        )
    except Exception as exc:
        duration = time.perf_counter() - started
        summary = sanitize_error(exc, client.secret_values)
        record.update(
            status="failed",
            ended_at=utc_now(),
            duration_seconds=round(duration, 3),
            output_file=None,
            error_summary=summary,
        )
        _save_checkpoint(checkpoint_path, checkpoint)
        LOGGER.error("[%s %d/%d] failed: %s", target_name, index + 1, total, summary)


def run_generation(
    plan: Sequence[Mapping[str, Any]],
    *,
    client: AgnesImageClient,
    output_dir: Path = OUTPUT_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    resume: bool = False,
    composition_suffix: str = "",
    target_name: str = "default",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _prepare_checkpoint(
        plan, checkpoint_path, output_dir, client, resume=resume
    )
    _save_checkpoint(checkpoint_path, checkpoint)

    total = len(plan)
    for index, item in enumerate(plan):
        _generate_one(
            item,
            index,
            total,
            client=client,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            resume=resume,
            composition_suffix=composition_suffix,
            target_name=target_name,
        )

    return checkpoint


AUDIT_QUEUE_DONE = object()


def run_streaming_audited_generation(
    plan: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[GenerationTarget],
    audited_items: "queue.Queue[object]",
    audit_errors: list[BaseException],
    audit_source_hash: str,
    resume: bool,
    wait_log_seconds: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """Consume audited prompts as they arrive and render each target serially."""

    if not targets:
        raise ValueError("at least one image generation target is required")
    checkpoints: dict[str, dict[str, Any]] = {}
    for target in targets:
        target.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _prepare_checkpoint(
            plan,
            target.checkpoint_path,
            target.output_dir,
            target.client,
            resume=resume,
            identity_plan=plan,
            audit_source_hash=audit_source_hash,
        )
        _save_checkpoint(target.checkpoint_path, checkpoint)
        checkpoints[target.name] = checkpoint

    total = len(plan)
    last_wait_log = 0.0
    while True:
        try:
            payload = audited_items.get(timeout=1.0)
        except queue.Empty:
            now = time.monotonic()
            if now - last_wait_log >= wait_log_seconds:
                LOGGER.info(
                    "Image generation has caught up with prompt auditing; waiting for the next audited prompt"
                )
                last_wait_log = now
            continue
        if payload is AUDIT_QUEUE_DONE:
            break
        if not (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[0], int)
            and isinstance(payload[1], dict)
        ):
            raise ValueError("invalid audited prompt queue item")
        index, audit = payload
        if not 0 <= index < total:
            raise ValueError(f"audited prompt index is outside the plan: {index}")
        audited_item = apply_audited_prompts([dict(plan[index])], [audit])[0]
        for target in targets:
            _generate_one(
                audited_item,
                index,
                total,
                client=target.client,
                output_dir=target.output_dir,
                checkpoint_path=target.checkpoint_path,
                checkpoint=checkpoints[target.name],
                resume=resume,
                composition_suffix=target.composition_suffix,
                target_name=target.name,
            )

    if audit_errors:
        raise RuntimeError(f"visual prompt audit worker failed: {audit_errors[0]}") from audit_errors[0]
    return checkpoints


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Skip successful checkpoint entries")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--novel", type=Path, default=NOVEL_PATH)
    parser.add_argument("--character-cards", type=Path, default=CHARACTER_CARDS_PATH)
    parser.add_argument(
        "--prompt-audit-checkpoint",
        type=Path,
        default=PROMPT_AUDIT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--skip-prompt-audit",
        action="store_true",
        help="Bypass SenseNova visual grounding (not recommended for production)",
    )
    parser.add_argument(
        "--force-prompt-audit",
        action="store_true",
        help="Ignore a compatible visual-prompt checkpoint and audit every prompt again",
    )
    parser.add_argument("--key-file", type=Path)
    parser.add_argument(
        "--provider",
        choices=("agnes", "local-http"),
        default=os.environ.get("IMAGE_PROVIDER", "agnes"),
    )
    parser.add_argument(
        "--endpoint", default=os.environ.get("AGNES_IMAGE_ENDPOINT", DEFAULT_ENDPOINT)
    )
    parser.add_argument("--model", default=os.environ.get("AGNES_IMAGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--size", default=os.environ.get("AGNES_IMAGE_SIZE", DEFAULT_SIZE))
    parser.add_argument("--composition-suffix", default="")
    parser.add_argument("--landscape-output-dir", type=Path)
    parser.add_argument("--landscape-checkpoint", type=Path)
    parser.add_argument("--landscape-size", default="1280x720")
    parser.add_argument("--landscape-composition-suffix", default="")
    parser.add_argument("--proxy", default=os.environ.get("AGNES_PROXY_URL", DEFAULT_PROXY))
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_float("AGNES_IMAGE_TIMEOUT_SECONDS", 180.0),
    )
    parser.add_argument(
        "--interval-min",
        type=float,
        default=_env_float("AGNES_IMAGE_INTERVAL_MIN", 1.0),
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=_env_float("AGNES_IMAGE_INTERVAL_MAX", 2.0),
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=_env_float("AGNES_IMAGE_BACKOFF_BASE_SECONDS", 2.0),
    )
    return parser


def _build_client(args: argparse.Namespace, size: str) -> AgnesImageClient:
    if args.provider == "local-http":
        return LocalImageClient(
            endpoint=args.endpoint,
            model=args.model,
            size=size,
            steps=args.steps,
            cfg=args.cfg,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            backoff_base=args.backoff_base,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
        )
    return AgnesImageClient(
        api_key=load_api_key(args.key_file),
        endpoint=args.endpoint,
        model=args.model,
        size=size,
        proxy=args.proxy,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        backoff_base=args.backoff_base,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interval_min < 0 or args.interval_max < args.interval_min:
        parser.error("request interval must satisfy 0 <= min <= max")
    if bool(args.landscape_output_dir) != bool(args.landscape_checkpoint):
        parser.error("--landscape-output-dir and --landscape-checkpoint must be used together")

    try:
        plan = load_plan(args.plan)
        targets = [
            GenerationTarget(
                name="portrait",
                client=_build_client(args, args.size),
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                composition_suffix=args.composition_suffix,
            )
        ]
        if args.landscape_output_dir and args.landscape_checkpoint:
            targets.append(
                GenerationTarget(
                    name="landscape",
                    client=_build_client(args, args.landscape_size),
                    output_dir=args.landscape_output_dir,
                    checkpoint_path=args.landscape_checkpoint,
                    composition_suffix=args.landscape_composition_suffix,
                )
            )
        started = time.perf_counter()
        if not args.skip_prompt_audit:
            backend_path = str(PROJECT_ROOT / "backend")
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from app.core.llm_client import LLMClient
            from app.core.visual_prompt_auditor import (
                audit_visual_prompts,
                visual_prompt_source_hash,
            )

            novel_text = args.novel.read_text(encoding="utf-8")
            character_cards_text = (
                args.character_cards.read_text(encoding="utf-8")
                if args.character_cards.is_file()
                else ""
            )
            LOGGER.info("Starting streaming prompt audit and image generation for %d scenes", len(plan))
            audit_client = LLMClient.for_flash_lite("illustration_prompt_audit")
            audit_source_hash = visual_prompt_source_hash(
                novel_text,
                plan,
                character_cards_text,
            )
            audited_items: "queue.Queue[object]" = queue.Queue()
            audit_errors: list[BaseException] = []

            def audit_worker() -> None:
                try:
                    audit_visual_prompts(
                        plan,
                        novel_text,
                        character_cards_text,
                        client=audit_client,
                        checkpoint_path=args.prompt_audit_checkpoint,
                        resume=not args.force_prompt_audit,
                        on_completed=lambda index, result: audited_items.put((index, result)),
                    )
                except BaseException as exc:  # propagated after queued completed work is consumed
                    audit_errors.append(exc)
                finally:
                    audited_items.put(AUDIT_QUEUE_DONE)

            thread = threading.Thread(
                target=audit_worker,
                name="visual-prompt-audit",
                daemon=True,
            )
            thread.start()
            checkpoints = run_streaming_audited_generation(
                plan,
                targets=targets,
                audited_items=audited_items,
                audit_errors=audit_errors,
                audit_source_hash=audit_source_hash,
                resume=args.resume,
            )
            thread.join()
            audit_client.log_summary()
        else:
            checkpoints = {}
            for target in targets:
                LOGGER.info(
                    "Starting image generation target=%s provider=%s images=%d size=%s output=%s",
                    target.name,
                    target.client.provider,
                    len(plan),
                    target.client.size,
                    target.output_dir,
                )
                checkpoints[target.name] = run_generation(
                    plan,
                    client=target.client,
                    output_dir=target.output_dir,
                    checkpoint_path=target.checkpoint_path,
                    resume=args.resume,
                    composition_suffix=target.composition_suffix,
                    target_name=target.name,
                )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Generation aborted: %s", sanitize_error(exc))
        return 2

    failed_total = 0
    for target in targets:
        checkpoint = checkpoints[target.name]
        statuses = [record["status"] for record in checkpoint["images"]]
        succeeded = statuses.count("success")
        failed = statuses.count("failed")
        failed_total += failed
        LOGGER.info(
            "Generation target finished: target=%s success=%d failed=%d total=%d checkpoint=%s",
            target.name,
            succeeded,
            failed,
            len(statuses),
            target.checkpoint_path,
        )
    LOGGER.info("All image targets finished in %.2fs", time.perf_counter() - started)
    return 0 if failed_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
