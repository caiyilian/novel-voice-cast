"""Generate illustration-plan images with the Agnes OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
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
DEFAULT_PROXY = "http://127.0.0.1:7890"
DEFAULT_SIZE = "896x1152"
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
CHECKPOINT_VERSION = 3
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
) -> dict[str, Any]:
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "provider": "agnes",
        "model": client.model,
        "endpoint": client.endpoint,
        "size": client.size,
        "source_hash": generation_source_hash(plan),
        "updated_at": utc_now(),
        "images": [_new_record(index, item) for index, item in enumerate(plan)],
    }
    if not resume or not checkpoint_path.exists():
        return checkpoint

    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    compatible = (
        isinstance(raw, dict)
        and raw.get("version") == CHECKPOINT_VERSION
        and raw.get("provider") == "agnes"
        and raw.get("model") == client.model
        and raw.get("endpoint") == client.endpoint
        and raw.get("size") == client.size
        and raw.get("source_hash") == checkpoint["source_hash"]
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


def run_generation(
    plan: Sequence[Mapping[str, Any]],
    *,
    client: AgnesImageClient,
    output_dir: Path = OUTPUT_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _prepare_checkpoint(
        plan, checkpoint_path, output_dir, client, resume=resume
    )
    _save_checkpoint(checkpoint_path, checkpoint)

    total = len(plan)
    for index, item in enumerate(plan):
        record = checkpoint["images"][index]
        if resume and record["status"] == "success":
            LOGGER.info("[%d/%d] already complete: %s", index + 1, total, record["title"])
            continue

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
            )
            _save_checkpoint(checkpoint_path, checkpoint)
            LOGGER.error("[%d/%d] failed: empty prompt", index + 1, total)
            continue

        LOGGER.info("[%d/%d] generating: %s", index + 1, total, record["title"])
        started = time.perf_counter()
        record.update(
            status="running",
            started_at=utc_now(),
            ended_at=None,
            duration_seconds=None,
            output_file=None,
            error_summary=None,
        )
        _save_checkpoint(checkpoint_path, checkpoint)

        def mark_attempt(_attempt: int) -> None:
            record["attempts"] = int(record.get("attempts") or 0) + 1
            record["status"] = "running"
            _save_checkpoint(checkpoint_path, checkpoint)

        try:
            result = client.generate(prompt.strip(), on_attempt=mark_attempt)
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
                "[%d/%d] success: %s (%s, %.2fs, %.1f KiB)",
                index + 1,
                total,
                output_path,
                result.source,
                duration,
                len(result.content) / 1024,
            )
        except Exception as exc:
            duration = time.perf_counter() - started
            summary = sanitize_error(exc, (client._api_key,))
            record.update(
                status="failed",
                ended_at=utc_now(),
                duration_seconds=round(duration, 3),
                output_file=None,
                error_summary=summary,
            )
            _save_checkpoint(checkpoint_path, checkpoint)
            LOGGER.error("[%d/%d] failed: %s", index + 1, total, summary)

    return checkpoint


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
        "--endpoint", default=os.environ.get("AGNES_IMAGE_ENDPOINT", DEFAULT_ENDPOINT)
    )
    parser.add_argument("--model", default=os.environ.get("AGNES_IMAGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--size", default=os.environ.get("AGNES_IMAGE_SIZE", DEFAULT_SIZE))
    parser.add_argument("--proxy", default=os.environ.get("AGNES_PROXY_URL", DEFAULT_PROXY))
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

    try:
        api_key = load_api_key(args.key_file)
        plan = load_plan(args.plan)
        if not args.skip_prompt_audit:
            backend_path = str(PROJECT_ROOT / "backend")
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from app.core.llm_client import LLMClient
            from app.core.visual_prompt_auditor import audit_visual_prompts

            novel_text = args.novel.read_text(encoding="utf-8")
            character_cards_text = (
                args.character_cards.read_text(encoding="utf-8")
                if args.character_cards.is_file()
                else ""
            )
            LOGGER.info(
                "Auditing %d visual prompts with SenseNova before Agnes generation",
                len(plan),
            )
            audit_client = LLMClient.for_flash_lite("illustration_prompt_audit")
            audits = audit_visual_prompts(
                plan,
                novel_text,
                character_cards_text,
                client=audit_client,
                checkpoint_path=args.prompt_audit_checkpoint,
                resume=not args.force_prompt_audit,
            )
            plan = apply_audited_prompts(plan, audits)
            audit_client.log_summary()
        client = AgnesImageClient(
            api_key=api_key,
            endpoint=args.endpoint,
            model=args.model,
            size=args.size,
            proxy=args.proxy,
            timeout=args.timeout,
            backoff_base=args.backoff_base,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
        )
        LOGGER.info(
            "Starting Agnes generation: images=%d model=%s endpoint=%s output=%s resume=%s",
            len(plan),
            client.model,
            client.endpoint,
            args.output_dir,
            args.resume,
        )
        started = time.perf_counter()
        checkpoint = run_generation(
            plan,
            client=client,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Generation aborted: %s", sanitize_error(exc))
        return 2

    statuses = [record["status"] for record in checkpoint["images"]]
    succeeded = statuses.count("success")
    failed = statuses.count("failed")
    LOGGER.info(
        "Generation finished: success=%d failed=%d total=%d elapsed=%.2fs checkpoint=%s",
        succeeded,
        failed,
        len(statuses),
        time.perf_counter() - started,
        args.checkpoint,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
