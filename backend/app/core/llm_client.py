"""OpenAI-compatible multi-account LLM client.

The default client preserves the illustration planner's DeepSeek -> Agnes
chain.  ``for_flash_lite`` is the strict SenseNova-only mode used by gender,
emotion, and BGM agents.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger("llm_client")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    account_index: int = -1
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


SENSENOVA_BASE = "https://token.sensenova.cn/v1"
AGNES_BASE = "https://apihub.agnes-ai.com/v1"
AGNES_PROXY = os.environ.get("AGNES_PROXY_URL", "http://127.0.0.1:7890")
SENSENOVA_MODEL = "deepseek-v4-flash"
SENSENOVA_FLASH_LITE_MODEL = "sensenova-6.7-flash-lite"
SENSENOVA_FLASH_LITE_CONTEXT_TOKENS = 256 * 1024
AGNES_MODEL = "agnes-2.0-flash"

MODEL_CONTEXT_WINDOWS = {
    SENSENOVA_FLASH_LITE_MODEL: SENSENOVA_FLASH_LITE_CONTEXT_TOKENS,
}

SENSENOVA_QUOTA_COOLDOWN_SECONDS = int(
    os.environ.get("SENSENOVA_QUOTA_COOLDOWN_SECONDS", str(5 * 60 * 60))
)
SENSENOVA_RETRY_COOLDOWN_SECONDS = int(
    os.environ.get("SENSENOVA_RETRY_COOLDOWN_SECONDS", "60")
)
SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS = int(
    os.environ.get("SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS", "15")
)
SENSENOVA_CALLS_PER_WINDOW = int(os.environ.get("SENSENOVA_CALLS_PER_WINDOW", "1500"))
SENSENOVA_WINDOW_SECONDS = int(os.environ.get("SENSENOVA_WINDOW_SECONDS", str(5 * 60 * 60)))

KEY_FILE_SENSENOVA = Path("config/sensenova_apikeys")
KEY_FILE_AGNES = Path("config/agnes_api_key")
DEFAULT_QUOTA_STATE_PATH = Path("logs/sensenova_quota_state.json")


class LLMClient:
    """Multi-account client with rotation, cooldowns, and call telemetry."""

    def __init__(
        self,
        *,
        sensenova_model: str = SENSENOVA_MODEL,
        allow_agnes_fallback: bool = True,
        wait_for_sensenova: bool = False,
        module_name: str = "general",
        telemetry_path: Path | str | None = None,
        context_window_tokens: Optional[int] = None,
        quota_state_path: Path | str | None = DEFAULT_QUOTA_STATE_PATH,
        sensenova_keys: Optional[list[str]] = None,
        agnes_key: Optional[str] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.sensenova_model = sensenova_model
        self.allow_agnes_fallback = allow_agnes_fallback
        self.wait_for_sensenova = wait_for_sensenova
        self.module_name = module_name
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        inferred_context_window = MODEL_CONTEXT_WINDOWS.get(sensenova_model, 32768)
        self.context_window_tokens = max(1, int(context_window_tokens or inferred_context_window))
        self.quota_state_path = Path(quota_state_path) if quota_state_path else None
        self._sensenova_keys = list(sensenova_keys) if sensenova_keys is not None else _load_lines(KEY_FILE_SENSENOVA)
        self._agnes_key = agnes_key if agnes_key is not None else _load_first_line(KEY_FILE_AGNES)
        self._sleep = sleep_fn
        self._round_robin_index = 0
        self._call_sequence = 0
        self.run_id = uuid.uuid4().hex
        self._sensenova_disabled_until = [0.0 for _ in self._sensenova_keys]
        self._account_calls = [deque() for _ in self._sensenova_keys]
        self._totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._load_quota_state()

    @classmethod
    def for_flash_lite(
        cls,
        module_name: str,
        telemetry_path: Path | str | None = None,
        **kwargs: Any,
    ) -> "LLMClient":
        """Build the strict client required by issue #99.

        There is deliberately no fallback model.  If every account is cooling
        down, the caller waits instead of silently changing model behavior.
        """
        return cls(
            sensenova_model=SENSENOVA_FLASH_LITE_MODEL,
            allow_agnes_fallback=False,
            wait_for_sensenova=True,
            module_name=module_name,
            telemetry_path=telemetry_path or Path("logs") / f"{module_name}_llm_calls.jsonl",
            **kwargs,
        )

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        agent_role: str = "",
        trace_id: str = "",
        agent_round: Optional[int] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> LLMResult:
        """Send one logical completion, rotating accounts when needed."""
        self._ensure_context_fits(
            self.sensenova_model,
            messages,
            tools,
            max_tokens,
            self.context_window_tokens,
        )
        if not self._sensenova_keys and not (self.allow_agnes_fallback and self._agnes_key):
            raise AllModelsExhausted("No SenseNova API keys configured")

        attempted: set[int] = set()
        while True:
            idx = self._next_sensenova_index(exclude=attempted)
            if idx is None:
                if self.allow_agnes_fallback and self._agnes_key:
                    self._ensure_context_fits(AGNES_MODEL, messages, tools, max_tokens, 32768)
                    logger.info("All SenseNova accounts unavailable; falling back to %s", AGNES_MODEL)
                    return self._request_and_record(
                        base_url=AGNES_BASE,
                        model=AGNES_MODEL,
                        api_key=self._agnes_key,
                        account_index=-1,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        proxy=AGNES_PROXY,
                        agent_role=agent_role,
                        trace_id=trace_id,
                        agent_round=agent_round,
                        extra_body=extra_body,
                    )
                if not self.wait_for_sensenova:
                    raise AllModelsExhausted("All SenseNova accounts are in cooldown")
                wait_seconds = self._seconds_until_available()
                logger.warning(
                    "All %d SenseNova accounts unavailable; waiting %.1fs for the earliest cooldown",
                    len(self._sensenova_keys),
                    wait_seconds,
                )
                self._sleep(max(0.05, wait_seconds))
                attempted.clear()
                continue

            attempted.add(idx)
            self._reserve_account_call(idx)
            try:
                return self._request_and_record(
                    base_url=SENSENOVA_BASE,
                    model=self.sensenova_model,
                    api_key=self._sensenova_keys[idx],
                    account_index=idx,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    proxy=None,
                    agent_role=agent_role,
                    trace_id=trace_id,
                    agent_round=agent_round,
                    extra_body=extra_body,
                )
            except InsufficientQuota as exc:
                self._disable_sensenova_key(idx, SENSENOVA_QUOTA_COOLDOWN_SECONDS)
                logger.warning("SenseNova account %d quota unavailable; cooldown %ds: %s", idx + 1, SENSENOVA_QUOTA_COOLDOWN_SECONDS, exc)
            except RateLimited as exc:
                self._disable_sensenova_key(idx, SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS)
                logger.warning("SenseNova account %d rate limited; cooldown %ds: %s", idx + 1, SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS, exc)
            except RetryableError as exc:
                self._disable_sensenova_key(idx, SENSENOVA_RETRY_COOLDOWN_SECONDS)
                logger.warning("SenseNova account %d retryable failure: %s", idx + 1, exc)
            except InvalidCredentials as exc:
                self._disable_sensenova_key(idx, SENSENOVA_QUOTA_COOLDOWN_SECONDS)
                logger.error(
                    "SenseNova account %d credentials rejected; disabled and rotating: %s",
                    idx + 1,
                    exc,
                )
            except FatalLLMError:
                raise
            except Exception as exc:
                self._disable_sensenova_key(idx, SENSENOVA_RETRY_COOLDOWN_SECONDS)
                logger.warning("SenseNova account %d failed: %s", idx + 1, exc)

    @staticmethod
    def _ensure_context_fits(
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        max_tokens: int,
        context_window_tokens: int,
    ) -> None:
        estimated_prompt = _estimate_prompt_tokens(messages, tools)
        reserved = estimated_prompt + max(0, int(max_tokens))
        if reserved > context_window_tokens:
            raise ContextWindowExceeded(
                f"{model} request reserves about {reserved} tokens "
                f"({estimated_prompt} prompt + {max_tokens} completion), "
                f"exceeding its {context_window_tokens}-token context window"
            )

    def _request_and_record(
        self,
        *,
        agent_role: str = "",
        trace_id: str = "",
        agent_round: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResult:
        messages = kwargs["messages"]
        model = kwargs["model"]
        account_index = kwargs["account_index"]
        started = time.perf_counter()
        self._call_sequence += 1
        try:
            result = self._call_openai(**kwargs, request_attempts=1)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._write_telemetry(
                messages,
                model,
                account_index,
                {},
                elapsed,
                "error",
                str(exc),
                agent_role,
                trace_id=trace_id,
                agent_round=agent_round,
                tools=kwargs.get("tools"),
                requested_max_tokens=int(kwargs.get("max_tokens", 0)),
            )
            raise

        elapsed = time.perf_counter() - started
        usage = _normalise_usage(result.usage, messages, result.content, kwargs.get("tools"))
        result = LLMResult(
            content=result.content,
            tool_calls=result.tool_calls,
            model=result.model,
            account_index=result.account_index,
            usage=usage,
            raw=result.raw,
            elapsed_seconds=elapsed,
        )
        choice = (result.raw.get("choices") or [{}])[0]
        raw_message = choice.get("message") or {}
        response_meta = {
            "finish_reason": choice.get("finish_reason"),
            "reasoning_chars": len(raw_message.get("reasoning") or raw_message.get("reasoning_content") or ""),
            "tool_call_count": len(result.tool_calls),
        }
        self._write_telemetry(
            messages,
            model,
            account_index,
            usage,
            elapsed,
            "ok",
            "",
            agent_role,
            response_meta,
            trace_id=trace_id,
            agent_round=agent_round,
            tools=kwargs.get("tools"),
            requested_max_tokens=int(kwargs.get("max_tokens", 0)),
        )
        return result

    def _next_sensenova_index(self, exclude: set[int] | None = None) -> Optional[int]:
        exclude = exclude or set()
        now = time.time()
        self._prune_call_windows(now)
        count = len(self._sensenova_keys)
        for _ in range(count):
            idx = self._round_robin_index % count
            self._round_robin_index += 1
            if idx in exclude:
                continue
            if len(self._account_calls[idx]) >= SENSENOVA_CALLS_PER_WINDOW:
                oldest = self._account_calls[idx][0]
                self._sensenova_disabled_until[idx] = max(
                    self._sensenova_disabled_until[idx], oldest + SENSENOVA_WINDOW_SECONDS
                )
            if self._sensenova_disabled_until[idx] <= now:
                return idx
        return None

    def _seconds_until_available(self) -> float:
        now = time.time()
        self._prune_call_windows(now)
        waits = []
        for idx, disabled_until in enumerate(self._sensenova_disabled_until):
            quota_until = 0.0
            if len(self._account_calls[idx]) >= SENSENOVA_CALLS_PER_WINDOW:
                quota_until = self._account_calls[idx][0] + SENSENOVA_WINDOW_SECONDS
            waits.append(max(disabled_until, quota_until) - now)
        return max(0.05, min(waits)) if waits else 0.05

    def _disable_sensenova_key(self, idx: int, seconds: int) -> None:
        self._sensenova_disabled_until[idx] = max(
            self._sensenova_disabled_until[idx], time.time() + max(0, seconds)
        )
        self._save_quota_state()

    def _reserve_account_call(self, idx: int) -> None:
        self._account_calls[idx].append(time.time())
        self._save_quota_state()

    def _prune_call_windows(self, now: float) -> None:
        cutoff = now - SENSENOVA_WINDOW_SECONDS
        for calls in self._account_calls:
            while calls and calls[0] <= cutoff:
                calls.popleft()

    def _load_quota_state(self) -> None:
        if not self.quota_state_path or not self.quota_state_path.exists():
            return
        try:
            raw = json.loads(self.quota_state_path.read_text(encoding="utf-8"))
            for idx, values in enumerate(raw.get("account_calls", [])):
                if idx < len(self._account_calls):
                    self._account_calls[idx].extend(float(value) for value in values)
            for idx, value in enumerate(raw.get("disabled_until", [])):
                if idx < len(self._sensenova_disabled_until):
                    self._sensenova_disabled_until[idx] = float(value)
            self._prune_call_windows(time.time())
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid SenseNova quota state %s: %s", self.quota_state_path, exc)

    def _save_quota_state(self) -> None:
        if not self.quota_state_path:
            return
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "window_seconds": SENSENOVA_WINDOW_SECONDS,
            "calls_per_window": SENSENOVA_CALLS_PER_WINDOW,
            "account_calls": [list(calls) for calls in self._account_calls],
            "disabled_until": self._sensenova_disabled_until,
        }
        _atomic_write_json(self.quota_state_path, payload)

    def _write_telemetry(
        self,
        messages: list[dict],
        model: str,
        account_index: int,
        usage: dict,
        elapsed: float,
        status: str,
        error: str,
        agent_role: str,
        response_meta: Optional[dict[str, Any]] = None,
        trace_id: str = "",
        agent_round: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        requested_max_tokens: int = 0,
    ) -> None:
        estimated_context = _estimate_prompt_tokens(messages, tools)
        context_tokens = int(usage.get("prompt_tokens", 0)) or estimated_context
        context_window = (
            self.context_window_tokens
            if model == self.sensenova_model
            else MODEL_CONTEXT_WINDOWS.get(model, 32768)
        )
        reserved_context = context_tokens + max(0, requested_max_tokens)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": self.module_name,
            "agent_role": agent_role or self.module_name,
            "trace_id": trace_id,
            "agent_round": agent_round,
            "run_id": self.run_id,
            "request_id": f"{self.run_id}:{self._call_sequence}",
            "call": self._call_sequence,
            "model": model,
            "account": account_index + 1 if account_index >= 0 else "agnes",
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "estimated_context_tokens": estimated_context,
            "context_tokens": context_tokens,
            "requested_max_tokens": requested_max_tokens,
            "reserved_context_tokens": reserved_context,
            "context_window_tokens": context_window,
            "prompt_context_utilization": round(context_tokens / context_window, 4),
            "context_utilization": round(reserved_context / context_window, 4),
            "usage_estimated": bool(usage.get("estimated", False)),
        }
        if error:
            record["error"] = error[:1000]
        if response_meta:
            record.update(response_meta)
        logger.info(
            "LLM call module=%s model=%s account=%s status=%s tokens=%s/%s/%s context=%s/%s elapsed=%.2fs",
            f"{self.module_name}:{record['agent_role']}",
            model,
            record["account"],
            status,
            record["prompt_tokens"],
            record["completion_tokens"],
            record["total_tokens"],
            context_tokens,
            context_window,
            elapsed,
        )
        if self.telemetry_path:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._totals["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._totals[key] += record[key]

    @staticmethod
    def _call_openai(
        base_url: str,
        model: str,
        api_key: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        proxy: Optional[str] = None,
        account_index: int = -1,
        request_attempts: int = 3,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        if extra_body:
            body.update(extra_body)

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        proxies = {"http": proxy, "https": proxy} if proxy else None
        url = base_url.rstrip("/") + "/chat/completions"
        last_error: Optional[Exception] = None
        started = time.perf_counter()
        for attempt in range(max(1, request_attempts)):
            try:
                response = requests.post(
                    url, headers=headers, json=body, proxies=proxies, timeout=180
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = RetryableError(str(exc))
                if attempt + 1 < request_attempts:
                    time.sleep(2**attempt)
                    continue
                raise last_error

            if response.status_code == 200:
                return _parse_response(response.json(), model, account_index, time.perf_counter() - started)

            error_body = response.text[:1000]
            if response.status_code == 429:
                if _looks_like_quota_error(error_body):
                    raise InsufficientQuota(f"HTTP 429: {error_body}")
                raise RateLimited(f"HTTP 429: {error_body}")
            if response.status_code == 400 and _looks_like_quota_error(error_body):
                raise InsufficientQuota(f"HTTP 400: {error_body}")
            if response.status_code in {401, 403}:
                raise InvalidCredentials(f"HTTP {response.status_code} on {model}: {error_body}")
            # SenseNova can return a short-lived, account-specific 404 even
            # while the same model is working on the other accounts.  Treat
            # only that explicit routing response as retryable so ``chat``
            # rotates to the next key.  A genuine bad endpoint/model 404
            # remains fatal instead of being hidden by endless rotation.
            if response.status_code == 404 and _looks_like_model_route_error(error_body):
                raise RetryableError(f"HTTP 404 on {model}: {error_body}")
            if response.status_code in {400, 404, 422}:
                raise FatalLLMError(f"HTTP {response.status_code} on {model}: {error_body}")
            if response.status_code >= 500:
                last_error = RetryableError(f"HTTP {response.status_code}: {error_body}")
                if attempt + 1 < request_attempts:
                    time.sleep(2**attempt)
                    continue
                raise last_error
            raise RuntimeError(f"Unexpected HTTP {response.status_code} on {model}: {error_body}")
        raise RetryableError(f"Request failed: {last_error}")

    def usage_summary(self) -> dict:
        return dict(self._totals)

    def log_summary(self) -> str:
        fallback = "agnes fallback" if self.allow_agnes_fallback and self._agnes_key else "no fallback"
        return f"LLMClient(model={self.sensenova_model}, keys={len(self._sensenova_keys)}, {fallback})"


def _parse_response(raw: dict, model: str, account_index: int, elapsed: float = 0.0) -> LLMResult:
    choice = (raw.get("choices") or [None])[0]
    if not choice:
        raise RuntimeError("No choices in response")
    message = choice.get("message", {})
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        tool_calls.append(ToolCall(call.get("id", ""), function.get("name", ""), arguments))
    return LLMResult(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        model=model,
        account_index=account_index,
        usage=raw.get("usage") or {},
        raw=raw,
        elapsed_seconds=elapsed,
    )


def _estimate_prompt_tokens(
    messages: list[dict], tools: Optional[list[dict]] = None
) -> int:
    payload: dict[str, Any] = {"messages": messages}
    if tools:
        payload["tools"] = tools
    prompt_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return max(1, (prompt_chars + 1) // 2)


def _normalise_usage(
    usage: dict,
    messages: list[dict],
    content: str,
    tools: Optional[list[dict]] = None,
) -> dict:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    estimated = prompt is None or completion is None
    if prompt is None:
        prompt = _estimate_prompt_tokens(messages, tools)
    if completion is None:
        completion = max(1, len(content) // 2) if content else 0
    if total is None:
        total = int(prompt) + int(completion)
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(total),
        "estimated": estimated,
    }


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        logger.warning("Key file not found: %s", path)
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_first_line(path: Path) -> Optional[str]:
    lines = _load_lines(path)
    return lines[0] if lines else None


def _looks_like_quota_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "insufficient_quota",
            "insufficient quota",
            "quota exceeded",
            "quota_exceeded",
            "out of quota",
            "billing",
            "balance",
            "credit",
        )
    )


def _looks_like_model_route_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "model route not found",
            "model_route_not_found",
        )
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (2**attempt))


class InsufficientQuota(Exception):
    """The account has no currently usable quota."""


class RateLimited(Exception):
    """The account hit a short rate limit."""


class RetryableError(Exception):
    """The request may succeed on another account."""


class FatalLLMError(Exception):
    """A request/model/authentication error that account rotation cannot fix."""


class InvalidCredentials(FatalLLMError):
    """One account rejected its credentials; other accounts may still work."""


class ContextWindowExceeded(FatalLLMError):
    """The estimated prompt plus reserved completion exceeds this model's window."""


class AllModelsExhausted(Exception):
    """No configured model/account can currently serve the request."""
