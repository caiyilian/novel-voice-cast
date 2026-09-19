"""Thin OpenAI-compatible LLM client for the SenseNova Pool local proxy.

All LLM stages (gender / emotion / performance / bgm) share one model:
``deepseek-v4-flash`` served by the local 20-account round-robin proxy at
``http://127.0.0.1:18787/v1``.  Multi-key rotation, per-account cooldowns and
the Agnes fallback have been removed — the proxy handles load balancing, so
this client is a single-endpoint, single-key wrapper around one POST.

The public surface (``LLMClient`` / ``LLMResult`` / ``ToolCall`` / the
exception classes / ``for_flash_lite``) is kept backward-compatible so the
four stage modules keep working unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
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


# --- 本地代理配置（非敏感：base_url 可提交；apiKey 从 config/llm_proxy_key 读） ---
PROXY_BASE_URL = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:18787/v1")
PROXY_API_KEY_ENV = "LLM_PROXY_API_KEY"
KEY_FILE_PROXY = Path("config/llm_proxy_key")

# 统一模型：所有 LLM 阶段都用 deepseek-v4-flash。
# SENSENOVA_FLASH_LITE_MODEL 保留旧名字以兼容各阶段模块的 import 与 checkpoint 校验，
# 但值统一为 deepseek-v4-flash，使历史「钉死 flash-lite」的引用自动对齐到新模型。
SENSENOVA_MODEL = "deepseek-v4-flash"
SENSENOVA_FLASH_LITE_MODEL = "deepseek-v4-flash"
DEFAULT_MODEL = SENSENOVA_MODEL

# deepseek-v4-flash 官方 context window = 1M。
DEFAULT_CONTEXT_WINDOW_TOKENS = 1024 * 1024
MODEL_CONTEXT_WINDOWS = {
    SENSENOVA_MODEL: DEFAULT_CONTEXT_WINDOW_TOKENS,
}

RATE_LIMIT_RETRY_SECONDS = 15.0
MAX_TRANSIENT_RETRIES = 3


class LLMClient:
    """Single-endpoint, single-key OpenAI-compatible client with telemetry."""

    def __init__(
        self,
        *,
        sensenova_model: str = DEFAULT_MODEL,
        module_name: str = "general",
        telemetry_path: Path | str | None = None,
        context_window_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.sensenova_model = sensenova_model
        self.module_name = module_name
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        inferred_context = MODEL_CONTEXT_WINDOWS.get(sensenova_model, DEFAULT_CONTEXT_WINDOW_TOKENS)
        self.context_window_tokens = max(1, int(context_window_tokens or inferred_context))
        self.base_url = (base_url or PROXY_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else _load_api_key()
        self._sleep = sleep_fn
        # 兼容旧属性：单账号代理下不存在 fallback / 等待轮询。
        self.allow_agnes_fallback = False
        self.wait_for_sensenova = True
        self._call_sequence = 0
        self.run_id = uuid.uuid4().hex
        self._totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @classmethod
    def for_flash_lite(
        cls,
        module_name: str,
        telemetry_path: Path | str | None = None,
        **kwargs: Any,
    ) -> "LLMClient":
        """Build the client used by gender / emotion / performance / BGM.

        Historical callers pass ``sensenova_keys`` / ``agnes_key`` /
        ``quota_state_path``; those are ignored (single proxy account now).
        ``context_window_tokens`` / ``api_key`` / ``base_url`` still pass through.
        """
        context_window_tokens = kwargs.pop("context_window_tokens", None)
        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        return cls(
            sensenova_model=SENSENOVA_FLASH_LITE_MODEL,
            module_name=module_name,
            telemetry_path=telemetry_path or Path("logs") / f"{module_name}_llm_calls.jsonl",
            context_window_tokens=context_window_tokens,
            api_key=api_key,
            base_url=base_url,
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
        """Send one completion, retrying transient / rate-limit failures."""
        self._ensure_context_fits(
            self.sensenova_model,
            messages,
            tools,
            max_tokens,
            self.context_window_tokens,
        )
        if not self.api_key:
            raise AllModelsExhausted("No LLM proxy API key configured")

        last_error: Optional[Exception] = None
        for attempt in range(MAX_TRANSIENT_RETRIES):
            try:
                return self._request_and_record(
                    base_url=self.base_url,
                    model=self.sensenova_model,
                    api_key=self.api_key,
                    account_index=0,
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
            except RateLimited as exc:
                last_error = exc
                logger.warning(
                    "Rate limited; retrying in %.0fs (attempt %d/%d): %s",
                    RATE_LIMIT_RETRY_SECONDS, attempt + 1, MAX_TRANSIENT_RETRIES, exc,
                )
                self._sleep(RATE_LIMIT_RETRY_SECONDS)
            except RetryableError as exc:
                last_error = exc
                wait = float(2 ** attempt)
                logger.warning(
                    "Transient failure; retrying in %.1fs (attempt %d/%d): %s",
                    wait, attempt + 1, MAX_TRANSIENT_RETRIES, exc,
                )
                self._sleep(wait)
        raise last_error if last_error is not None else AllModelsExhausted("LLM request failed")

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
        context_window = self.context_window_tokens
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
                    time.sleep(2 ** attempt)
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
            if response.status_code == 404 and _looks_like_model_route_error(error_body):
                raise RetryableError(f"HTTP 404 on {model}: {error_body}")
            if response.status_code in {400, 404, 422}:
                raise FatalLLMError(f"HTTP {response.status_code} on {model}: {error_body}")
            if response.status_code >= 500:
                last_error = RetryableError(f"HTTP {response.status_code}: {error_body}")
                if attempt + 1 < request_attempts:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error
            raise RuntimeError(f"Unexpected HTTP {response.status_code} on {model}: {error_body}")
        raise RetryableError(f"Request failed: {last_error}")

    def usage_summary(self) -> dict:
        return dict(self._totals)

    def log_summary(self) -> str:
        return (
            f"LLMClient(model={self.sensenova_model}, endpoint={self.base_url}, "
            f"single proxy key)"
        )


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


def _load_api_key() -> Optional[str]:
    env_key = os.environ.get(PROXY_API_KEY_ENV)
    if env_key:
        return env_key
    if KEY_FILE_PROXY.exists():
        lines = KEY_FILE_PROXY.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            return lines[0].strip()
    logger.warning("LLM proxy key not found (env %s or %s)", PROXY_API_KEY_ENV, KEY_FILE_PROXY)
    return None


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


class InsufficientQuota(Exception):
    """The account has no currently usable quota."""


class RateLimited(Exception):
    """The account hit a short rate limit."""


class RetryableError(Exception):
    """The request may succeed on retry."""


class FatalLLMError(Exception):
    """A request/model/authentication error that retrying cannot fix."""


class InvalidCredentials(FatalLLMError):
    """Credentials rejected."""


class ContextWindowExceeded(FatalLLMError):
    """The estimated prompt plus reserved completion exceeds this model's window."""


class AllModelsExhausted(Exception):
    """No configured model can currently serve the request."""
