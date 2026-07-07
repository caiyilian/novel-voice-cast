"""Multi-key, multi-model LLM client for OpenAI-compatible APIs.

Supports:
  - Round-robin key rotation across N accounts
  - Primary → fallback model chain (deepseek-v4-flash → agnes-2.0-flash)
  - Token usage logging per call
  - Proxy support (Agnes relay)
  - Tool calling (OpenAI-compatible format)

Usage:
    client = LLMClient()
    result = client.chat(messages, tools=tool_specs)
    print(result.content, result.tool_calls, result.model, result.usage)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("llm_client")


# ── Data classes ───────────────────────────────────────────────────────

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
    account_index: int = -1  # -1 = fallback model (Agnes)
    usage: dict = field(default_factory=dict)  # {prompt_tokens, completion_tokens, total_tokens}
    raw: dict = field(default_factory=dict)


# ── Client ─────────────────────────────────────────────────────────────

SENSENOVA_BASE = "https://token.sensenova.cn/v1"
AGNES_BASE = "https://apihub.agnes-ai.com/v1"
AGNES_PROXY = os.environ.get("AGNES_PROXY_URL", "http://127.0.0.1:7890")
SENSENOVA_MODEL = "deepseek-v4-flash"
AGNES_MODEL = "agnes-2.0-flash"
SENSENOVA_QUOTA_COOLDOWN_SECONDS = int(os.environ.get("SENSENOVA_QUOTA_COOLDOWN_SECONDS", str(5 * 60 * 60)))
SENSENOVA_RETRY_COOLDOWN_SECONDS = int(os.environ.get("SENSENOVA_RETRY_COOLDOWN_SECONDS", "60"))
SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS", "15"))

KEY_FILE_SENSENOVA = Path("config/sensenova_apikeys")
KEY_FILE_AGNES = Path("config/agnes_api_key")


class LLMClient:
    """OpenAI-compatible LLM client with multi-key round-robin + fallback."""

    def __init__(self):
        self._sensenova_keys = _load_lines(KEY_FILE_SENSENOVA)
        self._agnes_key = _load_first_line(KEY_FILE_AGNES)
        self._round_robin_index = 0
        self._sensenova_disabled_until = [0.0 for _ in self._sensenova_keys]

    # ── Public API ─────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResult:
        """Send a chat completion request.

        Tries each SenseNova key in round-robin order.  If all are exhausted
        (quota / rate-limit) it falls back to Agnes 2.0 Flash.
        """
        # ── Primary chain: deepseek-v4-flash ──
        n_keys = len(self._sensenova_keys)
        for attempt in range(n_keys):
            idx = self._next_sensenova_index()
            if idx is None:
                break
            key = self._sensenova_keys[idx]
            try:
                return self._call_openai(
                    base_url=SENSENOVA_BASE,
                    model=SENSENOVA_MODEL,
                    api_key=key,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    proxy=None,
                    account_index=idx,
                )
            except InsufficientQuota as exc:
                self._disable_sensenova_key(idx, SENSENOVA_QUOTA_COOLDOWN_SECONDS)
                logger.warning(
                    "Quota unavailable for account %d; cooling down %ds: %s",
                    idx,
                    SENSENOVA_QUOTA_COOLDOWN_SECONDS,
                    exc,
                )
                continue
            except RateLimited as exc:
                self._disable_sensenova_key(idx, SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS)
                logger.warning(
                    "Rate limited for account %d; cooling down %ds: %s",
                    idx,
                    SENSENOVA_RATE_LIMIT_COOLDOWN_SECONDS,
                    exc,
                )
                continue
            except RetryableError:
                self._disable_sensenova_key(idx, SENSENOVA_RETRY_COOLDOWN_SECONDS)
                logger.warning("Retryable error on account %d, trying next", idx)
                continue
            except Exception as exc:
                # Catch-all: any non-specific error (connection drop, timeout,
                # unexpected response) — try the next key before giving up.
                self._disable_sensenova_key(idx, SENSENOVA_RETRY_COOLDOWN_SECONDS)
                logger.warning("Account %d failed: %s — trying next", idx, exc)
                continue

        # ── Fallback chain: Agnes 2.0 Flash ──
        if self._agnes_key:
            logger.info("All SenseNova accounts are in local cooldown, falling back to Agnes 2.0 Flash")
            return self._call_openai(
                base_url=AGNES_BASE,
                model=AGNES_MODEL,
                api_key=self._agnes_key,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                proxy=AGNES_PROXY,
                account_index=-1,
            )

        raise AllModelsExhausted("All SenseNova accounts exhausted and no Agnes key configured")

    def _next_sensenova_index(self) -> Optional[int]:
        n_keys = len(self._sensenova_keys)
        if n_keys <= 0:
            return None
        now = time.time()
        for _ in range(n_keys):
            idx = self._round_robin_index % n_keys
            self._round_robin_index += 1
            if self._sensenova_disabled_until[idx] <= now:
                return idx
        return None

    def _disable_sensenova_key(self, idx: int, seconds: int) -> None:
        if idx < 0 or idx >= len(self._sensenova_disabled_until):
            return
        self._sensenova_disabled_until[idx] = max(
            self._sensenova_disabled_until[idx],
            time.time() + max(0, seconds),
        )

    # ── Internal ───────────────────────────────────────────────────

    @staticmethod
    def _call_openai(
        base_url: str,
        model: str,
        api_key: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        proxy: Optional[str] = None,
        account_index: int = -1,
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

        # Build request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        proxies = {"http": proxy, "https": proxy} if proxy else None
        url = base_url.rstrip("/") + "/chat/completions"

        # Post with retries
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    proxies=proxies,
                    timeout=180,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout on %s (attempt %d)", model, attempt + 1)
                last_error = RetryableError("timeout")
                time.sleep(2 ** attempt)
                continue
            except requests.exceptions.ConnectionError as e:
                logger.warning("Connection error on %s: %s", model, e)
                last_error = RetryableError(str(e))
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                raw = resp.json()
                return _parse_response(raw, model, account_index)

            if resp.status_code == 429:
                err_body = resp.text[:500]
                if _looks_like_quota_error(err_body):
                    raise InsufficientQuota(f"HTTP 429: {err_body}")
                raise RateLimited(f"HTTP 429: {err_body}")

            if resp.status_code == 400:
                err_body = resp.text[:500]
                # Check if it's a quota error in the response body
                if _looks_like_quota_error(err_body):
                    raise InsufficientQuota(err_body)
                raise RuntimeError(f"HTTP 400 on {model}: {err_body}")

            if resp.status_code >= 500:
                logger.warning("HTTP %d on %s (attempt %d)", resp.status_code, model, attempt + 1)
                last_error = RetryableError(f"HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue

            raise RuntimeError(f"Unexpected HTTP {resp.status_code} on {model}: {resp.text[:300]}")

        raise RetryableError(f"Failed after 3 retries: {last_error}")

    def log_summary(self) -> str:
        """Return a human-readable summary of all calls made."""
        return f"LLMClient(keys={len(self._sensenova_keys)} sensenova + {'agnes' if self._agnes_key else 'no'} fallback)"


# ── Response parsing ───────────────────────────────────────────────────

def _parse_response(raw: dict, model: str, account_index: int) -> LLMResult:
    choice = (raw.get("choices") or [None])[0]
    if not choice:
        raise RuntimeError("No choices in response")

    msg = choice.get("message", {})
    content = msg.get("content") or ""
    usage = raw.get("usage") or {}

    # Parse tool calls
    raw_calls = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for tc in raw_calls:
        func = tc.get("function", {})
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args_raw = json.loads(args_raw)
            except json.JSONDecodeError:
                args_raw = {}
        tool_calls.append(ToolCall(
            id=tc.get("id", ""),
            name=func.get("name", ""),
            arguments=args_raw,
        ))

    return LLMResult(
        content=content,
        tool_calls=tool_calls,
        model=model,
        account_index=account_index,
        usage=usage,
        raw=raw,
    )


# ── File helpers ───────────────────────────────────────────────────────

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
    quota_markers = (
        "insufficient_quota",
        "insufficient quota",
        "quota exceeded",
        "quota_exceeded",
        "out of quota",
        "billing",
        "balance",
        "credit",
        "账户余额",
        "余额不足",
        "额度不足",
        "资源包",
        "欠费",
    )
    return any(marker in lowered for marker in quota_markers)


# ── Custom errors ──────────────────────────────────────────────────────

class InsufficientQuota(Exception):
    """Raised when the account's spend quota or balance is unavailable."""


class RateLimited(Exception):
    """Raised when an account hits a temporary rate limit."""


class RetryableError(Exception):
    """Raised when the request can be retried on another account."""


class AllModelsExhausted(Exception):
    """All primary and fallback models have been exhausted."""
