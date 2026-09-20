"""Shared helpers reused across the multi-agent review stages.

The four LLM stages (gender / emotion / performance / bgm) each duplicated the
same small utility functions.  This module centralises the canonical (most
defensive) versions so a bug fix or hardening lands in one place.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON (mkdir + tmp file + replace, retry on PermissionError)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))


def normalise_usage_summary(raw: Any) -> dict[str, int]:
    """Normalise a usage summary to non-negative ints for the standard keys."""
    if not isinstance(raw, dict):
        raw = {}
    usage: dict[str, int] = {}
    for key in USAGE_KEYS:
        try:
            usage[key] = max(0, int(raw.get(key, 0) or 0))
        except (TypeError, ValueError):
            usage[key] = 0
    return usage


def usage_delta(current: Any, starting: Any) -> dict[str, int]:
    """Element-wise difference of two usage summaries, clamped at zero."""
    current_usage = normalise_usage_summary(current)
    starting_usage = normalise_usage_summary(starting)
    return {key: max(0, current_usage[key] - starting_usage[key]) for key in USAGE_KEYS}


def merge_usage_summaries(left: Any, right: Any) -> dict[str, int]:
    """Element-wise sum of two usage summaries."""
    left_usage = normalise_usage_summary(left)
    right_usage = normalise_usage_summary(right)
    return {key: left_usage[key] + right_usage[key] for key in USAGE_KEYS}


def cumulative_usage(client: Any, previous: Any, starting: Any) -> dict[str, int]:
    """Previous usage plus this run's delta (via client.usage_summary())."""
    return merge_usage_summaries(previous, usage_delta(client.usage_summary(), starting))


def assistant_tool_message(result: Any) -> dict[str, Any]:
    """Build an assistant message carrying tool_calls for OpenAI-compatible APIs."""
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in result.tool_calls
        ],
    }


def estimate_prompt_tokens(messages: list[dict], tools: Optional[list[dict]] = None) -> int:
    """Cheap prompt-token estimate (chars/2) for context-window checks."""
    payload: dict[str, Any] = {"messages": messages}
    if tools:
        payload["tools"] = tools
    prompt_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return max(1, (prompt_chars + 1) // 2)
