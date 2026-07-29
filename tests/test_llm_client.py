import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core import llm_client  # noqa: E402
from app.core.llm_client import (  # noqa: E402
    ContextWindowExceeded,
    InsufficientQuota,
    InvalidCredentials,
    LLMClient,
    LLMResult,
    RateLimited,
    RetryableError,
)
from app.core.ollama_client import OllamaClient, OllamaConfig, OllamaError  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def json(self):
        return {}


def test_429_rate_limit_is_not_treated_as_quota(monkeypatch):
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(429, '{"error":{"message":"rate limit exceeded"}}'),
    )

    with pytest.raises(RateLimited):
        LLMClient._call_openai(
            base_url="https://example.test/v1",
            model="deepseek-v4-flash",
            api_key="key",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_429_balance_error_is_treated_as_quota(monkeypatch):
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(429, '{"error":{"message":"insufficient quota"}}'),
    )

    with pytest.raises(InsufficientQuota):
        LLMClient._call_openai(
            base_url="https://example.test/v1",
            model="deepseek-v4-flash",
            api_key="key",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_404_model_route_not_found_is_retryable(monkeypatch):
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            404,
            '{"error":{"message":"model route not found","type":"not_found_error"}}',
        ),
    )

    with pytest.raises(RetryableError, match="model route not found"):
        LLMClient._call_openai(
            base_url="https://example.test/v1",
            model="sensenova-6.7-flash-lite",
            api_key="key",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_unrelated_404_remains_fatal(monkeypatch):
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(404, '{"error":{"message":"unknown endpoint"}}'),
    )

    with pytest.raises(llm_client.FatalLLMError, match="unknown endpoint"):
        LLMClient._call_openai(
            base_url="https://example.test/v1",
            model="sensenova-6.7-flash-lite",
            api_key="key",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_flash_lite_mode_has_no_fallback_and_writes_usage(tmp_path, monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return LLMResult(
            content="ok",
            model=kwargs["model"],
            account_index=kwargs["account_index"],
            usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        )

    monkeypatch.setattr(LLMClient, "_call_openai", staticmethod(fake_call))
    telemetry = tmp_path / "calls.jsonl"
    client = LLMClient.for_flash_lite(
        "test",
        telemetry,
        sensenova_keys=["a", "b"],
        agnes_key="unused",
        quota_state_path=None,
    )

    result = client.chat(
        [{"role": "user", "content": "hello"}],
        agent_role="verifier",
        trace_id="emotion:17:line:23",
        agent_round=2,
    )

    assert result.model == "sensenova-6.7-flash-lite"
    assert client.context_window_tokens == 262144
    assert calls[0]["account_index"] == 0
    assert client.allow_agnes_fallback is False
    record = llm_client.json.loads(telemetry.read_text(encoding="utf-8"))
    assert record["prompt_tokens"] == 12
    assert record["total_tokens"] == 15
    assert record["account"] == 1
    assert record["agent_role"] == "verifier"
    assert record["trace_id"] == "emotion:17:line:23"
    assert record["agent_round"] == 2
    assert record["run_id"]
    assert record["request_id"].startswith(record["run_id"] + ":")
    assert record["reserved_context_tokens"] == 12 + 4096
    assert record["context_utilization"] > record["prompt_context_utilization"]


def test_rate_limited_account_rotates_to_next_key(tmp_path, monkeypatch):
    attempts = []

    def fake_call(**kwargs):
        attempts.append(kwargs["account_index"])
        if kwargs["account_index"] == 0:
            raise RateLimited("short limit")
        return LLMResult(
            content="ok",
            model=kwargs["model"],
            account_index=kwargs["account_index"],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(LLMClient, "_call_openai", staticmethod(fake_call))
    client = LLMClient.for_flash_lite(
        "rotation",
        tmp_path / "calls.jsonl",
        sensenova_keys=["a", "b"],
        quota_state_path=None,
    )

    result = client.chat([{"role": "user", "content": "hello"}])

    assert attempts == [0, 1]
    assert result.account_index == 1


def test_invalid_credentials_disable_only_one_account(tmp_path, monkeypatch):
    attempts = []

    def fake_call(**kwargs):
        attempts.append(kwargs["account_index"])
        if kwargs["account_index"] == 0:
            raise InvalidCredentials("bad key")
        return LLMResult(
            content="ok",
            model=kwargs["model"],
            account_index=kwargs["account_index"],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(LLMClient, "_call_openai", staticmethod(fake_call))
    client = LLMClient.for_flash_lite(
        "credentials",
        tmp_path / "calls.jsonl",
        sensenova_keys=["bad", "good"],
        quota_state_path=None,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).account_index == 1
    assert attempts == [0, 1]


def test_flash_lite_rejects_prompt_plus_completion_over_its_own_window():
    client = LLMClient.for_flash_lite(
        "context",
        sensenova_keys=["key"],
        quota_state_path=None,
        context_window_tokens=100,
    )

    with pytest.raises(ContextWindowExceeded, match="100-token context window"):
        client.chat([{"role": "user", "content": "x" * 200}], max_tokens=20)


def test_ollama_qwen_uses_independent_40k_context_budget(monkeypatch):
    config = OllamaConfig(base_url="http://ollama.test", model="qwen3:32b", context_window_tokens=40000)
    client = OllamaClient(config)
    captured = {}

    def fake_post(body):
        captured.update(body)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    monkeypatch.setattr(client, "_post", fake_post)
    result = client.chat([{"role": "user", "content": "hello"}], max_tokens=2000)

    assert captured["options"]["num_ctx"] == 40000
    assert captured["max_tokens"] == 2000
    assert result.usage["context_window_tokens"] == 40000

    with pytest.raises(OllamaError, match="40000-token context window"):
        client.chat([{"role": "user", "content": "x" * 80000}], max_tokens=1)
