import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core import llm_client  # noqa: E402
from app.core.llm_client import InsufficientQuota, LLMClient, RateLimited  # noqa: E402


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
