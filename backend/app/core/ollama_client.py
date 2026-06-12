import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _load_config() -> dict:
    config = {}
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "ip_config"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = ""
    model: str = ""
    timeout: int = 120
    retries: int = 2
    retry_delay: float = 5.0

    @classmethod
    def from_ip_config(cls) -> "OllamaConfig":
        raw = _load_config()
        return cls(
            base_url=raw.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=raw.get("OLLAMA_MODEL", "qwen3:4b"),
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatResult:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionStatus:
    ok: bool
    message: str = ""
    model: str = ""


class OllamaClient:
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig.from_ip_config()

    @property
    def _chat_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def chat(self, messages: list, tools: Optional[list] = None,
             temperature: float = 0.3) -> ChatResult:
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools

        # ensure tool_call arguments are JSON strings (Ollama requirement)
        for msg in body["messages"]:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    if isinstance(func.get("arguments"), dict):
                        func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)

        raw = self._post(body)
        choices = raw.get("choices", [])
        if not choices:
            raise OllamaError("No choices in response")

        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = self._parse_tool_calls(msg.get("tool_calls"))
        return ChatResult(content=content, tool_calls=tool_calls, raw=raw)

    def check_connection(self) -> ConnectionStatus:
        try:
            result = self.chat([{"role": "user", "content": "ping"}])
            return ConnectionStatus(ok=True, message="connected", model=self.config.model)
        except Exception as e:
            return ConnectionStatus(ok=False, message=str(e), model=self.config.model)

    def _post(self, body: dict) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.base_url}",
            },
            method="POST",
        )

        last_error = None
        for attempt in range(1, self.config.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                last_error = e
                if attempt < self.config.retries:
                    time.sleep(self.config.retry_delay)
            except Exception as e:
                last_error = e
                if attempt < self.config.retries:
                    time.sleep(self.config.retry_delay)

        raise OllamaError(f"Request failed after {self.config.retries} retries: {last_error}")

    @staticmethod
    def _parse_tool_calls(raw_calls: Optional[list]) -> list:
        if not raw_calls:
            return []
        result = []
        for tc in raw_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result.append(ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            ))
        return result


class OllamaError(Exception):
    pass
