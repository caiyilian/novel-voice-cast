import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.emotion_labeler import (  # noqa: E402
    EMOTION_PIPELINE_VERSION,
    REVIEW_TOOL,
    TOOL_SPECS,
    emotion_source_hash,
    label_all_emotions,
    label_emotion,
)
from app.core.llm_client import LLMResult, SENSENOVA_FLASH_LITE_MODEL, ToolCall  # noqa: E402


class ScriptedClient:
    def __init__(self, responses, initial_calls=0):
        self.responses = list(responses)
        self.calls = initial_calls
        self.messages = []

    def usage_summary(self):
        return {"calls": self.calls, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.messages.append(json.loads(json.dumps(messages, ensure_ascii=False)))
        return self.responses.pop(0)


def emotion_call(index, emotion, tone, confidence, review=False):
    name = "submit_emotion_review" if review else "submit_emotion"
    return LLMResult(tool_calls=[ToolCall(name, name, {
        "emotion": emotion,
        "tone": tone,
        "confidence": confidence,
        "evidence": "line 2 narration cue",
        "evidence_lines": [2],
    })])


def _contains_key(value, key):
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_sensenova_emotion_schemas_avoid_unsupported_unique_items():
    """SenseNova rejects strict tool schemas containing the uniqueItems keyword."""
    assert not _contains_key(TOOL_SPECS, "uniqueItems")
    assert not _contains_key(REVIEW_TOOL, "uniqueItems")


def test_emotion_disagreement_uses_third_agent():
    client = ScriptedClient([
        emotion_call(0, "happy", "loud", 0.8),
        emotion_call(0, "angry", "serious", 0.8, review=True),
        emotion_call(0, "angry", "serious", 0.9, review=True),
    ])

    result = label_emotion("Stop.", 2, 0, "He frowned.\nStop.\nThe room fell silent.", client=client, speaker="A")

    assert (result["emotion"], result["tone"]) == ("angry", "serious")
    assert result["adjudicated"] is True
    assert result["agent_calls"] == 3
    assert result["source_line"] == 2
    assert "Candidate decisions" not in client.messages[1][1]["content"]
    assert "Index 0" not in client.messages[1][1]["content"]


def test_emotion_checkpoint_skips_completed_and_narration(tmp_path):
    checkpoint = tmp_path / "emotion.json"
    dialogues = [
        {"line": 1, "speaker": "\u65c1\u767d", "text": "Narration"},
        {"line": 2, "speaker": "A", "text": "Hello"},
    ]
    text = "Narration\nHello"
    checkpoint.write_text(json.dumps({
        "pipeline_version": EMOTION_PIPELINE_VERSION,
        "source_hash": emotion_source_hash(text, dialogues),
        "model": SENSENOVA_FLASH_LITE_MODEL,
        "llm_usage": {"calls": 7, "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "results": {"1": {
        "dialogue_index": 1,
        "source_line": 2,
        "emotion": "calm",
        "tone": "soft",
        "confidence": 0.9,
        "evidence": "line 2",
        "evidence_lines": [2],
    }}, "errors": {}}), encoding="utf-8")
    client = ScriptedClient([])

    results = label_all_emotions(dialogues, text, client=client, checkpoint_path=checkpoint)

    assert list(results) == ["1"]
    assert client.calls == 0


def test_old_checkpoint_is_not_reused(tmp_path):
    checkpoint = tmp_path / "emotion.json"
    checkpoint.write_text(json.dumps({"results": {"0": {"emotion": "cold"}}, "errors": {}}), encoding="utf-8")
    dialogues = [{"line": 1, "speaker": "A", "text": "Hello"}]
    client = ScriptedClient([
        LLMResult(tool_calls=[ToolCall("p", "submit_emotion", {
            "emotion": "calm", "tone": "serious", "confidence": 0.8,
            "evidence": "source line 1 is a direct greeting", "evidence_lines": [1],
        })]),
        LLMResult(tool_calls=[ToolCall("r", "submit_emotion_review", {
            "emotion": "calm", "tone": "serious", "confidence": 0.9,
            "evidence": "source line 1 has no intensity cue", "evidence_lines": [1],
        })]),
    ])

    results = label_all_emotions(dialogues, "Hello", client=client, checkpoint_path=checkpoint)

    assert results["0"]["emotion"] == "calm"
    assert client.calls == 2


def test_evidence_must_include_target_source_line():
    client = ScriptedClient([
        LLMResult(tool_calls=[ToolCall("bad", "submit_emotion", {
            "emotion": "calm", "tone": "serious", "confidence": 0.8,
            "evidence": "only cites the neighboring narration", "evidence_lines": [1],
        })]),
        LLMResult(tool_calls=[ToolCall("fixed", "submit_emotion", {
            "emotion": "calm", "tone": "serious", "confidence": 0.8,
            "evidence": "source line 2 is direct and unmarked", "evidence_lines": [2],
        })]),
        LLMResult(tool_calls=[ToolCall("review", "submit_emotion_review", {
            "emotion": "calm", "tone": "serious", "confidence": 0.9,
            "evidence": "source line 2 has no intensity cue", "evidence_lines": [2],
        })]),
    ])

    result = label_emotion("Hello", 2, 0, "Narration\nHello", client=client, speaker="A")

    assert result["evidence_lines"] == [2]
    assert result["agent_calls"] == 3
    assert "must include target source line 2" in client.messages[1][-1]["content"]


def test_emotion_checkpoint_usage_excludes_shared_client_history(tmp_path):
    checkpoint = tmp_path / "emotion.json"
    dialogues = [{"line": 1, "speaker": "A", "text": "Hello"}]
    client = ScriptedClient(
        [
            LLMResult(tool_calls=[ToolCall("p", "submit_emotion", {
                "emotion": "calm", "tone": "serious", "confidence": 0.8,
                "evidence": "source line 1 is a direct greeting", "evidence_lines": [1],
            })]),
            LLMResult(tool_calls=[ToolCall("r", "submit_emotion_review", {
                "emotion": "calm", "tone": "serious", "confidence": 0.9,
                "evidence": "source line 1 has no intensity cue", "evidence_lines": [1],
            })]),
        ],
        initial_calls=100,
    )

    label_all_emotions(dialogues, "Hello", client=client, checkpoint_path=checkpoint)

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["llm_usage"]["calls"] == 2
