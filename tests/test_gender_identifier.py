import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.gender_identifier import (  # noqa: E402
    GENDER_PIPELINE_VERSION,
    GenderBatchError,
    gender_source_hash,
    identify_all_genders,
    identify_gender,
)
from app.core.llm_client import LLMResult, SENSENOVA_FLASH_LITE_MODEL, ToolCall  # noqa: E402


class ScriptedClient:
    def __init__(self, responses, initial_calls=0):
        self.responses = list(responses)
        self.calls = initial_calls

    def usage_summary(self):
        return {"calls": self.calls, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def gender_call(name, gender, confidence, evidence, review=False):
    tool = "submit_gender_review" if review else "submit_gender"
    arguments = {"gender": gender, "confidence": confidence, "evidence": evidence}
    if review:
        arguments["reasoning_summary"] = "checked explicit wording"
    else:
        arguments["character_name"] = name
    return LLMResult(tool_calls=[ToolCall(f"call-{tool}", tool, arguments)])


def test_unknown_is_preserved_and_every_result_is_reviewed():
    client = ScriptedClient([
        gender_call("Cloud", "unknown", 0.4, "line 1 has no gender cue"),
        gender_call("Cloud", "unknown", 0.8, "line 1 remains ambiguous", review=True),
    ])

    result = identify_gender("Cloud", "Cloud appeared.\nNo pronoun was used.", client=client)

    assert result["gender"] == "unknown"
    assert result["verification"]["reviewed"] is True
    assert result["agent_calls"] == 2


def test_gender_checkpoint_resumes_without_new_calls(tmp_path):
    checkpoint = tmp_path / "gender.json"
    checkpoint.write_text(json.dumps({
        "pipeline_version": GENDER_PIPELINE_VERSION,
        "source_hash": gender_source_hash("A", ["A"]),
        "model": SENSENOVA_FLASH_LITE_MODEL,
        "llm_usage": {"calls": 2, "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        "results": [{
        "character_name": "A",
        "gender": "female",
        "confidence": 0.9,
        "evidence": "line 2 explicit pronoun",
    }]}), encoding="utf-8")
    client = ScriptedClient([])

    results = identify_all_genders(["A"], "A", client=client, checkpoint_path=checkpoint)

    assert results[0]["gender"] == "female"
    assert client.calls == 0


def test_gender_failure_is_not_checkpointed_as_completed(tmp_path):
    checkpoint = tmp_path / "gender.json"
    client = ScriptedClient([])

    with pytest.raises(GenderBatchError):
        identify_all_genders(["A"], "A", client=client, checkpoint_path=checkpoint, item_retries=2)

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["results"] == []
    assert "A" in payload["errors"]


def test_gender_checkpoint_usage_excludes_shared_client_history(tmp_path):
    checkpoint = tmp_path / "gender.json"
    client = ScriptedClient(
        [
            gender_call("A", "female", 0.9, "line 1 explicit pronoun"),
            gender_call("A", "female", 0.9, "line 1 confirms pronoun", review=True),
        ],
        initial_calls=100,
    )

    identify_all_genders(["A"], "A", client=client, checkpoint_path=checkpoint)

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["llm_usage"]["calls"] == 2
