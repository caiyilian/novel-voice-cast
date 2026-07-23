import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.llm_client import LLMResult, ToolCall  # noqa: E402
from app.core.visual_prompt_auditor import (  # noqa: E402
    VISUAL_PROMPT_PIPELINE_VERSION,
    audit_visual_prompt,
    audit_visual_prompts,
)


class ScriptedClient:
    sensenova_model = "sensenova-6.7-flash-lite"

    def __init__(self, responses, initial_calls=0):
        self.responses = list(responses)
        self.calls = []
        self.usage = {
            "calls": initial_calls,
            "prompt_tokens": initial_calls * 10,
            "completion_tokens": initial_calls * 2,
            "total_tokens": initial_calls * 12,
        }

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 10
        self.usage["completion_tokens"] += 2
        self.usage["total_tokens"] += 12
        if not self.responses:
            raise AssertionError("ScriptedClient received an unexpected chat call")
        tool_name, arguments = self.responses.pop(0)
        return LLMResult(
            tool_calls=[ToolCall(f"call_{len(self.calls)}", tool_name, arguments)],
            model=self.sensenova_model,
        )

    def usage_summary(self):
        return dict(self.usage)


def _plan(prompt="wind-bent wheat", *, start_line=1, end_line=1):
    return {
        "title": "麦浪",
        "description": "风吹麦田",
        "reason": "visible landscape beat",
        "characters": [],
        "composition": "wide shot",
        "prompt": prompt,
        "start_line": start_line,
        "end_line": end_line,
    }


def _candidate(
    prompt,
    *,
    lines=(1,),
    excluded=(),
    material=False,
    verdict=None,
):
    value = {
        "audited_prompt": prompt,
        "evidence_lines": list(lines),
        "excluded_nonliteral_entities": list(excluded),
        "retained_characters": [],
        "literal_entity_evidence": [
            {"entity": "wheat", "source": "novel", "evidence_lines": list(lines)},
            {"entity": "wind", "source": "novel", "evidence_lines": list(lines)},
        ],
        "material_changes": material,
        "rationale": "The source contains wheat and wind; the animal is only a comparison.",
    }
    if verdict is not None:
        value["verdict"] = verdict
    return value


def test_wheat_simile_never_materializes_wolves_and_uses_adjudication():
    text = "The wheat rolled like wolves beneath the hard wind."
    primary = _candidate(
        "wide wheat field bending in hard wind, rhythmic rolling motion, low golden light",
        excluded=("wolf", "wolves"),
        material=True,
    )
    review = _candidate(
        "wide wheat field swept by strong wind, layered wave motion, low golden light",
        excluded=("wolf", "wolves"),
        material=True,
        verdict="revise",
    )
    final = _candidate(
        "wide wheat field swept by strong wind, layered wave motion, low golden light",
        excluded=("wolf", "wolves"),
        material=True,
    )
    client = ScriptedClient([
        ("submit_visual_prompt_rewrite", primary),
        ("submit_visual_prompt_review", review),
        ("submit_visual_prompt_adjudication", final),
    ])

    result = audit_visual_prompt(
        _plan("a wheat field rolling like a pack of wolves in the wind"),
        text,
        client=client,
        illustration_index=7,
    )

    prompt = result["audited_prompt"].lower()
    assert "wolf" not in prompt
    assert "wolves" not in prompt
    assert "wheat" in prompt
    assert "wind" in prompt
    assert result["decision_path"] == "final_adjudication"
    assert [step["stage"] for step in result["decision_chain"]] == [
        "primary_rewrite",
        "independent_review",
        "final_adjudication",
    ]
    assert result["agent_calls"] == 3
    assert {call["trace_id"] for call in client.calls} == {"illustration_prompt:7"}
    assert [call["agent_round"] for call in client.calls] == [1, 2, 3]
    assert all(call["max_tokens"] == 8192 for call in client.calls)
    assert all(call["tools"][0]["function"]["strict"] is True for call in client.calls)
    assert all(call["tool_choice"]["function"]["name"] == call["tools"][0]["function"]["name"] for call in client.calls)


def test_checkpoint_resume_and_input_change_invalidation(tmp_path):
    checkpoint = tmp_path / "visual_prompt.checkpoint.json"
    prompt = "wind-bent wheat"
    first_client = ScriptedClient([
        ("submit_visual_prompt_rewrite", _candidate(prompt)),
        ("submit_visual_prompt_review", _candidate(prompt, verdict="approve")),
    ])

    first = audit_visual_prompts(
        [_plan(prompt)],
        "Wheat bends in the wind.",
        client=first_client,
        checkpoint_path=checkpoint,
    )

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["pipeline_version"] == VISUAL_PROMPT_PIPELINE_VERSION
    assert payload["completed_indices"] == [0]
    assert payload["results"] == first
    assert payload["llm_usage"] == {
        "calls": 2,
        "prompt_tokens": 20,
        "completion_tokens": 4,
        "total_tokens": 24,
    }
    assert not list(tmp_path.glob("*.tmp"))

    resumed_client = ScriptedClient([])
    resumed = audit_visual_prompts(
        [_plan(prompt)],
        "Wheat bends in the wind.",
        client=resumed_client,
        checkpoint_path=checkpoint,
    )
    assert resumed == first
    assert resumed_client.calls == []

    changed_client = ScriptedClient([
        ("submit_visual_prompt_rewrite", _candidate(prompt)),
        ("submit_visual_prompt_review", _candidate(prompt, verdict="approve")),
    ])
    changed = audit_visual_prompts(
        [_plan(prompt)],
        "Wheat bends sharply in the wind.",
        client=changed_client,
        checkpoint_path=checkpoint,
    )

    assert len(changed_client.calls) == 2
    assert changed[0]["audited_prompt"] == prompt
    changed_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert changed_payload["source_hash"] != payload["source_hash"]


def test_checkpoint_usage_uses_client_delta_not_shared_history(tmp_path):
    checkpoint = tmp_path / "usage.checkpoint.json"
    prompt = "wind-bent wheat"
    client = ScriptedClient(
        [
            ("submit_visual_prompt_rewrite", _candidate(prompt)),
            ("submit_visual_prompt_review", _candidate(prompt, verdict="approve")),
        ],
        initial_calls=100,
    )

    audit_visual_prompts(
        [_plan(prompt)],
        "Wheat bends in the wind.",
        client=client,
        checkpoint_path=checkpoint,
    )

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["llm_usage"]["calls"] == 2
    assert payload["llm_usage"]["total_tokens"] == 24


def test_invalid_evidence_line_is_rejected_then_retried():
    prompt = "wind-bent wheat"
    invalid = _candidate(prompt, lines=(2,))
    valid = _candidate(prompt, lines=(1,))
    client = ScriptedClient([
        ("submit_visual_prompt_rewrite", invalid),
        ("submit_visual_prompt_rewrite", valid),
        ("submit_visual_prompt_review", _candidate(prompt, verdict="approve")),
    ])

    result = audit_visual_prompt(_plan(prompt), "Wheat bends in the wind.", client=client)

    assert result["evidence_lines"] == [1]
    assert result["decision_path"] == "independent_agreement"
    assert result["agent_calls"] == 3
    assert [call["agent_round"] for call in client.calls] == [1, 2, 3]
    assert "invalid: [2]" in client.calls[1]["messages"][-1]["content"]


def test_character_card_can_add_stated_appearance_but_character_must_remain():
    plan = _plan("Agnes standing in wheat", start_line=1, end_line=1)
    plan["characters"] = ["Agnes"]
    value = _candidate("Agnes with explicitly carded silver hair standing beside wind-bent wheat")
    value["retained_characters"] = ["Agnes"]
    value["literal_entity_evidence"].append(
        {"entity": "silver hair", "source": "character_card", "evidence_lines": []}
    )
    review = dict(value)
    review["verdict"] = "approve"
    client = ScriptedClient([
        ("submit_visual_prompt_rewrite", value),
        ("submit_visual_prompt_review", review),
        ("submit_visual_prompt_adjudication", value),
    ])

    result = audit_visual_prompt(
        plan,
        "Agnes stands beside wheat in the wind.",
        character_cards_text="Agnes: silver hair",
        client=client,
    )

    assert "Agnes" in result["audited_prompt"]
    assert result["retained_characters"] == ["Agnes"]


@pytest.mark.parametrize("bad_line", [0, 2, 99])
def test_evidence_line_validation_exhaustion_raises(bad_line):
    invalid = _candidate("wind-bent wheat", lines=(bad_line,))
    client = ScriptedClient([
        ("submit_visual_prompt_rewrite", invalid),
        ("submit_visual_prompt_rewrite", invalid),
        ("submit_visual_prompt_rewrite", invalid),
    ])

    with pytest.raises(Exception, match="evidence"):
        audit_visual_prompt(_plan(), "Wheat bends in the wind.", client=client)
