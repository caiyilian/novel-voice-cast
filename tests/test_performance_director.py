import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.llm_client import LLMResult, ToolCall
from app.core.performance_director import (
    PERFORMANCE_DIRECTION_PIPELINE_VERSION,
    PERFORMANCE_PROMPT_SIGNATURE,
    PerformanceBatchError,
    build_performance_profiles,
    direct_all_performances,
    direct_character_profile,
    direct_line_performance,
    performance_direction_source_hash,
    validate_performance_payload,
)


NOVEL = "\n".join(
    [
        "罗伦斯看见追兵逼近，呼吸一紧。",
        "罗伦斯喊道：「听着，出口就在前面。抓紧我的手。」",
        "他说得很快，却始终压着音量。",
        "赫萝没有回头，只是用力点了点头。",
        "罗伦斯确认她跟上后喊道：「跟紧我，别回头。」",
    ]
)

DIALOGUES = [
    {
        "speaker": "罗伦斯",
        "text": "听着，出口就在前面。抓紧我的手。",
        "line": 2,
        "chapter": "one",
    },
    {
        "speaker": "罗伦斯",
        "text": "跟紧我，别回头。",
        "line": 5,
        "chapter": "one",
    },
]


class FakeClient:
    sensenova_model = "sensenova-6.7-flash-lite"

    def __init__(self, responses=None, initial_usage=None):
        self.responses = list(responses or [])
        self.calls = []
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if initial_usage:
            self.usage.update(initial_usage)

    def chat(self, messages, **kwargs):
        if not self.responses:
            raise AssertionError(f"unexpected LLM call after: {messages[-1]['content']}")
        name, arguments = self.responses.pop(0)
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 100
        self.usage["completion_tokens"] += 50
        self.usage["total_tokens"] += 150
        return LLMResult(tool_calls=[ToolCall(f"call-{self.usage['calls']}", name, copy.deepcopy(arguments))])

    def usage_summary(self):
        return dict(self.usage)


def profile_candidate(summary="克制而敏锐的旅行商人，说话务实清楚；危险时短句果断，但不丢失思考感。"):
    return {
        "character_name": "罗伦斯",
        "narrative_role": "以现实判断推进情节的旅行商人和主要视角人物",
        "stable_personality": "谨慎、务实，受惊时仍会努力维持判断力",
        "baseline_delivery": "日常表达克制直接，危险中会明显提速但保持清晰",
        "diction_and_rhythm": "措辞实用，先判断再行动，关键时使用短句",
        "emotional_range": "外层常保持镇定，压力升高时呼吸和停顿先暴露动摇",
        "relationship_dynamics": "面对赫萝既保护又尊重，不把她当成被动跟随者",
        "acting_constraints": ["不演成夸张英雄腔", "紧张时仍保持商人的思考感"],
        "profile_summary": summary,
        "evidence": "第2行表现危险中的行动指令，第3行明确快而压低音量。",
        "evidence_lines": [2, 3],
        "evidence_quotes": [
            {
                "line": 2,
                "quote": "听着，出口就在前面。抓紧我的手。",
                "supports": [
                    "narrative_role",
                    "stable_personality",
                    "baseline_delivery",
                    "diction_and_rhythm",
                    "emotional_range",
                    "relationship_dynamics",
                    "profile_summary",
                ],
            },
            {
                "line": 3,
                "quote": "他说得很快，却始终压着音量。",
                "supports": ["baseline_delivery", "diction_and_rhythm", "emotional_range"],
            },
        ],
        "confidence": 0.9,
    }


def performance_candidate(control=None, target_text=None, line=2):
    target_text = target_text or DIALOGUES[0]["text"]
    control = control or "呼吸稍急但保持清晰，语速加快，短句利落推进，重读抓紧，保持自然对话感"
    return {
        "intent": "带同伴一起脱离危险并稳定她的注意力",
        "subtext": "危险迫近，但我仍能判断路线，也不会丢下你",
        "continuity_state": "警觉已经升高，保护同伴的行动目标持续",
        "scene_relation": "build",
        "pace": "brisk",
        "volume": "firm",
        "intensity": 4,
        "breath": "轻微急促，在短句之间迅速换气",
        "rhythm": "短句清楚推进，在行动词前留极短停顿",
        "emotion_arc": "从快速判断转为坚定催促，结尾保持控制",
        "emphasis": ["抓紧" if "抓紧" in target_text else "跟紧"],
        "avoid": ["广告式喊腔", "恐慌到咬字失真"],
        "performance_control": control,
        "evidence": "目标台词是连续行动指令，附近叙述明确语速快且压着音量。",
        "evidence_lines": sorted({line, 3 if line == 2 else 5}),
        "evidence_quotes": [
            {
                "line": line,
                "quote": target_text,
                "supports": [
                    "intent",
                    "subtext",
                    "continuity_state",
                    "pace",
                    "volume",
                    "intensity",
                    "breath",
                    "rhythm",
                    "emotion_arc",
                    "performance_control",
                ],
            }
        ],
        "confidence": 0.88,
    }


def response_triplet(prefix, candidate):
    return [
        (prefix[0], candidate),
        (prefix[1], candidate),
        (prefix[2], candidate),
    ]


PROFILE_NAMES = (
    "submit_performance_profile",
    "submit_independent_performance_profile",
    "submit_final_performance_profile",
)
LINE_NAMES = (
    "submit_line_performance",
    "submit_independent_line_performance",
    "submit_final_line_performance",
)


def test_profile_uses_two_blind_agents_and_mandatory_adjudication():
    candidate = profile_candidate()
    client = FakeClient(response_triplet(PROFILE_NAMES, candidate))

    result = direct_character_profile("罗伦斯", NOVEL, DIALOGUES, client=client)

    assert result["profile_summary"] == candidate["profile_summary"]
    assert result["agent_calls"] == 3
    assert [item["stage"] for item in result["decision_chain"]] == [
        "primary",
        "independent_review",
        "final_adjudication",
    ]
    assert "PRIMARY PROFILE" not in client.calls[1]["messages"][1]["content"]
    assert "PRIMARY PROFILE" in client.calls[2]["messages"][1]["content"]
    assert [call["tool_choice"] for call in client.calls] == [
        {"type": "function", "function": {"name": name}}
        for name in PROFILE_NAMES
    ]
    assert [call["tools"][0]["function"]["name"] for call in client.calls] == list(PROFILE_NAMES)


def test_line_direction_is_blind_reviewed_adjudicated_and_executable():
    candidate = performance_candidate()
    client = FakeClient(response_triplet(LINE_NAMES, candidate))

    result = direct_line_performance(
        0,
        NOVEL,
        DIALOGUES,
        profile_candidate(),
        {"emotion": "nervous", "tone": "serious"},
        client=client,
    )

    assert result["performance_control"] == candidate["performance_control"]
    assert result["speaker"] == "罗伦斯"
    assert result["dialogue_index"] == 0
    assert result["agent_calls"] == 3
    assert "PRIMARY DIRECTION" not in client.calls[1]["messages"][1]["content"]
    assert "PRIMARY DIRECTION" in client.calls[2]["messages"][1]["content"]
    assert all(call["trace_id"].startswith("performance_direction:0:") for call in client.calls)
    assert [call["tool_choice"] for call in client.calls] == [
        {"type": "function", "function": {"name": name}}
        for name in LINE_NAMES
    ]
    assert [
        call["tools"][0]["function"]["parameters"]["properties"]["performance_control"]
        for call in client.calls
    ] == [{"type": "string", "minLength": 18, "maxLength": 140}] * 3


def test_line_submission_schema_uses_configured_control_bounds():
    candidate = performance_candidate()
    client = FakeClient(response_triplet(LINE_NAMES, candidate))

    direct_line_performance(
        0,
        NOVEL,
        DIALOGUES,
        profile_candidate(),
        client=client,
        min_control_chars=18,
        max_control_chars=140,
    )

    control_schema = client.calls[0]["tools"][0]["function"]["parameters"]["properties"][
        "performance_control"
    ]
    assert control_schema == {"type": "string", "minLength": 18, "maxLength": 140}


@pytest.mark.parametrize("control", ["轻声", "语速稍快，呼吸自然，节奏清晰，音量坚定，重点重读抓紧，保持克制推进，" * 8])
def test_out_of_bounds_control_is_compacted_without_accepting_other_invalid_fields(control):
    out_of_bounds = performance_candidate(
        control=control
    )
    valid = performance_candidate()
    client = FakeClient(
        [(LINE_NAMES[0], out_of_bounds), (LINE_NAMES[1], valid), (LINE_NAMES[2], valid)]
    )

    result = direct_line_performance(0, NOVEL, DIALOGUES, profile_candidate(), client=client)

    assert 18 <= len(result["performance_control"]) <= 140
    assert result["performance_control"] != out_of_bounds["performance_control"]
    assert result["agent_calls"] == 3


@pytest.mark.parametrize(
    "quote",
    ["他说得很快，却始终压着音量。", "听"],
)
def test_malformed_or_missing_target_quote_is_recovered_from_the_exact_target_dialogue(quote):
    missing_target_quote = performance_candidate()
    missing_target_quote["evidence_quotes"] = [
        {
            "line": 3 if quote.startswith("他") else 2,
            "quote": quote,
            "supports": list(missing_target_quote["evidence_quotes"][0]["supports"]),
        }
    ]
    valid = performance_candidate()
    client = FakeClient(
        [(LINE_NAMES[0], missing_target_quote), (LINE_NAMES[1], valid), (LINE_NAMES[2], valid)]
    )

    result = direct_line_performance(0, NOVEL, DIALOGUES, profile_candidate(), client=client)

    primary_quotes = result["decision_chain"][0]["evidence_quotes"]
    assert any(item["line"] == 2 and item["quote"] == DIALOGUES[0]["text"] for item in primary_quotes)
    assert result["agent_calls"] == 3


def test_punctuation_only_target_quote_is_valid_exact_evidence():
    novel = "他停下脚步。\n罗伦斯只说：「……」"
    dialogues = [{"speaker": "罗伦斯", "text": "……", "line": 2, "chapter": "one"}]
    candidate = performance_candidate(target_text="……", line=2)
    candidate["emphasis"] = []
    candidate["evidence_lines"] = [2]
    client = FakeClient(response_triplet(LINE_NAMES, candidate))

    result = direct_line_performance(0, novel, dialogues, profile_candidate(), client=client)

    assert result["text"] == "……"
    assert result["decision_chain"][0]["evidence_quotes"][0]["quote"] == "……"


@pytest.mark.parametrize(
    "mutator,error",
    [
        (
            lambda value: value.update(
                performance_control="(紧张但克制)语速加快，短句清楚推进并保持自然对话感"
            ),
            "parentheses",
        ),
        (
            lambda value: value.update(
                performance_control="年轻男声音色，呼吸稍急，语速加快并清楚重读，短句坚定收住"
            ),
            "identity",
        ),
        (lambda value: value.update(emphasis=["不存在的词"]), "substrings"),
        (lambda value: value.update(intensity=True), "integer"),
        (lambda value: value.update(evidence_quotes=[{"line": 2, "quote": "原文没有", "supports": ["intent"]}]), "exact substring"),
    ],
)
def test_invalid_direction_is_rejected_and_retried(mutator, error):
    invalid = performance_candidate()
    mutator(invalid)
    valid = performance_candidate()
    client = FakeClient(
        [(LINE_NAMES[0], invalid), (LINE_NAMES[0], valid), (LINE_NAMES[1], valid), (LINE_NAMES[2], valid)]
    )

    result = direct_line_performance(0, NOVEL, DIALOGUES, profile_candidate(), client=client)

    assert result["agent_calls"] == 4
    assert error in client.calls[1]["messages"][-1]["content"]


def test_direction_checkpoint_resumes_exactly_and_preserves_usage(tmp_path):
    checkpoint = tmp_path / "directions.checkpoint.json"
    profiles = {"罗伦斯": profile_candidate()}
    first = performance_candidate()
    second_text = DIALOGUES[1]["text"]
    second = performance_candidate(
        control="维持急行后的短促呼吸，语速快而清楚，跟紧二字坚定收住，保持贴近同伴的对话感",
        target_text=second_text,
        line=5,
    )
    client = FakeClient(
        response_triplet(LINE_NAMES, first) + response_triplet(LINE_NAMES, second),
        initial_usage={"calls": 7, "prompt_tokens": 700, "completion_tokens": 350, "total_tokens": 1050},
    )

    results = direct_all_performances(
        [0, 1],
        DIALOGUES,
        NOVEL,
        profiles,
        {},
        client=client,
        checkpoint_path=checkpoint,
    )

    assert set(results) == {"0", "1"}
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["llm_usage"] == {
        "calls": 6,
        "prompt_tokens": 600,
        "completion_tokens": 300,
        "total_tokens": 900,
    }
    resumed = FakeClient(initial_usage={"calls": 50, "total_tokens": 5000})
    assert direct_all_performances(
        [0, 1],
        DIALOGUES,
        NOVEL,
        profiles,
        {},
        client=resumed,
        checkpoint_path=checkpoint,
    ) == results
    assert resumed.calls == []


def test_direction_checkpoint_expansion_reuses_unaffected_and_regenerates_continuity_affected_results(
    tmp_path,
):
    checkpoint = tmp_path / "directions.checkpoint.json"
    extra_dialogue = {
        "speaker": "守卫",
        "text": "所有人立刻让开道路。",
        "line": 6,
        "chapter": "one",
    }
    dialogues = [DIALOGUES[0], extra_dialogue, DIALOGUES[1]]
    novel = NOVEL + "\n守卫喊道：『所有人立刻让开道路。』"
    old_profiles = {DIALOGUES[0]["speaker"]: profile_candidate()}
    old_first = performance_candidate()
    old_second = performance_candidate(
        control="维持急行后的短促呼吸，语速快而清楚，跟紧二字坚定收住，保持贴近同伴的对话感",
        target_text=DIALOGUES[1]["text"],
        line=5,
    )
    old_client = FakeClient(
        response_triplet(LINE_NAMES, old_first) + response_triplet(LINE_NAMES, old_second)
    )
    old_results = direct_all_performances(
        [0, 2],
        dialogues,
        novel,
        old_profiles,
        {},
        client=old_client,
        checkpoint_path=checkpoint,
    )

    extra_profile = copy.deepcopy(profile_candidate())
    extra_profile["character_name"] = "守卫"
    expanded_profiles = {**old_profiles, "守卫": extra_profile}
    extra_performance = performance_candidate(
        control="先吸一口气再迅速发令，音量提高但不嘶喊，重读立刻与让开，句尾果断收束",
        target_text=extra_dialogue["text"],
        line=6,
    )
    extra_performance["evidence_lines"] = [6]
    extra_performance["emphasis"] = ["立刻", "让开"]
    expanded_client = FakeClient(
        response_triplet(LINE_NAMES, extra_performance)
        + response_triplet(LINE_NAMES, old_second)
    )

    expanded = direct_all_performances(
        [0, 1, 2],
        dialogues,
        novel,
        expanded_profiles,
        {},
        client=expanded_client,
        checkpoint_path=checkpoint,
    )

    assert len(expanded_client.calls) == 6
    assert expanded["0"] == old_results["0"]
    assert expanded["2"]["performance_control"] == old_results["2"]["performance_control"]
    assert expanded["2"]["continuity_input_hash"] != old_results["2"]["continuity_input_hash"]
    assert expanded["2"]["decision_chain"][0]["input_hash"] != old_results["2"]["decision_chain"][0]["input_hash"]
    assert expanded["1"]["evidence_quotes"][0]["quote"] == extra_dialogue["text"]
    rewritten = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert rewritten["target_indices"] == [0, 1, 2]
    assert rewritten["completed_indices"] == [0, 1, 2]
    final_payload = {
        "meta": {
            "pipeline_version": PERFORMANCE_DIRECTION_PIPELINE_VERSION,
            "prompt_signature": PERFORMANCE_PROMPT_SIGNATURE,
            "model": "sensenova-6.7-flash-lite",
            "source_hash": performance_direction_source_hash(
                novel, [0, 1, 2], dialogues, expanded_profiles, {}
            ),
        },
        "results": expanded,
    }
    assert validate_performance_payload(
        final_payload,
        [0, 1, 2],
        dialogues,
        novel,
        expanded_profiles,
        {},
    ) == []


def test_final_payload_validation_binds_source_and_continuity(tmp_path):
    checkpoint = tmp_path / "directions.checkpoint.json"
    profiles = {"罗伦斯": profile_candidate()}
    first = performance_candidate()
    second = performance_candidate(
        control="维持急行后的短促呼吸，语速快而清楚，跟紧二字坚定收住，保持贴近同伴的对话感",
        target_text=DIALOGUES[1]["text"],
        line=5,
    )
    client = FakeClient(response_triplet(LINE_NAMES, first) + response_triplet(LINE_NAMES, second))
    results = direct_all_performances(
        [0, 1], DIALOGUES, NOVEL, profiles, {}, client=client, checkpoint_path=checkpoint
    )
    source_hash = performance_direction_source_hash(NOVEL, [0, 1], DIALOGUES, profiles, {})
    payload = {
        "meta": {
                "pipeline_version": PERFORMANCE_DIRECTION_PIPELINE_VERSION,
                "prompt_signature": PERFORMANCE_PROMPT_SIGNATURE,
            "model": "sensenova-6.7-flash-lite",
            "source_hash": source_hash,
        },
        "results": results,
    }

    assert validate_performance_payload(payload, [0, 1], DIALOGUES, NOVEL, profiles, {}) == []
    payload["results"]["1"]["continuity_input_hash"] = "wrong"
    problems = validate_performance_payload(payload, [0, 1], DIALOGUES, NOVEL, profiles, {})
    assert any("continuity_input_hash does not match" in problem for problem in problems)


def test_profile_checkpoint_usage_is_client_delta_not_shared_history(tmp_path):
    checkpoint = tmp_path / "profiles.checkpoint.json"
    candidate = profile_candidate()
    client = FakeClient(
        response_triplet(PROFILE_NAMES, candidate),
        initial_usage={"calls": 11, "prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    )

    profiles = build_performance_profiles(
        ["罗伦斯"], NOVEL, DIALOGUES, client=client, checkpoint_path=checkpoint
    )

    assert profiles["罗伦斯"]["profile_summary"] == candidate["profile_summary"]
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["llm_usage"] == {
        "calls": 3,
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "total_tokens": 450,
    }


def test_profile_checkpoint_expands_speaker_set_without_regenerating_existing_profile(tmp_path):
    checkpoint = tmp_path / "profiles.checkpoint.json"
    extra_dialogue = {
        "speaker": "守卫",
        "text": "所有人立刻让开道路。",
        "line": 6,
        "chapter": "one",
    }
    dialogues = [*DIALOGUES, extra_dialogue]
    novel = NOVEL + "\n守卫喊道：『所有人立刻让开道路。』"
    original_client = FakeClient(response_triplet(PROFILE_NAMES, profile_candidate()))
    original = build_performance_profiles(
        [DIALOGUES[0]["speaker"]],
        novel,
        dialogues,
        client=original_client,
        checkpoint_path=checkpoint,
    )

    guard_profile = copy.deepcopy(profile_candidate())
    guard_profile["character_name"] = "守卫"
    guard_profile["evidence_lines"] = [6]
    guard_profile["evidence_quotes"] = [
        {
            "line": 6,
            "quote": "所有人",
            "supports": copy.deepcopy(profile_candidate()["evidence_quotes"][0]["supports"]),
        },
        {
            "line": 6,
            "quote": "立刻让开道路",
            "supports": ["baseline_delivery", "diction_and_rhythm", "emotional_range"],
        },
    ]
    expanded_client = FakeClient(response_triplet(PROFILE_NAMES, guard_profile))

    expanded = build_performance_profiles(
        [DIALOGUES[0]["speaker"], "守卫"],
        novel,
        dialogues,
        client=expanded_client,
        checkpoint_path=checkpoint,
    )

    assert len(expanded_client.calls) == 3
    assert expanded[DIALOGUES[0]["speaker"]] == original[DIALOGUES[0]["speaker"]]
    assert expanded["守卫"]["character_name"] == "守卫"
    rewritten = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert rewritten["target_speakers"] == [DIALOGUES[0]["speaker"], "守卫"]


def test_direction_checkpoint_resumes_inside_item_after_primary(tmp_path):
    checkpoint = tmp_path / "directions.checkpoint.json"
    profiles = {"罗伦斯": profile_candidate()}
    candidate = performance_candidate()
    interrupted = FakeClient([(LINE_NAMES[0], candidate)])

    with pytest.raises(PerformanceBatchError):
        direct_all_performances(
            [0],
            DIALOGUES,
            NOVEL,
            profiles,
            {},
            client=interrupted,
            checkpoint_path=checkpoint,
            max_agent_rounds=1,
            item_retries=1,
        )

    partial = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert set(partial["inflight"]["stages"]) == {"primary"}
    assert partial["inflight"]["stages"]["primary"]["usage"]["total_tokens"] == 150

    resumed = FakeClient(
        [
            (LINE_NAMES[1], candidate),
            (LINE_NAMES[2], candidate),
        ]
    )
    results = direct_all_performances(
        [0],
        DIALOGUES,
        NOVEL,
        profiles,
        {},
        client=resumed,
        checkpoint_path=checkpoint,
        max_agent_rounds=1,
        item_retries=1,
    )

    assert len(resumed.calls) == 2
    assert results["0"]["agent_calls"] == 3
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["inflight"] == {}


def test_final_payload_requires_auditable_three_stage_chain(tmp_path):
    checkpoint = tmp_path / "directions.checkpoint.json"
    profiles = {"罗伦斯": profile_candidate()}
    candidate = performance_candidate()
    client = FakeClient(response_triplet(LINE_NAMES, candidate))
    results = direct_all_performances(
        [0], DIALOGUES, NOVEL, profiles, {}, client=client, checkpoint_path=checkpoint
    )
    payload = {
        "meta": {
            "pipeline_version": PERFORMANCE_DIRECTION_PIPELINE_VERSION,
            "prompt_signature": PERFORMANCE_PROMPT_SIGNATURE,
            "model": "sensenova-6.7-flash-lite",
            "source_hash": performance_direction_source_hash(NOVEL, [0], DIALOGUES, profiles, {}),
        },
        "results": copy.deepcopy(results),
    }
    del payload["results"]["0"]["decision_chain"]

    problems = validate_performance_payload(payload, [0], DIALOGUES, NOVEL, profiles, {})

    assert any("decision_chain" in problem for problem in problems)
