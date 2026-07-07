import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.illustration_planner import (  # noqa: E402
    NovelIndex,
    _context_stats,
    _dedupe_proposals,
    _extract_visual_memory_from_content,
    _extract_proposals_from_content,
    _filter_non_story_proposals,
    _format_character_cards,
    _format_visual_memory_for_chunk,
    _line_labels_from_input,
    _load_character_cards,
    _merge_visual_memory,
    _merge_proposals,
    _run_tool,
    plan_illustrations,
)
from app.core.llm_client import LLMResult, ToolCall  # noqa: E402


def test_plan_illustrations_falls_back_to_direct_json_when_tools_are_missing(tmp_path):
    text = "\n".join(f"line {i}" for i in range(1, 9))
    labels = ["Narrator"] * 8
    client = FakeNoToolThenJsonClient()
    output = tmp_path / "plan.json"

    proposals = plan_illustrations(
        text=text,
        labels=labels,
        client=client,
        output_path=output,
        debug=True,
        enable_visual_memory=False,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(proposals) == 2
    assert data["summary"]["total_illustrations"] == 2
    assert data["summary"]["total_llm_calls"] == 3
    assert data["summary"]["agent_usage"]["illustration_planner"]["calls"] == 2
    assert data["summary"]["agent_usage"]["direct_planner"]["calls"] == 1
    assert data["call_log"][0]["phase"] == "tool"
    assert data["call_log"][0]["agent"] == "illustration_planner"
    assert data["call_log"][0]["context"]["estimated_tokens"] > 0
    assert data["call_log"][-1]["phase"] == "direct"
    assert (tmp_path / "plan.debug").is_dir()


def test_dedupe_keeps_overlapping_camera_beats():
    proposals = [
        {"title": "first", "start_line": 1, "end_line": 10, "description": "wide shot"},
        {"title": "second", "start_line": 5, "end_line": 12, "description": "reaction shot"},
    ]

    assert len(_dedupe_proposals(proposals)) == 2
    assert len(_merge_proposals(proposals)) == 2


def test_dedupe_removes_same_range_near_duplicates():
    proposals = [
        {"title": "摸狼爪", "start_line": 1, "end_line": 3, "description": "罗伦斯伸手触摸狼爪"},
        {"title": "触摸狼爪", "start_line": 1, "end_line": 3, "description": "罗伦斯伸手去摸狼爪"},
        {"title": "惊讶反应", "start_line": 2, "end_line": 4, "description": "赫萝突然出声吓到罗伦斯"},
    ]

    deduped = _dedupe_proposals(proposals)

    assert len(deduped) == 2
    assert [p["title"] for p in deduped] == ["摸狼爪", "惊讶反应"]


def test_filter_non_story_markers_and_afterword():
    source_lines = ["正文开始", "插图", "终", "幕", "正文继续", "后记", "作者的话"]
    proposals = [
        {"title": "正文", "start_line": 1, "end_line": 1},
        {"title": "终幕卡", "start_line": 2, "end_line": 4},
        {"title": "后记", "start_line": 6, "end_line": 7},
        {"title": "正文继续", "start_line": 5, "end_line": 5},
    ]

    filtered = _filter_non_story_proposals(proposals, source_lines)

    assert [p["title"] for p in filtered] == ["正文", "正文继续"]


def test_run_tool_accepts_empty_line_numbers():
    index = NovelIndex("one\ntwo\nthree")
    proposals = []

    result = _run_tool(
        ToolCall(
            "call_1",
            "propose_illustration",
            {
                "title": "empty lines",
                "start_line": "",
                "end_line": "",
                "description": "a beat with missing line numbers",
                "reason": "model supplied empty strings",
                "characters": "A,B",
                "composition": "close-up",
                "prompt": "two characters reacting",
            },
        ),
        index,
        proposals,
    )

    assert result.startswith("OK proposal")
    assert proposals[0]["start_line"] == 1
    assert proposals[0]["end_line"] == 1
    assert proposals[0]["characters"] == ["A", "B"]
    assert "anime style" in proposals[0]["prompt"]


def test_list_proposals_returns_compact_review_without_prompts():
    index = NovelIndex("one\ntwo\nthree")
    proposals = [{
        "title": "beat",
        "start_line": 1,
        "end_line": 2,
        "description": "x" * 200,
        "reason": "because",
        "characters": ["A"],
        "composition": "close-up",
        "prompt": "long image prompt that should not be echoed back into context",
    }]

    result = _run_tool(ToolCall("call_1", "list_proposals", {}), index, proposals)
    rows = json.loads(result)

    assert rows[0]["title"] == "beat"
    assert "prompt" not in rows[0]
    assert len(rows[0]["description"]) < 130


def test_extract_proposals_from_fenced_json():
    index = NovelIndex("one\ntwo")
    content = """```json
{"illustrations": [{"title": "beat", "start_line": 1, "end_line": 2, "description": "desc"}]}
```"""

    proposals = _extract_proposals_from_content(content, index)

    assert len(proposals) == 1
    assert proposals[0]["title"] == "beat"
    assert proposals[0]["end_line"] == 2


def test_dialogue_order_labels_are_mapped_to_source_lines():
    text = "\n".join([
        "opening narration",
        "「hello」",
        "middle narration",
        "「yes」「no」",
    ])
    labels = ["Alice", "Bob", "Carol"]

    line_labels = _line_labels_from_input(text, labels)

    assert line_labels == ["", "Alice", "", "Bob / Carol"]


def test_character_card_markdown_supports_future_appearance_fields(tmp_path):
    path = tmp_path / "cards.md"
    path.write_text(
        "\n".join([
            "| 角色名 | 对话数 | 性别 | 置信度 | 外观 | 服装 |",
            "|---|---:|---|---:|---|---|",
            "| Alice | 10 | female | 0.9 | silver hair | red cloak |",
        ]),
        encoding="utf-8",
    )

    cards = _load_character_cards(path)
    context = _format_character_cards(["Alice"], cards)

    assert cards["Alice"]["appearance"] == "silver hair"
    assert "appearance=silver hair" in context
    assert "clothing=red cloak" in context


def test_visual_memory_tracks_stable_facts_and_state_changes():
    update = _extract_visual_memory_from_content(json.dumps({
        "characters": [
            {
                "name": "Alice",
                "stable": {"appearance": "silver hair", "visual_anchor": "silver-haired traveler"},
                "states": [
                    {"start_line": 2, "clothing": "red cloak", "evidence_lines": [2]},
                    {"start_line": 7, "clothing": "blue coat", "evidence_lines": [7]},
                ],
            }
        ]
    }))
    memory = _merge_visual_memory({"characters": {}}, update)

    assert memory["characters"]["Alice"]["stable"]["appearance"] == "silver hair"
    assert [s["clothing"] for s in memory["characters"]["Alice"]["states"]] == ["red cloak", "blue coat"]
    chunk_context = _format_visual_memory_for_chunk(["Alice"], memory, 8, 10)
    assert "silver hair" in chunk_context
    assert "blue coat" in chunk_context

    _merge_visual_memory(memory, {"characters": [{"name": "Alice", "stable": {"appearance": "silver hair / silver hair"}}]})
    assert memory["characters"]["Alice"]["stable"]["appearance"] == "silver hair"


def test_context_stats_estimates_request_size():
    stats = _context_stats([{"role": "user", "content": "x" * 100}], tools=[{"type": "function"}])

    assert stats["message_count"] == 1
    assert stats["tool_count"] == 1
    assert stats["estimated_tokens"] > 0
    assert stats["risk"] == "low"


def test_plan_illustrations_builds_visual_memory_and_logs_agent_context(tmp_path):
    text = "\n".join([
        "Alice entered the room.",
        "She wore a red cloak.",
        "Alice smiled at the window.",
        "The light changed.",
    ])
    labels = ["Alice", "", "Alice", ""]
    client = FakeNoToolThenJsonClient()
    output = tmp_path / "plan.json"
    memory_path = tmp_path / "visual_memory.json"

    proposals = plan_illustrations(
        text=text,
        labels=labels,
        client=client,
        output_path=output,
        debug=True,
        visual_memory_path=memory_path,
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert len(proposals) == 2
    assert memory["characters"]["Alice"]["stable"]["appearance"] == "silver hair"
    assert data["call_log"][0]["agent"] == "visual_memory"
    assert data["call_log"][0]["context"]["estimated_tokens"] > 0
    assert data["summary"]["visual_memory_characters"] == 1
    assert data["summary"]["agent_usage"]["visual_memory"]["calls"] == 1

    tool_requests = [req for req in client.requests if req["tools"]]
    assert tool_requests
    tool_prompt = tool_requests[0]["messages"][1]["content"]
    assert "Visual memory for continuity" in tool_prompt
    assert "silver hair" in tool_prompt


class FakeNoToolThenJsonClient:
    def __init__(self):
        self.calls = 0
        self.requests = []

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=4096, tool_choice="auto"):
        self.calls += 1
        self.requests.append({"messages": messages, "tools": tools})
        if "Visual Memory Agent" in messages[0]["content"]:
            return LLMResult(
                content=json.dumps(
                    {
                        "characters": [
                            {
                                "name": "Alice",
                                "stable": {
                                    "appearance": "silver hair",
                                    "visual_anchor": "silver-haired traveler",
                                },
                                "states": [
                                    {
                                        "start_line": 2,
                                        "clothing": "red cloak",
                                        "evidence_lines": [2],
                                        "confidence": 0.9,
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                model="fake-memory-model",
                account_index=0,
                usage={"prompt_tokens": 8, "completion_tokens": 9, "total_tokens": 17},
                raw={"mode": "visual-memory", "call": self.calls},
            )
        if tools:
            return LLMResult(
                content="I can describe the plan, but I am not using tools.",
                model="fake-tool-model",
                account_index=0,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                raw={"mode": "tool-noop", "call": self.calls},
            )
        return LLMResult(
            content=json.dumps(
                {
                    "illustrations": [
                        {
                            "title": "opening",
                            "start_line": 1,
                            "end_line": 3,
                            "description": "first visible beat",
                            "reason": "opening camera beat",
                            "characters": ["Narrator"],
                            "composition": "wide shot",
                            "prompt": "quiet opening scene, anime style, masterpiece, high quality",
                        },
                        {
                            "title": "reaction",
                            "start_line": 4,
                            "end_line": 6,
                            "description": "second visible beat",
                            "reason": "reaction beat",
                            "characters": ["Narrator"],
                            "composition": "close-up",
                            "prompt": "subtle character reaction, anime style, masterpiece, high quality",
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            model="fake-direct-model",
            account_index=-1,
            usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            raw={"mode": "direct-json", "call": self.calls},
        )
