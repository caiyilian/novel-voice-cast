import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.bgm_segmenter import (  # noqa: E402
    BGM_SEGMENTATION_PIPELINE_VERSION,
    BGM_TYPE_PIPELINE_VERSION,
    NovelSegmentationIndex,
    _merge_title_card_segments,
    _musical_coherence_problems,
    _natural_chunks,
    _segment_inputs_hash,
    bgm_source_hash,
    label_bgm_types,
    save_segments,
    segment_chunk_direct,
    segment_novel,
    segment_novel_chunked,
    validate_segments,
)
from app.core.llm_client import LLMResult, SENSENOVA_FLASH_LITE_MODEL  # noqa: E402
from app.core.ollama_client import ChatResult, ToolCall  # noqa: E402
from app.core.parser import parse  # noqa: E402


def test_parser_includes_source_line_numbers():
    text = "\n".join([
        "第一章",
        "这是一段足够长的旁白说明。",
        "「你好」",
        "罗伦斯：「走吧」",
        "他说完以后「中间的对话」又继续前进。",
    ])
    labels = ["赫萝", "罗伦斯", "赫萝"]

    dialogues, _ = parse(text, labels)

    assert [dialogue["line"] for dialogue in dialogues] == [2, 3, 4, 5]
    assert dialogues[0]["speaker"] == "旁白"
    assert dialogues[1]["speaker"] == "赫萝"


def test_index_returns_dialogues_in_line_range():
    dialogues = [
        {"line": 3, "speaker": "赫萝", "chapter": "第一章", "text": "你好"},
        {"line": 8, "speaker": "罗伦斯", "chapter": "第一章", "text": "走吧"},
    ]
    index = NovelSegmentationIndex("\n".join(f"line {i}" for i in range(1, 11)), dialogues)

    result = index.get_dialogues(1, 5)

    assert result["total_dialogues"] == 1
    assert result["dialogues"][0]["line"] == 3
    assert result["dialogues"][0]["dialogue_index"] == 1


def test_segment_novel_tool_loop_returns_valid_segments():
    text = "\n".join(f"第 {i} 行" for i in range(1, 51))
    dialogues = [{"line": i, "speaker": "旁白", "text": f"第 {i} 行"} for i in range(1, 51)]
    client = FakeSegmentationClient()

    segments = segment_novel(text, dialogues=dialogues, client=client, max_tool_steps=5)

    assert len(segments) == 5
    assert segments[0]["start_line"] == 1
    assert segments[-1]["end_line"] == 50
    assert validate_segments(segments, total_lines=50) == []


def test_save_segments_writes_json_array(tmp_path):
    segments = [
        {"start_line": 1, "end_line": 10, "title": "开端", "description": "故事开始"},
        {"start_line": 11, "end_line": 20, "title": "发展", "description": "冲突推进"},
    ]

    output_path = save_segments(tmp_path / "bgm_segments.json", segments)

    assert output_path.read_text(encoding="utf-8").startswith("[")


class FakeSegmentationClient:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.2, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatResult(tool_calls=[
                ToolCall("call_1", "get_novel_text", {"start_line": 1, "end_line": 50}),
                ToolCall("call_2", "get_dialogues", {"start_line": 1, "end_line": 50}),
            ])
        if self.calls == 2:
            return ChatResult(tool_calls=[
                ToolCall("call_3", "submit_segment", {
                    "start_line": 1,
                    "end_line": 10,
                    "title": "开端",
                    "description": "故事建立基调",
                }),
                ToolCall("call_4", "submit_segment", {
                    "start_line": 11,
                    "end_line": 20,
                    "title": "相遇",
                    "description": "角色互动展开",
                }),
                ToolCall("call_5", "submit_segment", {
                    "start_line": 21,
                    "end_line": 30,
                    "title": "转折",
                    "description": "新的问题出现",
                }),
                ToolCall("call_6", "submit_segment", {
                    "start_line": 31,
                    "end_line": 40,
                    "title": "推进",
                    "description": "冲突继续发展",
                }),
                ToolCall("call_7", "submit_segment", {
                    "start_line": 41,
                    "end_line": 50,
                    "title": "收束",
                    "description": "当前卷情节告一段落",
                }),
            ])
        return ChatResult(tool_calls=[
            ToolCall("call_8", "list_segments", {}),
            ToolCall("call_9", "finish_segmentation", {}),
        ])


class DirectClient:
    def __init__(self, responses, starting_usage=None):
        self.responses = list(responses)
        self.calls = 0
        self.sensenova_model = SENSENOVA_FLASH_LITE_MODEL
        self._usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._usage.update(starting_usage or {})

    def usage_summary(self):
        return dict(self._usage)

    def chat(self, messages, **kwargs):
        self.calls += 1
        self._usage["calls"] += 1
        response = self.responses.pop(0)
        usage = getattr(response, "usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._usage[key] += int(usage.get(key, 0) or 0)
        return response


def test_chunk_segmentation_always_gets_second_pass_review():
    candidate = '[{"start_line":10,"end_line":11,"title":"one","description":"setup"},{"start_line":12,"end_line":12,"title":"two","description":"turn"}]'
    reviewed = '[{"start_line":10,"end_line":12,"title":"one","description":"continuous scene"}]'
    client = DirectClient([LLMResult(content=candidate), LLMResult(content=reviewed)])

    segments = segment_chunk_direct("a\nb\nc", base_line=10, client=client)

    assert client.calls == 2
    assert segments == [{"start_line": 1, "end_line": 3, "title": "one", "description": "continuous scene"}]


def test_bgm_type_is_independently_reviewed(tmp_path):
    primary = LLMResult(tool_calls=[ToolCall("p", "submit_bgm_type", {
        "segment_index": 1, "bgm_type": "suspense", "confidence": 0.8, "evidence": "line 2 hidden danger",
    })])
    review = LLMResult(tool_calls=[ToolCall("r", "submit_bgm_type", {
        "segment_index": 1, "bgm_type": "suspense", "confidence": 0.9, "evidence": "line 3 rising tension",
    })])
    client = DirectClient([primary, review])
    segments = [{"start_line": 1, "end_line": 3, "title": "risk", "description": "danger approaches"}]

    result = label_bgm_types(
        segments,
        client=client,
        novel_text="quiet\na shadow moved\nthe door opened",
        checkpoint_path=tmp_path / "types.json",
    )

    assert result[0]["bgm_type"] == "suspense"
    assert result[0]["bgm_agent_calls"] == 2


def test_chunked_resume_accumulates_only_new_client_usage(tmp_path):
    text = "one\ntwo\nthree\nfour\nfive\nsix"
    ranges = _natural_chunks(text.splitlines(), 2)
    checkpoint_path = tmp_path / "segments.json"
    previous_usage = {
        "calls": 4,
        "prompt_tokens": 40,
        "completion_tokens": 10,
        "total_tokens": 50,
    }
    checkpoint_path.write_text(
        json.dumps(
            {
                "pipeline_version": BGM_SEGMENTATION_PIPELINE_VERSION,
                "source_hash": bgm_source_hash(text),
                "model": SENSENOVA_FLASH_LITE_MODEL,
                "ranges": [list(item) for item in ranges],
                "chunks": {
                    "0": [{
                        "start_line": ranges[0][0],
                        "end_line": ranges[0][1],
                        "title": "resumed",
                        "description": "already completed",
                    }]
                },
                "llm_usage": previous_usage,
            }
        ),
        encoding="utf-8",
    )
    new_chunk = json.dumps([{
        "start_line": ranges[1][0],
        "end_line": ranges[1][1],
        "title": "continued",
        "description": "new work",
    }])
    client = DirectClient(
        [
            LLMResult(
                content=new_chunk,
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            ),
            LLMResult(
                content=new_chunk,
                usage={"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
            ),
            LLMResult(
                tool_calls=[ToolCall("boundary", "submit_boundary_review", {
                    "merge": False,
                    "reason": "different events",
                    "merged_title": "",
                    "merged_description": "",
                })],
                usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            ),
        ],
        starting_usage={
            "calls": 10,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    )

    result = segment_novel_chunked(
        text,
        client=client,
        num_chunks=2,
        checkpoint_path=checkpoint_path,
    )

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert client.calls == 3
    assert result[0]["title"] == "resumed"
    assert payload["llm_usage"] == {
        "calls": 7,
        "prompt_tokens": 55,
        "completion_tokens": 16,
        "total_tokens": 71,
    }
    assert all(
        segment["segmentation_pipeline_version"] == BGM_SEGMENTATION_PIPELINE_VERSION
        and segment["segmentation_source_hash"] == bgm_source_hash(text)
        and segment["segmentation_model"] == SENSENOVA_FLASH_LITE_MODEL
        for segment in result
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("pipeline_version", 0),
        ("source_hash", "old-source"),
        ("model", "old-model"),
    ],
)
def test_chunked_checkpoint_identity_mismatch_recomputes(tmp_path, field, bad_value):
    text = "one\ntwo\nthree"
    ranges = _natural_chunks(text.splitlines(), 1)
    checkpoint_path = tmp_path / f"segments-{field}.json"
    payload = {
        "pipeline_version": BGM_SEGMENTATION_PIPELINE_VERSION,
        "source_hash": bgm_source_hash(text),
        "model": SENSENOVA_FLASH_LITE_MODEL,
        "ranges": [list(item) for item in ranges],
        "chunks": {
            "0": [{
                "start_line": 1,
                "end_line": 3,
                "title": "stale",
                "description": "must not resume",
            }]
        },
        "llm_usage": {"calls": 9, "prompt_tokens": 90, "completion_tokens": 9, "total_tokens": 99},
    }
    payload[field] = bad_value
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    fresh = json.dumps([{
        "start_line": 1,
        "end_line": 3,
        "title": "fresh",
        "description": "recomputed",
    }])
    client = DirectClient([LLMResult(content=fresh), LLMResult(content=fresh)])

    result = segment_novel_chunked(
        text,
        client=client,
        num_chunks=1,
        checkpoint_path=checkpoint_path,
    )

    rewritten = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert client.calls == 2
    assert result[0]["title"] == "fresh"
    assert rewritten["pipeline_version"] == BGM_SEGMENTATION_PIPELINE_VERSION
    assert rewritten["source_hash"] == bgm_source_hash(text)
    assert rewritten["model"] == SENSENOVA_FLASH_LITE_MODEL


def test_type_resume_accumulates_only_new_client_usage(tmp_path):
    novel_text = "quiet\ndanger\ncalm\nhome"
    segments = [
        {"start_line": 1, "end_line": 2, "title": "risk", "description": "danger"},
        {"start_line": 3, "end_line": 4, "title": "return", "description": "calm"},
    ]
    checkpoint_path = tmp_path / "types.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "pipeline_version": BGM_TYPE_PIPELINE_VERSION,
                "source_hash": bgm_source_hash(novel_text),
                "model": SENSENOVA_FLASH_LITE_MODEL,
                "segments_hash": _segment_inputs_hash(segments),
                "results": {
                    "1": {
                        "segment_index": 1,
                        "bgm_type": "suspense",
                        "confidence": 0.8,
                        "evidence": "line 2 danger",
                        "agent_calls": 2,
                    }
                },
                "llm_usage": {
                    "calls": 5,
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "total_tokens": 70,
                },
            }
        ),
        encoding="utf-8",
    )
    primary = LLMResult(
        tool_calls=[ToolCall("p2", "submit_bgm_type", {
            "segment_index": 2,
            "bgm_type": "daily",
            "confidence": 0.7,
            "evidence": "line 4 calm return",
        })],
        usage={"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
    )
    review = LLMResult(
        tool_calls=[ToolCall("r2", "submit_bgm_type", {
            "segment_index": 2,
            "bgm_type": "daily",
            "confidence": 0.9,
            "evidence": "line 3 quiet homecoming",
        })],
        usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    )
    client = DirectClient(
        [primary, review],
        starting_usage={
            "calls": 11,
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
        },
    )

    result = label_bgm_types(
        segments,
        client=client,
        novel_text=novel_text,
        checkpoint_path=checkpoint_path,
    )

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert client.calls == 2
    assert [segment["bgm_type"] for segment in result] == ["suspense", "daily"]
    assert payload["llm_usage"] == {
        "calls": 7,
        "prompt_tokens": 67,
        "completion_tokens": 27,
        "total_tokens": 94,
    }
    assert payload["source_hash"] == bgm_source_hash(novel_text)
    assert all(
        segment["bgm_pipeline_version"] == BGM_TYPE_PIPELINE_VERSION
        and segment["bgm_source_hash"] == bgm_source_hash(novel_text)
        and segment["bgm_model"] == SENSENOVA_FLASH_LITE_MODEL
        for segment in result
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("pipeline_version", 0),
        ("source_hash", "old-source"),
        ("model", "old-model"),
    ],
)
def test_type_checkpoint_identity_mismatch_recomputes(tmp_path, field, bad_value):
    novel_text = "quiet\ndanger"
    segments = [{"start_line": 1, "end_line": 2, "title": "risk", "description": "danger"}]
    checkpoint_path = tmp_path / f"types-{field}.json"
    payload = {
        "pipeline_version": BGM_TYPE_PIPELINE_VERSION,
        "source_hash": bgm_source_hash(novel_text),
        "model": SENSENOVA_FLASH_LITE_MODEL,
        "segments_hash": _segment_inputs_hash(segments),
        "results": {
            "1": {
                "segment_index": 1,
                "bgm_type": "sad",
                "confidence": 0.8,
                "evidence": "stale result",
            }
        },
        "llm_usage": {"calls": 2, "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }
    payload[field] = bad_value
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    primary = LLMResult(tool_calls=[ToolCall("p", "submit_bgm_type", {
        "segment_index": 1,
        "bgm_type": "suspense",
        "confidence": 0.8,
        "evidence": "line 2 danger",
    })])
    review = LLMResult(tool_calls=[ToolCall("r", "submit_bgm_type", {
        "segment_index": 1,
        "bgm_type": "suspense",
        "confidence": 0.9,
        "evidence": "line 2 rising danger",
    })])
    client = DirectClient([primary, review])

    result = label_bgm_types(
        segments,
        client=client,
        novel_text=novel_text,
        checkpoint_path=checkpoint_path,
    )

    rewritten = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert client.calls == 2
    assert result[0]["bgm_type"] == "suspense"
    assert rewritten["pipeline_version"] == BGM_TYPE_PIPELINE_VERSION
    assert rewritten["source_hash"] == bgm_source_hash(novel_text)
    assert rewritten["model"] == SENSENOVA_FLASH_LITE_MODEL


def test_internal_music_transition_is_rejected():
    problems = _musical_coherence_problems([{
        "start_line": 1,
        "end_line": 20,
        "title": "mixed",
        "description": "The music shifts from relaxed travel to tense confrontation.",
    }])

    assert problems


def test_short_title_card_is_merged_into_following_scene():
    result = _merge_title_card_segments([
        {"start_line": 1, "end_line": 5, "title": "Act two title card", "description": "title"},
        {"start_line": 6, "end_line": 20, "title": "Journey", "description": "calm travel"},
    ])

    assert len(result) == 1
    assert result[0]["start_line"] == 1
    assert result[0]["end_line"] == 20
