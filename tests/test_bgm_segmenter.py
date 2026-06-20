import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.bgm_segmenter import (  # noqa: E402
    NovelSegmentationIndex,
    save_segments,
    segment_novel,
    validate_segments,
)
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

    def chat(self, messages, tools=None, temperature=0.2):
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
