"""
Stage 7-a BGM scene segmenter.

The agent lets an Ollama model inspect the novel through tool calls and submit
scene/event boundaries that can later drive BGM generation.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.ollama_client import ChatResult, OllamaClient, ToolCall


DEFAULT_OUTPUT_PATH = Path("backend/data/bgm_segments.json")


class SegmentationError(Exception):
    """Raised when the model cannot produce a valid segmentation."""


class NovelSegmentationIndex:
    """Line-oriented access to the novel and parsed dialogue metadata."""

    def __init__(self, text: str, dialogues: Optional[List[Dict[str, Any]]] = None):
        self.text = text
        self.lines = text.splitlines()
        self.dialogues = dialogues or []

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    def get_novel_text(self, start_line: int, end_line: int, limit: int = 350) -> Dict[str, Any]:
        """Return novel text for a 1-based inclusive line range."""
        start, end = self._normalise_range(start_line, end_line)
        selected = self.lines[start - 1 : end]
        truncated = False
        if len(selected) > limit:
            selected = selected[:limit]
            end = start + limit - 1
            truncated = True

        return {
            "start_line": start,
            "end_line": end,
            "total_lines": self.total_lines,
            "truncated": truncated,
            "text": "\n".join(
                f"{start + offset}: {line.strip()}"
                for offset, line in enumerate(selected)
            ),
        }

    def get_dialogues(self, start_line: int, end_line: int, limit: int = 160) -> Dict[str, Any]:
        """Return parsed dialogues whose source line falls inside the range."""
        start, end = self._normalise_range(start_line, end_line)
        matches = []

        for dialogue_index, dialogue in enumerate(self.dialogues, start=1):
            line = dialogue.get("line")
            if not isinstance(line, int):
                continue
            if start <= line <= end:
                matches.append({
                    "dialogue_index": dialogue_index,
                    "line": line,
                    "chapter": dialogue.get("chapter", ""),
                    "speaker": dialogue.get("speaker", ""),
                    "text": dialogue.get("text", ""),
                })

        truncated = len(matches) > limit
        visible = matches[:limit]
        return {
            "start_line": start,
            "end_line": end,
            "total_dialogues": len(matches),
            "truncated": truncated,
            "dialogues": visible,
        }

    def _normalise_range(self, start_line: int, end_line: int) -> Tuple[int, int]:
        if self.total_lines <= 0:
            return 1, 0
        raw_start = int(start_line)
        raw_end = int(end_line)
        if raw_start > raw_end:
            raw_start, raw_end = raw_end, raw_start
        start = min(max(1, raw_start), self.total_lines)
        end = min(max(1, raw_end), self.total_lines)
        return start, end


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_novel_text",
            "description": "Get original novel text for a line range. Lines are 1-based and inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "description": "Start line, 1-based"},
                    "end_line": {"type": "integer", "description": "End line, inclusive"},
                },
                "required": ["start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dialogues",
            "description": "Get parsed dialogues in a line range, including speaker, chapter, and dialogue index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "description": "Start line, 1-based"},
                    "end_line": {"type": "integer", "description": "End line, inclusive"},
                },
                "required": ["start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_segment",
            "description": "Submit one event/scene segment boundary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short scene title"},
                    "description": {"type": "string", "description": "Scene summary and emotional/music context"},
                    "start_line": {"type": "integer", "description": "Segment start line, 1-based"},
                    "end_line": {"type": "integer", "description": "Segment end line, inclusive"},
                },
                "required": ["title", "description", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_segments",
            "description": "List currently submitted segments before review or boundary adjustment.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_segment",
            "description": "Update a submitted segment by 1-based segment index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer", "description": "1-based index from list_segments"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["segment_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_segment",
            "description": "Delete a submitted segment by 1-based segment index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer", "description": "1-based index from list_segments"},
                },
                "required": ["segment_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_segmentation",
            "description": "Finish after all segments are submitted and reviewed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


SYSTEM_PROMPT = """You are Stage 7-a of Novel Voice Cast: a BGM scene segmentation agent.

Goal:
- Understand the whole novel structure by calling tools.
- Segment the full volume by plot/event/scene transitions, not by equal line counts.
- Produce 5-20 non-overlapping, continuous segments covering the whole novel.
- Every segment must have start_line, end_line, title, and description.

Tool use rules:
1. Use get_novel_text to inspect broad ranges first, then narrower ranges around transitions.
2. Use get_dialogues to confirm boundaries against parsed dialogue line numbers.
3. Submit segments with submit_segment.
4. Review submitted segments with list_segments and update_segment/delete_segment when needed.
5. Call finish_segmentation only after the full novel is covered from line 1 through the final line.

Boundary rules:
- Lines are 1-based and inclusive.
- Prefer scene/event turning points over fixed sizes.
- Keep boundaries compatible with dialogue source lines, so later BGM timing can map back to dialogues.
- Titles and descriptions should be in Chinese when the novel is Chinese.
"""


def segment_novel(
    text: str,
    dialogues: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OllamaClient] = None,
    min_segments: int = 5,
    max_segments: int = 20,
    max_tool_steps: int = 80,
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    """Segment a novel into BGM scenes using Ollama tool calling."""
    if not text.strip():
        raise SegmentationError("Novel text is empty")

    client = client or OllamaClient()
    index = NovelSegmentationIndex(text, dialogues)
    segments: List[Dict[str, Any]] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The novel has {index.total_lines} lines and "
                f"{len(index.dialogues)} parsed dialogue/narration entries.\n"
                "Use the tools to inspect the full volume, submit 5-20 scene/event "
                "segments, review the boundaries, then finish."
            ),
        },
    ]

    for _ in range(max_tool_steps):
        result = client.chat(messages, tools=TOOL_SPECS, temperature=temperature)

        if result.tool_calls:
            messages.append(_assistant_tool_message(result))
            finished = _execute_tool_calls(
                result.tool_calls,
                index=index,
                segments=segments,
                messages=messages,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            if finished is not None:
                return finished
            continue

        fallback = _extract_segments_from_content(result.content)
        if fallback:
            problems = validate_segments(
                fallback,
                total_lines=index.total_lines,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            if not problems:
                return _sorted_segments(fallback)

        messages.append({
            "role": "user",
            "content": "Please continue using the provided tools. Submit valid segments and call finish_segmentation.",
        })

    problems = validate_segments(
        segments,
        total_lines=index.total_lines,
        min_segments=min_segments,
        max_segments=max_segments,
    )
    if not problems:
        return _sorted_segments(segments)

    raise SegmentationError("Segmentation did not finish: " + "; ".join(problems))


def validate_segments(
    segments: List[Dict[str, Any]],
    total_lines: int,
    min_segments: int = 5,
    max_segments: int = 20,
) -> List[str]:
    """Return validation problems, or an empty list when the segmentation is valid."""
    problems: List[str] = []
    if len(segments) < min_segments:
        problems.append(f"expected at least {min_segments} segments, got {len(segments)}")
    if len(segments) > max_segments:
        problems.append(f"expected at most {max_segments} segments, got {len(segments)}")

    normalised = []
    for index, segment in enumerate(segments, start=1):
        try:
            title = str(segment.get("title", "")).strip()
            description = str(segment.get("description", "")).strip()
            start = int(segment.get("start_line"))
            end = int(segment.get("end_line"))
        except (TypeError, ValueError):
            problems.append(f"segment {index} has invalid line numbers")
            continue

        if not title:
            problems.append(f"segment {index} is missing title")
        if not description:
            problems.append(f"segment {index} is missing description")
        if start < 1 or end > total_lines or start > end:
            problems.append(
                f"segment {index} has invalid range {start}-{end} for total {total_lines} lines"
            )
        normalised.append((start, end, index))

    expected_start = 1
    for start, end, original_index in sorted(normalised):
        if start != expected_start:
            if start > expected_start:
                problems.append(
                    f"gap before segment {original_index}: expected line {expected_start}, got {start}"
                )
            else:
                problems.append(
                    f"overlap at segment {original_index}: starts at {start}, expected {expected_start}"
                )
        expected_start = max(expected_start, end + 1)

    if total_lines > 0 and expected_start <= total_lines:
        problems.append(f"missing coverage after line {expected_start - 1}; final line is {total_lines}")

    return problems


def save_segments(path: Path | str, segments: List[Dict[str, Any]]) -> Path:
    """Save BGM segments as a JSON array."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sorted_segments(segments), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_segments(path: Path | str) -> List[Dict[str, Any]]:
    """Load cached BGM segments."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("segments", [])
    if not isinstance(raw, list):
        raise SegmentationError(f"Invalid segment cache format: {path}")
    return [_normalise_segment(segment) for segment in raw]


def _execute_tool_calls(
    tool_calls: List[ToolCall],
    index: NovelSegmentationIndex,
    segments: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    min_segments: int,
    max_segments: int,
) -> Optional[List[Dict[str, Any]]]:
    finished_segments = None

    for tool_call in tool_calls:
        content = _execute_tool(tool_call, index, segments, min_segments, max_segments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": content,
        })

        if tool_call.name == "finish_segmentation":
            problems = validate_segments(
                segments,
                total_lines=index.total_lines,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            if not problems:
                finished_segments = _sorted_segments(segments)

    return finished_segments


def _execute_tool(
    tool_call: ToolCall,
    index: NovelSegmentationIndex,
    segments: List[Dict[str, Any]],
    min_segments: int,
    max_segments: int,
) -> str:
    args = tool_call.arguments
    try:
        if tool_call.name == "get_novel_text":
            result = index.get_novel_text(args.get("start_line", 1), args.get("end_line", 1))
            return json.dumps(result, ensure_ascii=False)

        if tool_call.name == "get_dialogues":
            result = index.get_dialogues(args.get("start_line", 1), args.get("end_line", 1))
            return json.dumps(result, ensure_ascii=False)

        if tool_call.name == "submit_segment":
            segment = _normalise_segment(args)
            segments.append(segment)
            return f"Accepted segment {len(segments)}: {segment['start_line']}-{segment['end_line']}"

        if tool_call.name == "list_segments":
            return json.dumps(_sorted_segments(segments), ensure_ascii=False)

        if tool_call.name == "update_segment":
            segment_index = int(args.get("segment_index", 0))
            if segment_index < 1 or segment_index > len(segments):
                return f"Invalid segment_index {segment_index}; there are {len(segments)} segments"
            current = dict(segments[segment_index - 1])
            for key in ("title", "description", "start_line", "end_line"):
                if key in args and args[key] is not None:
                    current[key] = args[key]
            segments[segment_index - 1] = _normalise_segment(current)
            return f"Updated segment {segment_index}"

        if tool_call.name == "delete_segment":
            segment_index = int(args.get("segment_index", 0))
            if segment_index < 1 or segment_index > len(segments):
                return f"Invalid segment_index {segment_index}; there are {len(segments)} segments"
            removed = segments.pop(segment_index - 1)
            return f"Deleted segment {segment_index}: {removed['title']}"

        if tool_call.name == "finish_segmentation":
            problems = validate_segments(
                segments,
                total_lines=index.total_lines,
                min_segments=min_segments,
                max_segments=max_segments,
            )
            if problems:
                return "Cannot finish yet:\n- " + "\n- ".join(problems)
            return json.dumps({"segments": _sorted_segments(segments)}, ensure_ascii=False)
    except (TypeError, ValueError, KeyError) as exc:
        return f"Tool error for {tool_call.name}: {exc}"

    return f"Unknown tool: {tool_call.name}"


def _assistant_tool_message(result: ChatResult) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for tool_call in result.tool_calls
        ],
    }


def _normalise_segment(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_line": int(raw["start_line"]),
        "end_line": int(raw["end_line"]),
        "title": str(raw["title"]).strip(),
        "description": str(raw["description"]).strip(),
    }


def _sorted_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_normalise_segment(segment) for segment in sorted(segments, key=lambda item: int(item["start_line"]))]


def _extract_segments_from_content(content: str) -> List[Dict[str, Any]]:
    if not content:
        return []

    candidates = [content]
    match = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        candidates.insert(0, match.group(1))
    match = re.search(r"(\{.*\}|\[.*\])", content, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            raw = raw.get("segments", [])
        if isinstance(raw, list):
            try:
                return [_normalise_segment(segment) for segment in raw]
            except (TypeError, ValueError, KeyError):
                return []
    return []
