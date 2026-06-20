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
    result: Dict[str, Any] = {
        "start_line": int(raw["start_line"]),
        "end_line": int(raw["end_line"]),
        "title": str(raw["title"]).strip(),
        "description": str(raw["description"]).strip(),
    }
    # Preserve BGM type annotations (set by label_bgm_types)
    for key in ("bgm_type", "bgm_type_zh"):
        if key in raw and raw[key] is not None:
            result[key] = raw[key]
    return result


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


# ─── Multi-agent chunked segmentation ─────────────────────────────

CHUNK_SYSTEM_PROMPT = """You are a novel scene segmentation agent. Given a section of a novel, identify each distinct scene or event and output them as a JSON array.

Each segment must include:
- start_line: 1-based line number where the scene starts
- end_line: 1-based line number where the scene ends (inclusive)
- title: short Chinese scene title (5-15 chars)
- description: brief Chinese description of the scene's mood and atmosphere for BGM selection

Rules:
- Segment by plot/event/scene transitions, NOT by equal line count
- Every line must be covered by exactly one segment (no gaps, no overlaps)
- Lines are 1-based relative to the beginning of this section
- The section starts at line 1

Output ONLY valid JSON array, no other text. Example:
[
  {"start_line": 1, "end_line": 30, "title": "开场", "description": "主角登场，轻松明快的氛围"},
  {"start_line": 31, "end_line": 80, "title": "冲突", "description": "矛盾爆发，紧张激烈的对峙"}
]"""


def segment_chunk_direct(
    chunk_text: str,
    base_line: int,
    client: OllamaClient,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Segment one chunk by directly prompting the LLM (no tool calls).

    The chunk text is included in the prompt; the LLM returns JSON segments
    in one response.
    """
    chunk_lines = len(chunk_text.splitlines())
    message = {
        "role": "user",
        "content": (
            f"This section has {chunk_lines} lines (1-based, starting from line 1).\n\n"
            f"Novel text:\n{chunk_text}\n\n"
            "Output the scene segments as a JSON array:"
        ),
    }

    for attempt in range(max_retries):
        # Use chat without tools - send the prompt directly
        result = client.chat(
            messages=[{"role": "system", "content": CHUNK_SYSTEM_PROMPT}, message],
            temperature=temperature,
        )

        segments = _extract_segments_from_content(result.content)
        if not segments:
            if attempt < max_retries - 1:
                continue
            raise SegmentationError(f"Could not parse segments from LLM output after {max_retries} attempts")

        # Validate within-chunk coverage
        problems = validate_segments(segments, total_lines=chunk_lines, min_segments=1, max_segments=20)
        if not problems:
            return _sorted_segments(segments)

        # Remap error line numbers to chunk-local for clarity before retry
        p_str = "; ".join(problems)
        message = {"role": "user", "content": f"Validation errors: {p_str}. Please fix and output corrected JSON."}

    raise SegmentationError("Failed to produce valid segments after retries")


def _chunk_novel(
    total_lines: int,
    num_chunks: int = 4,
    overlap: int = 10,
) -> List[Tuple[int, int]]:
    """Split a novel into roughly equal chunks with overlap.

    Returns list of (chunk_start, chunk_end) where both are 1-based.
    The overlap lines are shared between adjacent chunks.
    """
    if num_chunks <= 1:
        return [(1, total_lines)]

    base = total_lines // num_chunks
    extra = total_lines % num_chunks

    chunks: List[Tuple[int, int]] = []
    current = 1
    for i in range(num_chunks):
        size = base + (1 if i < extra else 0)
        chunk_start = current
        chunk_end = min(total_lines, current + size - 1 + (overlap if i < num_chunks - 1 else 0))
        chunks.append((chunk_start, chunk_end))
        current += size

    # ensure last chunk reaches the final line
    chunks[-1] = (chunks[-1][0], total_lines)
    return chunks


def segment_novel_chunked(
    text: str,
    dialogues: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OllamaClient] = None,
    num_chunks: int = 6,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Segment a novel using multiple independent agent calls (same model).

    The novel is split into chunks; each chunk is sent to the LLM in one
    prompt (no iterative tool calling). Results are merged, deduplicated
    at overlaps, and validated.

    Args:
        text: Full novel text.
        dialogues: Not used in this direct-prompt mode (kept for API compat).
        client: Ollama client.
        num_chunks: How many chunks to divide the novel into.
        temperature: LLM temperature.

    Returns:
        Validated, sorted list of segments covering the full novel.
    """
    client = client or OllamaClient()
    lines = text.splitlines()
    total_lines = len(lines)
    chunks = _chunk_novel(total_lines, num_chunks, overlap=10)
    all_segments: List[Dict[str, Any]] = []

    print(f"  Multi-agent: {num_chunks} chunks, {total_lines} total lines")

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_lines = chunk_end - chunk_start + 1
        chunk_text = "\n".join(lines[chunk_start - 1 : chunk_end])
        print(f"    Chunk {chunk_index + 1}: lines {chunk_start}-{chunk_end} ({chunk_lines} lines)")

        try:
            chunk_segments = segment_chunk_direct(
                chunk_text=chunk_text,
                base_line=chunk_start,
                client=client,
                temperature=temperature,
            )
        except SegmentationError as exc:
            print(f"    Chunk {chunk_index + 1} failed: {exc}")
            raise

        # remap line numbers to full novel
        for seg in chunk_segments:
            seg["start_line"] += chunk_start - 1
            seg["end_line"] += chunk_start - 1
            all_segments.append(seg)

        print(f"    Chunk {chunk_index + 1}: {len(chunk_segments)} segments")

    # merge: at overlap boundaries, deduplicate
    all_segments.sort(key=lambda s: s["start_line"])
    merged = []
    for seg in all_segments:
        if merged and seg["start_line"] <= merged[-1]["end_line"]:
            # overlap: extend if this segment covers more
            if seg["end_line"] > merged[-1]["end_line"]:
                merged[-1]["end_line"] = seg["end_line"]
        else:
            merged.append(seg)

    # fill any remaining gaps between segments
    for i in range(len(merged) - 1):
        if merged[i + 1]["start_line"] > merged[i]["end_line"] + 1:
            merged[i]["end_line"] = merged[i + 1]["start_line"] - 1

    # final validation on full novel
    problems = validate_segments(
        merged,
        total_lines=total_lines,
        min_segments=num_chunks,
        max_segments=num_chunks * 8,
    )
    if problems:
        raise SegmentationError("Merged segments invalid: " + "; ".join(problems))

    print(f"  Merged: {len(merged)} segments covering {total_lines} lines")
    return _sorted_segments(merged)


# ─── BGM type labeling (Stage 7-b) ──────────────────────────────

BGM_TYPE_PROMPT = """You are a BGM type classification agent for a novel audiobook project.

You will be given a list of scene segments from a novel. For each segment, classify
its atmosphere into exactly one of these BGM types:

- daily: 日常 — 轻松、温馨、日常聊天
- suspense: 悬疑 — 紧张、神秘、诡异
- battle: 战斗 — 热血、激昂、紧张
- sad: 悲伤 — 忧伤、沉重、感人
- romantic: 浪漫 — 温柔、甜蜜、抒情
- epic: 史诗 — 宏大、壮丽、震撼
- comedy: 滑稽 — 搞笑、欢快、调皮
- horror: 恐怖 — 阴森、惊悚、不安

Output a JSON array with the same number of items as input. Each item:
  {"segment_index": <1-based index>, "bgm_type": "<type>"}

Output ONLY valid JSON, no other text."""


BGM_TYPE_MAP_ZH = {
    "daily": "日常",
    "suspense": "悬疑",
    "battle": "战斗",
    "sad": "悲伤",
    "romantic": "浪漫",
    "epic": "史诗",
    "comedy": "滑稽",
    "horror": "恐怖",
}

BGM_TYPE_MAP_EN = {v: k for k, v in BGM_TYPE_MAP_ZH.items()}


def label_bgm_types(
    segments: List[Dict[str, Any]],
    novel_text: str = "",
    client: Optional[OllamaClient] = None,
    batch_size: int = 50,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Classify each segment's atmosphere type using the LLM.

    Segments are sent in batches of ``batch_size``. Each batch gets one
    LLM call.

    Args:
        segments: List of segment dicts with title, description, start_line, end_line.
        novel_text: Full novel text (optional — used for extra context).
        client: Ollama client.
        batch_size: Max segments per API call.
        temperature: LLM temperature.

    Returns:
        Segments with ``bgm_type`` (English key) and ``bgm_type_zh`` (Chinese) added.
    """
    client = client or OllamaClient()
    result_segments = [dict(seg) for seg in _sorted_segments(segments)]

    for batch_start in range(0, len(result_segments), batch_size):
        batch = result_segments[batch_start:batch_start + batch_size]
        batch_info = [
            {"segment_index": i + 1, "title": s["title"], "description": s["description"]}
            for i, s in enumerate(batch)
        ]

        prompt = (
            f"Classify these {len(batch)} scene segments:\n\n"
            f"{json.dumps(batch_info, ensure_ascii=False, indent=2)}\n\n"
            "Output exactly the same count of items with bgm_type added."
        )

        result = client.chat(
            messages=[{"role": "system", "content": BGM_TYPE_PROMPT}, {"role": "user", "content": prompt}],
            temperature=temperature,
        )

        types = _extract_bgm_types(result.content, len(batch))
        if not types:
            # fallback: all daily
            types = [{"segment_index": i + 1, "bgm_type": "daily"} for i in range(len(batch))]

        for item in types:
            idx = item["segment_index"] - 1
            if 0 <= idx < len(batch):
                bgm_type = item.get("bgm_type", "daily")
                bgm_type_zh = BGM_TYPE_MAP_ZH.get(bgm_type, "日常")
                batch[idx]["bgm_type"] = bgm_type
                batch[idx]["bgm_type_zh"] = bgm_type_zh

    return result_segments


def _extract_bgm_types(content: str, expected: int) -> Optional[List[Dict[str, Any]]]:
    """Extract BGM type classification from LLM output."""
    if not content:
        return None

    candidates = [content]
    match = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        candidates.insert(0, match.group(1))
    match = re.search(r"(\[.*\])", content, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, list) and len(raw) == expected:
            valid = True
            for item in raw:
                if not isinstance(item, dict) or "segment_index" not in item or "bgm_type" not in item:
                    valid = False
                    break
            if valid:
                return raw
    return None
