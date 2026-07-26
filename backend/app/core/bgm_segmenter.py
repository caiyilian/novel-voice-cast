"""BGM scene segmentation and evidence-grounded type classification."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from app.core.llm_client import LLMClient, LLMResult, SENSENOVA_FLASH_LITE_MODEL, ToolCall

logger = logging.getLogger("bgm_segmenter")

DEFAULT_OUTPUT_PATH = Path("backend/data/bgm_segments.json")
DEFAULT_SEGMENT_CHECKPOINT = Path("backend/data/bgm_segmentation.checkpoint.json")
DEFAULT_TYPE_CHECKPOINT = Path("backend/data/bgm_types.checkpoint.json")
BGM_SEGMENTATION_PIPELINE_VERSION = 2
BGM_TYPE_PIPELINE_VERSION = 2
_USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


class SegmentationError(RuntimeError):
    pass


def bgm_source_hash(text: str) -> str:
    """Return the novel identity shared by both BGM LLM stages."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _flash_lite_model(client: LLMClient) -> str:
    model_name = str(getattr(client, "sensenova_model", SENSENOVA_FLASH_LITE_MODEL))
    if model_name != SENSENOVA_FLASH_LITE_MODEL:
        raise SegmentationError(
            f"BGM LLM stages require {SENSENOVA_FLASH_LITE_MODEL}, got {model_name}"
        )
    return model_name


def _normalise_usage_summary(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raw = {}
    usage: dict[str, int] = {}
    for key in _USAGE_KEYS:
        try:
            usage[key] = max(0, int(raw.get(key, 0) or 0))
        except (TypeError, ValueError):
            usage[key] = 0
    return usage


def _usage_delta(current: Any, starting: Any) -> dict[str, int]:
    current_usage = _normalise_usage_summary(current)
    starting_usage = _normalise_usage_summary(starting)
    return {key: max(0, current_usage[key] - starting_usage[key]) for key in _USAGE_KEYS}


def _merge_usage_summaries(left: Any, right: Any) -> dict[str, int]:
    left_usage = _normalise_usage_summary(left)
    right_usage = _normalise_usage_summary(right)
    return {key: left_usage[key] + right_usage[key] for key in _USAGE_KEYS}


def _cumulative_usage(client: LLMClient, previous: Any, starting: Any) -> dict[str, int]:
    return _merge_usage_summaries(previous, _usage_delta(client.usage_summary(), starting))


def _segment_inputs_hash(segments: list[dict[str, Any]]) -> str:
    stable_inputs = [
        {
            key: segment.get(key)
            for key in ("start_line", "end_line", "title", "description")
        }
        for segment in segments
    ]
    encoded = json.dumps(
        stable_inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _add_segmentation_metadata(
    segments: list[dict[str, Any]], source_hash: str, model_name: str
) -> list[dict[str, Any]]:
    for segment in segments:
        segment.update(
            {
                "segmentation_pipeline_version": BGM_SEGMENTATION_PIPELINE_VERSION,
                "segmentation_source_hash": source_hash,
                "segmentation_model": model_name,
            }
        )
    return segments


class NovelSegmentationIndex:
    def __init__(self, text: str, dialogues: Optional[list[dict[str, Any]]] = None):
        self.lines = text.splitlines()
        self.dialogues = dialogues or []

    def get_novel_text(self, start_line: int, end_line: int, limit: int = 600) -> dict[str, Any]:
        start = max(1, int(start_line))
        end = min(len(self.lines), int(end_line))
        selected = self.lines[start - 1 : end]
        truncated = len(selected) > limit
        selected = selected[:limit]
        return {
            "start_line": start,
            "end_line": start + len(selected) - 1,
            "truncated": truncated,
            "text": "\n".join(f"{start + offset}: {line}" for offset, line in enumerate(selected)),
        }

    def get_dialogues(self, start_line: int, end_line: int, limit: int = 200) -> dict[str, Any]:
        start = max(1, int(start_line))
        end = min(len(self.lines), int(end_line))
        selected = []
        for dialogue_index, dialogue in enumerate(self.dialogues, 1):
            line = int(dialogue.get("line", 0) or 0)
            if start <= line <= end:
                selected.append({
                    "dialogue_index": dialogue_index,
                    "line": line,
                    "speaker": dialogue.get("speaker", ""),
                    "text": dialogue.get("text", ""),
                })
        return {
            "total_dialogues": len(selected),
            "truncated": len(selected) > limit,
            "dialogues": selected[:limit],
        }


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_novel_text",
            "description": "Read a bounded source range with absolute line numbers.",
            "strict": True,
            "parameters": _schema(
                {"start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}},
                ["start_line", "end_line"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dialogues",
            "description": "Read speaker-labelled dialogue metadata in a source range.",
            "strict": True,
            "parameters": _schema(
                {"start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}},
                ["start_line", "end_line"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_segment",
            "description": "Add one contiguous scene segment.",
            "strict": True,
            "parameters": _schema(
                {
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "title": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                },
                ["start_line", "end_line", "title", "description"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_segments",
            "description": "List current segments and coverage problems.",
            "strict": True,
            "parameters": _schema({}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_segment",
            "description": "Replace one existing 1-based segment.",
            "strict": True,
            "parameters": _schema(
                {
                    "segment_index": {"type": "integer", "minimum": 1},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "title": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                },
                ["segment_index", "start_line", "end_line", "title", "description"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_segment",
            "description": "Delete one existing 1-based segment.",
            "strict": True,
            "parameters": _schema({"segment_index": {"type": "integer", "minimum": 1}}, ["segment_index"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_segmentation",
            "description": "Finish only after list_segments reports valid complete coverage.",
            "strict": True,
            "parameters": _schema({}, []),
        },
    },
]

SYSTEM_PROMPT = """You are the BGM scene editor for a complete audiobook volume.
Create contiguous segments around real changes in location, time, active goal,
conflict, emotional energy, or narrative mode. Do not divide by equal length. Do not
split a continuous exchange merely because it is long; do split when the musical
function changes materially. Titles and descriptions must summarize source facts and
the audible mood, not invent visuals or events.

Use the reading tools to inspect the complete volume in bounded ranges. Build 5-20
segments unless source structure clearly requires otherwise. Every source line must
belong to exactly one segment: start at line 1, no gaps or overlaps, and end at the
reported final line. Use list_segments to audit coverage before finish_segmentation.
"""


def segment_novel(
    text: str,
    dialogues: Optional[list[dict[str, Any]]] = None,
    client: Optional[LLMClient] = None,
    min_segments: int = 5,
    max_segments: int = 20,
    max_tool_steps: int = 80,
    temperature: float = 0.15,
) -> list[dict[str, Any]]:
    client = client or LLMClient.for_flash_lite("bgm_segmentation")
    model_name = _flash_lite_model(client)
    source_hash = bgm_source_hash(text)
    index = NovelSegmentationIndex(text, dialogues)
    total_lines = len(index.lines)
    segments: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Segment this volume. It has exactly {total_lines} source lines. Begin by inspecting the text."},
    ]
    for step in range(1, max_tool_steps + 1):
        result = client.chat(
            messages,
            tools=TOOL_SPECS,
            temperature=temperature,
            agent_role="bgm_full_segmenter",
            trace_id="bgm:full-volume",
            agent_round=step,
        )
        if not result.tool_calls:
            parsed = _extract_segments_from_content(result.content)
            if parsed and not validate_segments(parsed, total_lines, min_segments, max_segments):
                output = _sorted_segments(parsed)
                return _add_segmentation_metadata(output, source_hash, model_name)
            messages.extend([
                {"role": "assistant", "content": result.content or ""},
                {"role": "user", "content": "Continue with tools; finish only after complete coverage validation."},
            ])
            continue
        messages.append(_assistant_tool_message(result))
        finished = False
        for call in result.tool_calls:
            output, did_finish = _execute_tool(call, index, segments, min_segments, max_segments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            finished = finished or did_finish
        if finished:
            problems = validate_segments(segments, total_lines, min_segments, max_segments)
            if not problems:
                output = _sorted_segments(segments)
                return _add_segmentation_metadata(output, source_hash, model_name)
    raise SegmentationError(f"Agent did not finish a valid segmentation in {max_tool_steps} calls")


def validate_segments(
    segments: list[dict[str, Any]],
    total_lines: int,
    min_segments: int = 5,
    max_segments: int = 20,
) -> list[str]:
    problems = []
    if not isinstance(segments, list):
        return ["segments must be a list"]
    if not min_segments <= len(segments) <= max_segments:
        problems.append(f"segment count {len(segments)} is outside {min_segments}-{max_segments}")
    normalised = []
    for position, segment in enumerate(segments, 1):
        if not isinstance(segment, dict):
            problems.append(f"segment {position} is not an object")
            continue
        try:
            start = int(segment.get("start_line"))
            end = int(segment.get("end_line"))
        except (TypeError, ValueError):
            problems.append(f"segment {position} has non-integer boundaries")
            continue
        if start < 1 or end < start or end > total_lines:
            problems.append(f"segment {position} has invalid range {start}-{end}")
        if not str(segment.get("title", "")).strip():
            problems.append(f"segment {position} has no title")
        if not str(segment.get("description", "")).strip():
            problems.append(f"segment {position} has no description")
        normalised.append((start, end, position))
    normalised.sort()
    if normalised:
        if normalised[0][0] != 1:
            problems.append(f"coverage starts at {normalised[0][0]}, expected 1")
        if normalised[-1][1] != total_lines:
            problems.append(f"coverage ends at {normalised[-1][1]}, expected {total_lines}")
        for previous, current in zip(normalised, normalised[1:]):
            if current[0] <= previous[1]:
                problems.append(f"segments {previous[2]} and {current[2]} overlap")
            elif current[0] != previous[1] + 1:
                problems.append(f"gap between lines {previous[1]} and {current[0]}")
    elif total_lines:
        problems.append("no source coverage")
    return problems


def save_segments(path: Path | str, segments: list[dict[str, Any]]) -> Path:
    output = Path(path)
    _atomic_write_json(output, segments)
    return output


def load_segments(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SegmentationError("Segment cache must contain a JSON array")
    return payload


def _execute_tool(
    call: ToolCall,
    index: NovelSegmentationIndex,
    segments: list[dict[str, Any]],
    min_segments: int,
    max_segments: int,
) -> tuple[str, bool]:
    if call.name == "get_novel_text":
        payload = index.get_novel_text(call.arguments.get("start_line", 1), call.arguments.get("end_line", 1))
        return json.dumps(payload, ensure_ascii=False), False
    if call.name == "get_dialogues":
        payload = index.get_dialogues(call.arguments.get("start_line", 1), call.arguments.get("end_line", 1))
        return json.dumps(payload, ensure_ascii=False), False
    if call.name == "submit_segment":
        segment = _normalise_segment(call.arguments)
        if segment["end_line"] < segment["start_line"]:
            return "Rejected: end_line precedes start_line", False
        segments.append(segment)
        return f"Accepted segment {len(segments)}", False
    if call.name == "list_segments":
        payload = {"segments": _sorted_segments(segments), "problems": validate_segments(segments, len(index.lines), min_segments, max_segments)}
        return json.dumps(payload, ensure_ascii=False), False
    if call.name == "update_segment":
        position = int(call.arguments.get("segment_index", 0)) - 1
        if not 0 <= position < len(segments):
            return "Rejected: segment_index does not exist", False
        segments[position] = _normalise_segment(call.arguments)
        return f"Updated segment {position + 1}", False
    if call.name == "delete_segment":
        position = int(call.arguments.get("segment_index", 0)) - 1
        if not 0 <= position < len(segments):
            return "Rejected: segment_index does not exist", False
        segments.pop(position)
        return f"Deleted segment {position + 1}", False
    if call.name == "finish_segmentation":
        problems = validate_segments(segments, len(index.lines), min_segments, max_segments)
        return ("Accepted: segmentation is complete", True) if not problems else ("Rejected: " + "; ".join(problems), False)
    return f"Unknown tool: {call.name}", False


def _assistant_tool_message(result: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in result.tool_calls
        ],
    }


def _normalise_segment(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_line": int(raw.get("start_line", 0)),
        "end_line": int(raw.get("end_line", 0)),
        "title": str(raw.get("title", "")).strip(),
        "description": str(raw.get("description", "")).strip(),
    }


def _sorted_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((_normalise_segment(segment) | {key: value for key, value in segment.items() if key not in {"start_line", "end_line", "title", "description"}} for segment in segments), key=lambda item: item["start_line"])


def _extract_json_array(content: str) -> Optional[list[Any]]:
    content = content.strip()
    candidates = [content]
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = content.find("["), content.rfind("]")
    if start >= 0 and end > start:
        candidates.append(content[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _extract_segments_from_content(content: str) -> list[dict[str, Any]]:
    value = _extract_json_array(content)
    if not value:
        return []
    try:
        return [_normalise_segment(item) for item in value if isinstance(item, dict)]
    except (TypeError, ValueError):
        return []


CHUNK_SYSTEM_PROMPT = """Segment only the OWNED source lines into BGM scenes.
The source is line-numbered. Context outside the owned range is read-only and must
not be included in output. A new segment means the audiobook's music should actually
change. Keep plot beats, dialogue exchanges, title cards, and travel details together
when their musical function remains continuous. Split on a material change in mood,
energy, danger, location/time, point of view, or narrative mode. Aim for a cue change
about every 12-24 source lines when the story supports it. For roughly 120-180 owned
lines, prefer 6-14 musically useful segments. Avoid segments shorter than 6 lines
unless the source makes an abrupt tonal turn, and avoid segments longer than 35 lines
unless they are genuinely one sustained scene. Avoid
equal-size divisions and invented facts. Write title and description in Chinese.
Each segment must have one dominant musical function. If its description would say
the music changes from one mood/function to another, split at that change instead.
Return only a JSON array. Every owned line must be covered exactly once with no gaps
or overlap. Each item requires start_line, end_line, title, and description, using
absolute source line numbers.
"""

CHUNK_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_chunk_segmentation",
        "description": "Submit the complete, validated segmentation for the owned source range.",
        "strict": True,
        "parameters": _schema(
            {
                "segments": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": _schema(
                        {
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "title": {"type": "string", "minLength": 1},
                            "description": {"type": "string", "minLength": 1},
                        },
                        ["start_line", "end_line", "title", "description"],
                    ),
                }
            },
            ["segments"],
        ),
    },
}]

CHUNK_PRIMARY_MAX_TOKENS = 8000
CHUNK_REVIEW_MAX_TOKENS = 8000
CHUNK_REVIEW_ATTEMPTS = 6


def _chunk_segments_from_result(result: LLMResult) -> list[dict[str, Any]]:
    for call in result.tool_calls:
        if call.name == "submit_chunk_segmentation":
            try:
                return [_normalise_segment(item) for item in call.arguments.get("segments", [])]
            except (TypeError, ValueError):
                return []
    return _extract_segments_from_content(result.content)


def _musical_coherence_problems(segments: list[dict[str, Any]]) -> list[str]:
    problems = []
    patterns = (
        r"\u97f3\u4e50.{0,40}(?:\u8f6c\u4e3a|\u5207\u6362|\u8f6c\u5165|\u8f6c\u5411|\u53d8\u5316)",
        r"(?:\u6c1b\u56f4|\u60c5\u7eea).{0,30}(?:\u4ece.+\u8f6c\u4e3a|\u5728.+\u95f4\u5207\u6362)",
        r"\bmusic\b.{0,80}\b(?:shifts?|changes?|transitions?|switches?)\b",
    )
    for index, segment in enumerate(segments, 1):
        description = str(segment.get("description", "")).replace("\u97f3\u4e50\u5e94\u8f6c\u4e3a", "\u97f3\u4e50\u5e94\u4e3a")
        if any(re.search(pattern, description, re.IGNORECASE) for pattern in patterns):
            problems.append(
                f"segment {index} describes an internal music transition; split it so each segment has one musical function"
            )
    return problems


def _merge_title_card_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach short act/chapter title cards to the following musical scene."""
    output = [dict(segment) for segment in _sorted_segments(segments)]
    index = 0
    while index < len(output):
        segment = output[index]
        length = segment["end_line"] - segment["start_line"] + 1
        label = f"{segment.get('title', '')} {segment.get('description', '')}"
        is_title_card = length <= 10 and (
            "\u6807\u9898" in label
            or "\u6807\u9898\u5361" in label
            or "title card" in label.lower()
            or re.search(r"\u7b2c[^\s]{1,6}\u5e55", str(segment.get("title", "")))
        )
        if not is_title_card:
            index += 1
            continue
        if index + 1 < len(output):
            following = output[index + 1]
            following["start_line"] = segment["start_line"]
            following["merged_title_card"] = segment.get("title", "")
            output.pop(index)
            continue
        if index > 0:
            output[index - 1]["end_line"] = segment["end_line"]
            output[index - 1]["merged_title_card"] = segment.get("title", "")
            output.pop(index)
            continue
        index += 1
    return output


def segment_chunk_direct(
    chunk_text: str,
    base_line: int,
    client: LLMClient,
    temperature: float = 0.15,
    max_retries: int = 8,
    context_before: str = "",
    context_after: str = "",
    previous_memory: str = "",
) -> list[dict[str, Any]]:
    """Segment one owned chunk; return chunk-local line numbers for compatibility."""
    raw_lines = chunk_text.splitlines()
    chunk_lines = len(raw_lines)
    owned_end = base_line + chunk_lines - 1
    target_min = max(1 if chunk_lines < 36 else 2, math.ceil(chunk_lines / 26))
    target_max = max(target_min, min(20, max(3, math.ceil(chunk_lines / 11))))
    numbered = "\n".join(f"{base_line + offset}: {line}" for offset, line in enumerate(raw_lines))
    prompt = (
        f"OWNED RANGE: {base_line}-{owned_end}\n"
        f"REQUIRED MUSICAL SEGMENT COUNT: {target_min}-{target_max}\n"
        f"Previous scene memory: {previous_memory or 'none'}\n"
        f"Read-only context before:\n{context_before or 'none'}\n\n"
        f"OWNED SOURCE:\n{numbered}\n\n"
        f"Read-only context after:\n{context_after or 'none'}"
    )
    base_messages = [{"role": "system", "content": CHUNK_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    messages = list(base_messages)
    last_problems = ["no response"]
    candidate: Optional[list[dict[str, Any]]] = None
    coherence_corrections = 0
    trace_id = f"bgm:chunk:{base_line}-{owned_end}"
    for step in range(1, max_retries + 1):
        result = client.chat(
            messages,
            tools=CHUNK_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_chunk_segmentation"}},
            temperature=temperature,
            max_tokens=CHUNK_PRIMARY_MAX_TOKENS,
            agent_role="bgm_chunk_primary",
            trace_id=trace_id,
            agent_round=step,
        )
        absolute = _chunk_segments_from_result(result)
        local = [
            dict(segment, start_line=segment["start_line"] - base_line + 1, end_line=segment["end_line"] - base_line + 1)
            for segment in absolute
        ]
        hard_problems = validate_segments(local, chunk_lines, min_segments=target_min, max_segments=target_max)
        coherence_warnings = _musical_coherence_problems(local)
        last_problems = hard_problems + coherence_warnings
        if absolute and not hard_problems and (not coherence_warnings or coherence_corrections >= 1):
            candidate = _sorted_segments(local)
            if coherence_warnings:
                logger.warning("Chunk primary retained %d musical-coherence warnings after correction", len(coherence_warnings))
            break
        if absolute and not hard_problems and coherence_warnings:
            coherence_corrections += 1
        previous = json.dumps(absolute, ensure_ascii=False) if absolute else (result.content or "")[-2000:]
        messages = list(base_messages) + [{
            "role": "user",
            "content": "A previous independent attempt was rejected: " + "; ".join(last_problems)
            + "\nPrevious candidate:\n" + previous
            + "\nCreate a fresh corrected complete tool call. Do not preserve a bad boundary just to resemble the candidate.",
        }]
    if candidate is None:
        raise SegmentationError("Chunk validation failed: " + "; ".join(last_problems))

    review_base_messages = [
        {
            "role": "system",
            "content": CHUNK_SYSTEM_PROMPT
            + "\nYou are the second-pass editor. Independently audit every proposed boundary against the source. "
            "Return a complete corrected array, even when no changes are needed. Do not narrate your reasoning: "
            "immediately call submit_chunk_segmentation once, with concise titles and descriptions.",
        },
        {
            "role": "user",
            "content": prompt
            + "\n\nFIRST AGENT CANDIDATE (absolute source line numbers):\n"
            + json.dumps([
                dict(segment, start_line=segment["start_line"] + base_line - 1, end_line=segment["end_line"] + base_line - 1)
                for segment in candidate
            ], ensure_ascii=False),
        },
    ]
    review_messages = list(review_base_messages)
    review_problems = ["no review response"]
    review_coherence_corrections = 0
    for step in range(1, CHUNK_REVIEW_ATTEMPTS + 1):
        review = client.chat(
            review_messages,
            tools=CHUNK_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_chunk_segmentation"}},
            temperature=0.0,
            max_tokens=CHUNK_REVIEW_MAX_TOKENS,
            agent_role="bgm_chunk_reviewer",
            trace_id=trace_id,
            agent_round=step,
        )
        reviewed_absolute = _chunk_segments_from_result(review)
        reviewed_local = [
            dict(segment, start_line=segment["start_line"] - base_line + 1, end_line=segment["end_line"] - base_line + 1)
            for segment in reviewed_absolute
        ]
        review_hard_problems = validate_segments(reviewed_local, chunk_lines, min_segments=target_min, max_segments=target_max)
        review_coherence_warnings = _musical_coherence_problems(reviewed_local)
        review_problems = review_hard_problems + review_coherence_warnings
        if reviewed_absolute and not review_hard_problems and (
            not review_coherence_warnings or review_coherence_corrections >= 1
        ):
            output = _sorted_segments(reviewed_local)
            for warning in review_coherence_warnings:
                logger.warning("Chunk reviewer warning retained: %s", warning)
            return output
        finish_reason = "unknown"
        choices = review.raw.get("choices") if isinstance(review.raw, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason") or "unknown")
        logger.warning(
            "Chunk reviewer rejected trace=%s attempt=%d/%d finish_reason=%s segments=%d: %s",
            trace_id,
            step,
            CHUNK_REVIEW_ATTEMPTS,
            finish_reason,
            len(reviewed_absolute),
            "; ".join(review_problems),
        )
        if reviewed_absolute and not review_hard_problems and review_coherence_warnings:
            review_coherence_corrections += 1
        previous = json.dumps(reviewed_absolute, ensure_ascii=False) if reviewed_absolute else (review.content or "")[-2000:]
        review_messages = list(review_base_messages) + [{
            "role": "user",
            "content": "The prior review was rejected: " + "; ".join(review_problems)
            + "\nRejected candidate:\n" + previous
            + "\nAudit again from source and submit a fresh complete corrected tool call.",
        }]
    # The primary candidate passed all hard coverage/count validation before review.
    # A reviewer that repeatedly truncates or emits an invalid tool call must not
    # discard that valid work or prevent the caller from checkpointing the chunk.
    logger.warning(
        "Chunk reviewer exhausted trace=%s after %d attempts; falling back to the hard-validated primary candidate: %s",
        trace_id,
        CHUNK_REVIEW_ATTEMPTS,
        "; ".join(review_problems),
    )
    return candidate


def _chunk_novel(total_lines: int, num_chunks: int = 4, overlap: int = 10) -> list[tuple[int, int]]:
    if total_lines <= 0:
        return []
    num_chunks = max(1, min(num_chunks, total_lines))
    base, extra = divmod(total_lines, num_chunks)
    ranges = []
    current = 1
    for index in range(num_chunks):
        size = base + (1 if index < extra else 0)
        end = current + size - 1
        ranges.append((current, min(total_lines, end + (overlap if index < num_chunks - 1 else 0))))
        current = end + 1
    return ranges


def _natural_chunks(lines: list[str], num_chunks: int) -> list[tuple[int, int]]:
    total = len(lines)
    if num_chunks <= 1:
        return [(1, total)]
    boundaries = [1]
    for part in range(1, num_chunks):
        target = round(total * part / num_chunks)
        low = max(boundaries[-1] + 1, target - 50)
        high = min(total - (num_chunks - part), target + 50)
        candidates = range(low, high + 1)
        best = min(candidates, key=lambda line: (0 if not lines[line - 1].strip() else 1, abs(line - target)))
        boundaries.append(best)
    boundaries.append(total + 1)
    return [(boundaries[i], boundaries[i + 1] - 1) for i in range(len(boundaries) - 1)]


BOUNDARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_boundary_review",
        "description": "Review whether two chunk-edge scenes are musically continuous.",
        "strict": True,
        "parameters": _schema(
            {
                "merge": {"type": "boolean"},
                "reason": {"type": "string", "minLength": 1},
                "merged_title": {"type": "string"},
                "merged_description": {"type": "string"},
            },
            ["merge", "reason", "merged_title", "merged_description"],
        ),
    },
}]


def _review_boundary(
    previous: dict[str, Any],
    current: dict[str, Any],
    lines: list[str],
    client: LLMClient,
) -> Optional[dict[str, Any]]:
    boundary = previous["end_line"]
    start, end = max(1, boundary - 60), min(len(lines), boundary + 60)
    context = "\n".join(f"{line}: {lines[line - 1]}" for line in range(start, end + 1))
    messages = [
        {"role": "system", "content": "Audit a chunk boundary. Merge only if both edge segments are one continuous event and musical mood; keep separate for a genuine transition. Use the review tool."},
        {"role": "user", "content": f"Left: {json.dumps(previous, ensure_ascii=False)}\nRight: {json.dumps(current, ensure_ascii=False)}\n\nSource:\n{context}"},
    ]
    for step in range(1, 4):
        result = client.chat(
            messages,
            tools=BOUNDARY_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_boundary_review"}},
            temperature=0.0,
            max_tokens=1400,
            agent_role="bgm_boundary_reviewer",
            trace_id=f"bgm:boundary:{boundary}",
            agent_round=step,
        )
        for call in result.tool_calls:
            if call.name == "submit_boundary_review" and isinstance(call.arguments.get("merge"), bool):
                return call.arguments
        messages.append({"role": "assistant", "content": result.content or ""})
        messages.append({"role": "user", "content": "Return a valid submit_boundary_review tool call."})
    return None


def _valid_checkpoint_chunks(
    raw_chunks: Any, ranges: list[tuple[int, int]]
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw_chunks, dict):
        return {}
    completed: dict[str, list[dict[str, Any]]] = {}
    for key, raw_segments in raw_chunks.items():
        try:
            chunk_index = int(key)
            start, end = ranges[chunk_index]
            if chunk_index < 0 or not isinstance(raw_segments, list):
                continue
            local = [
                dict(
                    segment,
                    start_line=int(segment["start_line"]) - start + 1,
                    end_line=int(segment["end_line"]) - start + 1,
                )
                for segment in raw_segments
                if isinstance(segment, dict)
            ]
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if len(local) != len(raw_segments):
            continue
        if validate_segments(local, end - start + 1, min_segments=1, max_segments=20):
            continue
        completed[str(chunk_index)] = [dict(segment) for segment in raw_segments]
    return completed


def segment_novel_chunked(
    text: str,
    dialogues: Optional[list[dict[str, Any]]] = None,
    client: Optional[LLMClient] = None,
    num_chunks: int = 6,
    temperature: float = 0.15,
    checkpoint_path: Path | str | None = DEFAULT_SEGMENT_CHECKPOINT,
    resume: bool = True,
) -> list[dict[str, Any]]:
    client = client or LLMClient.for_flash_lite("bgm_segmentation")
    model_name = _flash_lite_model(client)
    usage_at_start = _normalise_usage_summary(client.usage_summary())
    lines = text.splitlines()
    total_lines = len(lines)
    ranges = _natural_chunks(lines, max(1, min(num_chunks, total_lines)))
    ranges_payload = [list(item) for item in ranges]
    source_hash = bgm_source_hash(text)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed: dict[str, list[dict[str, Any]]] = {}
    previous_usage = _normalise_usage_summary({})
    if resume and checkpoint and checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            compatible = (
                payload.get("pipeline_version") == BGM_SEGMENTATION_PIPELINE_VERSION
                and payload.get("source_hash") == source_hash
                and payload.get("model") == model_name
                and payload.get("ranges") == ranges_payload
            )
            if compatible:
                completed = _valid_checkpoint_chunks(payload.get("chunks"), ranges)
                previous_usage = _normalise_usage_summary(payload.get("llm_usage"))
            else:
                logger.warning(
                    "Ignoring incompatible BGM segmentation checkpoint %s "
                    "(version/source/model/ranges changed)",
                    checkpoint,
                )
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid BGM checkpoint: %s", checkpoint)

    def write_checkpoint() -> None:
        if checkpoint:
            _atomic_write_json(
                checkpoint,
                {
                    "pipeline_version": BGM_SEGMENTATION_PIPELINE_VERSION,
                    "source_hash": source_hash,
                    "model": model_name,
                    "ranges": ranges_payload,
                    "chunks": completed,
                    "llm_usage": _cumulative_usage(client, previous_usage, usage_at_start),
                },
            )

    all_segments: list[dict[str, Any]] = []
    previous_memory = ""
    for chunk_index, (start, end) in enumerate(ranges):
        key = str(chunk_index)
        if key in completed:
            absolute = completed[key]
        else:
            local = segment_chunk_direct(
                "\n".join(lines[start - 1 : end]),
                base_line=start,
                client=client,
                temperature=temperature,
                context_before="\n".join(f"{line}: {lines[line - 1]}" for line in range(max(1, start - 60), start)),
                context_after="\n".join(f"{line}: {lines[line - 1]}" for line in range(end + 1, min(total_lines, end + 60) + 1)),
                previous_memory=previous_memory,
            )
            absolute = [dict(segment, start_line=segment["start_line"] + start - 1, end_line=segment["end_line"] + start - 1) for segment in local]
            completed[key] = absolute
            write_checkpoint()
        all_segments.extend(absolute)
        if absolute:
            previous_memory = json.dumps(absolute[-1], ensure_ascii=False)
        logger.info("BGM chunk %d/%d lines=%d-%d segments=%d", chunk_index + 1, len(ranges), start, end, len(absolute))

    merged = _merge_title_card_segments(all_segments)
    chunk_boundaries = {end for _, end in ranges[:-1]}
    position = 0
    while position < len(merged) - 1:
        left, right = merged[position], merged[position + 1]
        if left["end_line"] in chunk_boundaries:
            decision = _review_boundary(left, right, lines, client)
            if decision and decision.get("merge"):
                merged_candidate = {
                    "start_line": left["start_line"],
                    "end_line": right["end_line"],
                    "title": str(decision.get("merged_title") or left["title"]),
                    "description": str(decision.get("merged_description") or f"{left['description']} {right['description']}").strip(),
                    "boundary_review": str(decision.get("reason", "")),
                }
                if _musical_coherence_problems([merged_candidate]):
                    position += 1
                    continue
                merged[position] = merged_candidate
                merged.pop(position + 1)
                continue
        position += 1

    problems = validate_segments(merged, total_lines, min_segments=1, max_segments=max(20, num_chunks * 20))
    if problems:
        raise SegmentationError("Merged segments invalid: " + "; ".join(problems))
    write_checkpoint()
    return _add_segmentation_metadata(merged, source_hash, model_name)


BGM_TYPES = ("daily", "suspense", "battle", "sad", "romantic", "epic", "comedy", "horror")
BGM_TYPE_MAP_ZH = {
    "daily": "\u65e5\u5e38",
    "suspense": "\u60ac\u7591",
    "battle": "\u6218\u6597",
    "sad": "\u60b2\u4f24",
    "romantic": "\u6d6a\u6f2b",
    "epic": "\u53f2\u8bd7",
    "comedy": "\u6ed1\u7a3d",
    "horror": "\u6050\u6016",
}
BGM_TYPE_MAP_EN = {value: key for key, value in BGM_TYPE_MAP_ZH.items()}

BGM_TYPE_PROMPT = """Act as a senior film-score music director for one audiobook scene.
Classify its dominant musical function and write a production-ready English music
brief for ACE-Step. The music must support spoken Chinese dialogue rather than compete
with it. Base every choice on the supplied source, not generic genre keywords.
- daily: relaxed, warm, conversational, routine, or reflective neutral life
- suspense: uncertainty, investigation, hidden danger, scheming, or mounting tension
- battle: active combat, chase, confrontation, or forceful kinetic action
- sad: grief, loss, regret, loneliness, or emotionally heavy aftermath
- romantic: mutual affection, intimacy, tenderness, or romantic longing
- epic: awe, revelation, triumph, vast stakes, ceremony, or grand resolve
- comedy: a scene whose main function is humor, absurdity, teasing, or comic relief
- horror: dread, grotesque threat, terror, or sustained uncanny fear

Use the source excerpt over the title. Classify atmosphere, not isolated keywords.
Consider neighboring segments to maintain score continuity while preserving genuine
transitions. The English music_prompt must specify mood, period-appropriate
instrumentation, texture, dynamics, tempo feel, and a beginning-to-end dramatic arc.
Keep music_prompt concise: target 140-300 characters and never exceed 420 characters.
Keep instrumentation and key_mode as compact production labels. Keep avoid under 180
characters and reserve it for unwanted musical/audio traits rather than explanations.
It must request an instrumental, sparse dialogue underscore with no vocals, lyrics,
spoken words, sound effects, or oversized trailer impacts. Do not narrate plot events
or include character names in the music prompt. Cite concrete source lines in evidence.
Submit exactly one result with submit_bgm_type.
"""

TYPE_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_bgm_type",
        "description": "Submit one grounded BGM type classification.",
        "strict": True,
        "parameters": _schema(
            {
                "segment_index": {"type": "integer", "minimum": 1},
                "bgm_type": {"type": "string", "enum": list(BGM_TYPES)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "minLength": 1},
                "music_prompt": {"type": "string", "minLength": 80, "maxLength": 420},
                "instrumentation": {"type": "string", "minLength": 4, "maxLength": 240},
                "tempo_bpm": {"type": "integer", "minimum": 40, "maximum": 180},
                "key_mode": {"type": "string", "minLength": 2, "maxLength": 80},
                "energy": {"type": "integer", "minimum": 1, "maximum": 5},
                "narrative_arc": {
                    "type": "string",
                    "enum": ["stable", "building", "releasing", "rise_and_fall"],
                },
                "transition": {
                    "type": "string",
                    "enum": ["seamless", "gentle", "abrupt"],
                },
                "avoid": {"type": "string", "minLength": 4, "maxLength": 240},
            },
            [
                "segment_index", "bgm_type", "confidence", "evidence",
                "music_prompt", "instrumentation", "tempo_bpm", "key_mode",
                "energy", "narrative_arc", "transition", "avoid",
            ],
        ),
    },
}]


def _request_bgm_decision(
    client: LLMClient,
    messages: list[dict[str, Any]],
    segment_index: int,
    max_retries: int,
    agent_role: str,
) -> dict[str, Any]:
    last_error = "No tool call"
    last_tool_candidate: Optional[dict[str, Any]] = None
    for step in range(1, max_retries + 1):
        result = client.chat(
            messages,
            tools=TYPE_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_bgm_type"}},
            temperature=0.05,
            max_tokens=4000,
            agent_role=agent_role,
            trace_id=f"bgm:type:{segment_index}",
            agent_round=step,
        )
        for call in result.tool_calls:
            if call.name == "submit_bgm_type":
                last_tool_candidate = call.arguments
                candidate, last_error = _validate_bgm_type(call.arguments, segment_index)
                if candidate:
                    return candidate
        if result.tool_calls:
            messages.append(_assistant_tool_message(result))
            for call in result.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": f"Rejected: {last_error}"})
        else:
            messages.append({"role": "assistant", "content": result.content or ""})
        messages.append({"role": "user", "content": f"Invalid classification: {last_error}. Submit a corrected tool call."})
    if last_tool_candidate is not None:
        candidate, repair_error = _validate_bgm_type(
            last_tool_candidate,
            segment_index,
            repair_fields=True,
        )
        if candidate:
            logger.warning(
                "BGM type segment=%d role=%s exhausted %d attempts; retained the last valid tool result "
                "after deterministic field repair: %s",
                segment_index,
                agent_role,
                max_retries,
                last_error,
            )
            return candidate
        last_error = repair_error
    raise SegmentationError(f"BGM type classification failed for segment {segment_index}: {last_error}")


def _valid_bgm_checkpoint_results(raw_results: Any, total_segments: int) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_results, dict):
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for key, raw_decision in raw_results.items():
        try:
            segment_index = int(key)
        except (TypeError, ValueError):
            continue
        if not 1 <= segment_index <= total_segments or not isinstance(raw_decision, dict):
            continue
        candidate, _ = _validate_bgm_type(raw_decision, segment_index)
        if candidate:
            completed[str(segment_index)] = dict(raw_decision, **candidate)
    return completed


def label_bgm_types(
    segments: list[dict[str, Any]],
    client: Optional[LLMClient] = None,
    novel_text: str = "",
    checkpoint_path: Path | str | None = DEFAULT_TYPE_CHECKPOINT,
    resume: bool = True,
    max_retries: int = 4,
) -> list[dict[str, Any]]:
    """Classify one segment per API decision, with source excerpts and checkpoints."""
    client = client or LLMClient.for_flash_lite("bgm_classification")
    model_name = _flash_lite_model(client)
    usage_at_start = _normalise_usage_summary(client.usage_summary())
    lines = novel_text.splitlines()
    source_hash = bgm_source_hash(novel_text)
    segments_hash = _segment_inputs_hash(segments)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed: dict[str, dict[str, Any]] = {}
    previous_usage = _normalise_usage_summary({})
    if resume and checkpoint and checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            compatible = (
                payload.get("pipeline_version") == BGM_TYPE_PIPELINE_VERSION
                and payload.get("source_hash") == source_hash
                and payload.get("model") == model_name
                and payload.get("segments_hash") == segments_hash
            )
            if compatible:
                completed = _valid_bgm_checkpoint_results(payload.get("results"), len(segments))
                previous_usage = _normalise_usage_summary(payload.get("llm_usage"))
            else:
                logger.warning(
                    "Ignoring incompatible BGM type checkpoint %s "
                    "(version/source/model/segments changed)",
                    checkpoint,
                )
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid BGM type checkpoint: %s", checkpoint)

    def write_checkpoint() -> None:
        if checkpoint:
            _atomic_write_json(
                checkpoint,
                {
                    "pipeline_version": BGM_TYPE_PIPELINE_VERSION,
                    "source_hash": source_hash,
                    "model": model_name,
                    "segments_hash": segments_hash,
                    "results": completed,
                    "llm_usage": _cumulative_usage(client, previous_usage, usage_at_start),
                },
            )

    output = [dict(segment) for segment in segments]
    previous_type = "none"
    for zero_index, segment in enumerate(output):
        one_index = zero_index + 1
        key = str(one_index)
        if key in completed:
            decision = completed[key]
        else:
            start, end = int(segment["start_line"]), int(segment["end_line"])
            excerpt = _scene_excerpt(lines, start, end) if lines else "Source text was not supplied; rely on validated segment metadata."
            neighbors = {
                "previous": output[zero_index - 1] if zero_index else None,
                "current": segment,
                "next": output[zero_index + 1] if zero_index + 1 < len(output) else None,
                "previous_bgm_type": previous_type,
            }
            messages = [
                {"role": "system", "content": BGM_TYPE_PROMPT},
                {"role": "user", "content": f"Segment index: {one_index}\nMetadata: {json.dumps(neighbors, ensure_ascii=False)}\n\nSource excerpt:\n{excerpt}"},
            ]
            calls_before = _normalise_usage_summary(client.usage_summary())["calls"]
            primary = _request_bgm_decision(client, messages, one_index, max_retries, "bgm_type_primary")
            review: Optional[dict[str, Any]] = None
            review_error = ""
            try:
                review = _request_bgm_decision(
                    client,
                    [
                        {"role": "system", "content": BGM_TYPE_PROMPT + "\nAct as an independent second classifier. Challenge keyword matching and continuity bias."},
                        {"role": "user", "content": f"Segment index: {one_index}\nFirst classifier: {json.dumps(primary, ensure_ascii=False)}\nMetadata: {json.dumps(neighbors, ensure_ascii=False)}\n\nSource excerpt:\n{excerpt}"},
                    ],
                    one_index,
                    max_retries,
                    "bgm_type_reviewer",
                )
            except SegmentationError as exc:
                review_error = str(exc)
                logger.warning(
                    "BGM type reviewer failed for segment %d; retaining the validated primary decision: %s",
                    one_index,
                    review_error,
                )
            if review is None:
                decision = dict(primary)
                decision["review_fallback"] = True
                decision["review_error"] = review_error
            elif review["bgm_type"] == primary["bgm_type"]:
                decision = dict(
                    max((primary, review), key=lambda item: float(item["confidence"]))
                )
                decision["confidence"] = round(
                    (primary["confidence"] + review["confidence"]) / 2, 4
                )
                decision["evidence"] = (
                    f"Primary: {primary['evidence']} | Review: {review['evidence']}"
                )
            else:
                try:
                    decision = _request_bgm_decision(
                        client,
                        [
                            {
                                "role": "system",
                                "content": BGM_TYPE_PROMPT
                                + "\nAct as the final scoring director. Resolve both drafts and "
                                "deliver the most source-specific production brief.",
                            },
                            {
                                "role": "user",
                                "content": f"Segment index: {one_index}\nDrafts: "
                                f"{json.dumps({'primary': primary, 'review': review}, ensure_ascii=False)}"
                                f"\nMetadata: {json.dumps(neighbors, ensure_ascii=False)}"
                                f"\n\nSource excerpt:\n{excerpt}",
                            },
                        ],
                        one_index,
                        max_retries,
                        "bgm_type_adjudicator",
                    )
                    decision["adjudicated"] = True
                except SegmentationError as exc:
                    decision = dict(max(
                        (primary, review), key=lambda item: float(item["confidence"])
                    ))
                    decision["adjudication_fallback"] = True
                    decision["adjudication_error"] = str(exc)
                    logger.warning(
                        "BGM type adjudicator failed for segment %d; retaining the higher-confidence "
                        "validated classifier decision: %s",
                        one_index,
                        exc,
                    )
            decision["review"] = review
            decision["primary_decision"] = primary
            decision["review_decision"] = review
            decision["agent_calls"] = max(
                0, _normalise_usage_summary(client.usage_summary())["calls"] - calls_before
            )
            completed[key] = decision
            write_checkpoint()
        segment.update({
            "bgm_type": decision["bgm_type"],
            "bgm_type_zh": BGM_TYPE_MAP_ZH[decision["bgm_type"]],
            "bgm_confidence": decision["confidence"],
            "bgm_evidence": decision["evidence"],
            "bgm_music_prompt": decision["music_prompt"],
            "bgm_instrumentation": decision["instrumentation"],
            "bgm_tempo_bpm": decision["tempo_bpm"],
            "bgm_key_mode": decision["key_mode"],
            "bgm_energy": decision["energy"],
            "bgm_narrative_arc": decision["narrative_arc"],
            "bgm_transition": decision["transition"],
            "bgm_avoid": decision["avoid"],
            "bgm_agent_calls": decision.get("agent_calls", 1),
            "bgm_pipeline_version": BGM_TYPE_PIPELINE_VERSION,
            "bgm_source_hash": source_hash,
            "bgm_model": model_name,
        })
        previous_type = decision["bgm_type"]
        logger.info("BGM type progress %d/%d: %s", one_index, len(output), previous_type)
    write_checkpoint()
    return output


def _scene_excerpt(lines: list[str], start: int, end: int, budget_lines: int = 90) -> str:
    if not lines:
        return ""
    start, end = max(1, start), min(len(lines), end)
    count = end - start + 1
    if count <= budget_lines:
        selected = list(range(start, end + 1))
    else:
        third = budget_lines // 3
        middle = (start + end) // 2
        selected = (
            list(range(start, start + third))
            + list(range(max(start + third, middle - third // 2), min(end - third + 1, middle + third // 2 + 1)))
            + list(range(end - third + 1, end + 1))
        )
        selected = sorted(set(selected))
    return "\n".join(f"{line}: {lines[line - 1]}" for line in selected)


MUSIC_PROMPT_SAFETY_SUFFIX = (
    "Keep it instrumental and sparse beneath dialogue, with restrained dynamics, clear midrange space, "
    "a coherent beginning-to-end arc, and no vocals, lyrics, spoken words, sound effects, or oversized trailer impacts."
)
AVOID_SAFETY_SUFFIX = "vocals, lyrics, spoken words, sound effects, and oversized trailer impacts"


def _truncate_text_at_word(value: str, max_chars: int) -> str:
    shortened = value[:max_chars].rstrip()
    if len(value) > max_chars and " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" ,;:.")


def _fit_music_prompt_length(music_prompt: str, fallback: str) -> str:
    prompt = re.sub(r"\s+", " ", music_prompt).strip() or fallback
    if len(prompt) < 80:
        prompt = f"{prompt.rstrip(' ,;:.')}. {MUSIC_PROMPT_SAFETY_SUFFIX}"
    if len(prompt) > 420:
        head_budget = 420 - len(MUSIC_PROMPT_SAFETY_SUFFIX) - 2
        head = _truncate_text_at_word(prompt, head_budget)
        prompt = f"{head}. {MUSIC_PROMPT_SAFETY_SUFFIX}"
    return prompt[:420].rstrip()


def _fit_bounded_text(value: str, fallback: str, min_chars: int, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) < min_chars:
        text = fallback
    if len(text) > max_chars:
        text = _truncate_text_at_word(text, max_chars)
    return text


def _fit_avoid_length(value: str, fallback: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) < 4:
        return fallback
    if len(text) > 240:
        head_budget = 240 - len(AVOID_SAFETY_SUFFIX) - 2
        head = _truncate_text_at_word(text, head_budget)
        text = f"{head}; {AVOID_SAFETY_SUFFIX}"
    return text[:240].rstrip()


def _coerce_bgm_number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else default


def _validate_bgm_type(
    raw: dict[str, Any],
    expected_index: int,
    *,
    repair_fields: bool = False,
) -> tuple[Optional[dict[str, Any]], str]:
    bgm_type = str(raw.get("bgm_type", ""))
    defaults = {
        "music_prompt": (
            f"Subtle {bgm_type or 'cinematic'} instrumental underscore with restrained "
            "dynamics, sparse acoustic texture, a coherent scene-length arc, and ample "
            "midrange space beneath spoken narration; no vocals, lyrics, or sound effects."
        ),
        "instrumentation": "soft piano, chamber strings, and restrained ambient texture",
        "tempo_bpm": 76,
        "key_mode": "minor or modal",
        "energy": 2,
        "narrative_arc": "stable",
        "transition": "gentle",
        "avoid": "vocals, lyrics, spoken words, sound effects, and oversized impacts",
    }
    numeric_fields_repaired: list[str] = []
    if repair_fields:
        index = expected_index
        confidence = _coerce_bgm_number(raw.get("confidence"), 0.5)
        if "%" in str(raw.get("confidence", "")) or 1 < confidence <= 100:
            confidence /= 100
        confidence = min(1.0, max(0.0, confidence))
        tempo_bpm = min(180, max(40, round(_coerce_bgm_number(
            raw.get("tempo_bpm", defaults["tempo_bpm"]), defaults["tempo_bpm"]
        ))))
        energy = min(5, max(1, round(_coerce_bgm_number(
            raw.get("energy", defaults["energy"]), defaults["energy"]
        ))))
        numeric_values = {
            "segment_index": index,
            "confidence": confidence,
            "tempo_bpm": tempo_bpm,
            "energy": energy,
        }
        for field, value in numeric_values.items():
            try:
                if float(raw.get(field)) != float(value):
                    numeric_fields_repaired.append(field)
            except (TypeError, ValueError):
                numeric_fields_repaired.append(field)
    else:
        try:
            index = int(raw.get("segment_index"))
        except (TypeError, ValueError):
            return None, f"segment_index must be numeric (got {raw.get('segment_index')!r})"
        try:
            confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            return None, f"confidence must be numeric (got {raw.get('confidence')!r})"
        try:
            tempo_bpm = int(raw.get("tempo_bpm", defaults["tempo_bpm"]))
        except (TypeError, ValueError):
            return None, f"tempo_bpm must be numeric (got {raw.get('tempo_bpm')!r})"
        try:
            energy = int(raw.get("energy", defaults["energy"]))
        except (TypeError, ValueError):
            return None, f"energy must be numeric (got {raw.get('energy')!r})"
    evidence = str(raw.get("evidence", "")).strip()
    music_prompt = re.sub(r"\s+", " ", str(raw.get("music_prompt", defaults["music_prompt"]))).strip()
    instrumentation = re.sub(r"\s+", " ", str(raw.get("instrumentation", defaults["instrumentation"]))).strip()
    key_mode = str(raw.get("key_mode", defaults["key_mode"])).strip()
    narrative_arc = str(raw.get("narrative_arc", defaults["narrative_arc"]))
    transition = str(raw.get("transition", defaults["transition"]))
    avoid = re.sub(r"\s+", " ", str(raw.get("avoid", defaults["avoid"]))).strip()
    repaired_fields: list[str] = []
    if repair_fields:
        repaired_values = {
            "music_prompt": _fit_music_prompt_length(music_prompt, defaults["music_prompt"]),
            "instrumentation": _fit_bounded_text(
                instrumentation, defaults["instrumentation"], 4, 240
            ),
            "key_mode": _fit_bounded_text(key_mode, defaults["key_mode"], 2, 80),
            "avoid": _fit_avoid_length(avoid, defaults["avoid"]),
        }
        original_values = {
            "music_prompt": music_prompt,
            "instrumentation": instrumentation,
            "key_mode": key_mode,
            "avoid": avoid,
        }
        repaired_fields = [
            field for field, value in repaired_values.items() if value != original_values[field]
        ]
        music_prompt = repaired_values["music_prompt"]
        instrumentation = repaired_values["instrumentation"]
        key_mode = repaired_values["key_mode"]
        avoid = repaired_values["avoid"]
    if index != expected_index:
        return None, f"segment_index must be {expected_index}"
    if bgm_type not in BGM_TYPES:
        return None, f"bgm_type must be one of {BGM_TYPES}"
    if not 0 <= confidence <= 1:
        return None, "confidence must be between 0 and 1"
    if len(evidence) < 4:
        return None, "evidence is too short"
    if not 80 <= len(music_prompt) <= 420:
        return None, f"music_prompt must contain 80..420 characters (got {len(music_prompt)})"
    if not 4 <= len(instrumentation) <= 240:
        return None, f"instrumentation must contain 4..240 characters (got {len(instrumentation)})"
    if not 40 <= tempo_bpm <= 180:
        return None, "tempo_bpm must be between 40 and 180"
    if not 1 <= energy <= 5:
        return None, "energy must be between 1 and 5"
    if not 2 <= len(key_mode) <= 80:
        return None, f"key_mode must contain 2..80 characters (got {len(key_mode)})"
    if narrative_arc not in {"stable", "building", "releasing", "rise_and_fall"}:
        return None, "narrative_arc is invalid"
    if transition not in {"seamless", "gentle", "abrupt"}:
        return None, "transition is invalid"
    if not 4 <= len(avoid) <= 240:
        return None, f"avoid must contain 4..240 characters (got {len(avoid)})"
    return {
        "segment_index": index,
        "bgm_type": bgm_type,
        "confidence": confidence,
        "evidence": evidence,
        "music_prompt": music_prompt,
        "instrumentation": instrumentation,
        "tempo_bpm": tempo_bpm,
        "key_mode": key_mode,
        "energy": energy,
        "narrative_arc": narrative_arc,
        "transition": transition,
        "avoid": avoid,
        "music_prompt_repaired": "music_prompt" in repaired_fields,
        "text_fields_repaired": repaired_fields,
        "numeric_fields_repaired": numeric_fields_repaired,
    }, ""


def _extract_bgm_types(content: str, expected: int) -> Optional[list[dict[str, Any]]]:
    """Compatibility parser retained for older cached/batched responses."""
    values = _extract_json_array(content)
    if not values or len(values) != expected:
        return None
    checked = []
    for position, value in enumerate(values, 1):
        if not isinstance(value, dict):
            return None
        bgm_type = value.get("bgm_type")
        if bgm_type not in BGM_TYPES or int(value.get("segment_index", -1)) != position:
            return None
        checked.append({"segment_index": position, "bgm_type": bgm_type})
    return checked


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (2**attempt))
