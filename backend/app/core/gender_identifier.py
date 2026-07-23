"""Tool-calling agents that identify character gender for voice casting."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from app.core.llm_client import LLMClient, LLMResult, SENSENOVA_FLASH_LITE_MODEL, ToolCall

logger = logging.getLogger("gender_identifier")

GENDERS = ("male", "female", "unknown")
DEFAULT_CHECKPOINT = Path("backend/data/gender_results.checkpoint.json")
GENDER_PIPELINE_VERSION = 2


def gender_source_hash(text: str, character_names: list[str], dialogues: Optional[list[dict]] = None) -> str:
    identity = {
        "characters": list(character_names),
        "dialogues": [
            {
                "line": int(dialogue.get("line", 0) or 0),
                "speaker": str(dialogue.get("speaker", "")),
                "text": str(dialogue.get("text", "")),
            }
            for dialogue in (dialogues or [])
        ],
    }
    digest = hashlib.sha256(text.encode("utf-8"))
    digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


class GenderBatchError(RuntimeError):
    """A character failed repeatedly; completed characters remain checkpointed."""


class NovelIndex:
    """One reusable novel index with cached evidence packets."""

    def __init__(self, text: str, dialogues: Optional[list[dict]] = None):
        self.text = text
        self.lines = text.splitlines()
        self.dialogues = dialogues or []
        self._search_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._evidence_cache: dict[str, str] = {}

    def search(self, keyword: str, limit: int = 20) -> dict[str, Any]:
        key = (keyword, limit)
        if key in self._search_cache:
            return self._search_cache[key]
        all_matches = [
            {"line_number": number, "line": line.strip()[:240]}
            for number, line in enumerate(self.lines, 1)
            if keyword and keyword in line
        ]
        result = {
            "total_matches": len(all_matches),
            "truncated": len(all_matches) > limit,
            "matches": all_matches[:limit],
        }
        self._search_cache[key] = result
        return result

    def read_lines(self, start: int, end: int, limit: int = 240) -> dict[str, Any]:
        start = max(1, int(start))
        end = min(len(self.lines), int(end))
        if start > end:
            return {"text": "", "truncated": False}
        selected = self.lines[start - 1 : end]
        truncated = len(selected) > limit
        selected = selected[:limit]
        return {
            "text": "\n".join(f"{start + offset}: {line.strip()}" for offset, line in enumerate(selected)),
            "truncated": truncated,
        }

    def get_dialogues(self, character_name: str, limit: int = 50) -> list[dict]:
        matched = []
        for index, dialogue in enumerate(self.dialogues):
            if dialogue.get("speaker") == character_name:
                matched.append(
                    {
                        "dialogue_index": index,
                        "line_number": int(dialogue.get("line", 0)),
                        "text": str(dialogue.get("text", ""))[:240],
                    }
                )
                if len(matched) >= limit:
                    break
        if matched:
            return matched
        for number, line in enumerate(self.lines, 1):
            if character_name in line and ("\u300c" in line or "\u300d" in line):
                matched.append({"dialogue_index": -1, "line_number": number, "text": line.strip()[:240]})
                if len(matched) >= limit:
                    break
        return matched

    def evidence_packet(self, character_name: str, max_occurrences: int = 12, radius: int = 5) -> str:
        if character_name in self._evidence_cache:
            return self._evidence_cache[character_name]
        matches = self.search(character_name, limit=80)["matches"]
        if not matches:
            packet = "No literal name occurrence was found. Use dialogue metadata and return unknown if evidence remains absent."
            self._evidence_cache[character_name] = packet
            return packet

        positions = [item["line_number"] for item in matches]
        if len(positions) > max_occurrences:
            last = len(positions) - 1
            chosen = sorted({positions[round(i * last / (max_occurrences - 1))] for i in range(max_occurrences)})
        else:
            chosen = positions
        blocks = []
        for position in chosen:
            block = self.read_lines(position - radius, position + radius, limit=radius * 2 + 1)["text"]
            blocks.append(block)
        dialogues = self.get_dialogues(character_name, limit=12)
        dialogue_text = "\n".join(
            f"line {item['line_number']}: {item['text']}" for item in dialogues
        ) or "No speaker-labelled dialogue found."
        packet = (
            f"Representative name contexts ({len(chosen)} of {len(positions)} occurrences):\n"
            + "\n---\n".join(blocks)
            + "\n\nSpeaker-labelled dialogue samples:\n"
            + dialogue_text
        )
        self._evidence_cache[character_name] = packet
        return packet


def _object_schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_novel",
            "description": "Search exact text in the complete novel and return source line numbers.",
            "strict": True,
            "parameters": _object_schema(
                {
                    "keyword": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                ["keyword", "limit"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": "Read a bounded, 1-based inclusive range from the complete novel.",
            "strict": True,
            "parameters": _object_schema(
                {
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                ["start_line", "end_line"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dialogues",
            "description": "Return dialogue metadata already assigned to this speaker.",
            "strict": True,
            "parameters": _object_schema(
                {
                    "character_name": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                ["character_name", "limit"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_gender",
            "description": "Submit the final evidence-grounded voice-casting gender decision.",
            "strict": True,
            "parameters": _object_schema(
                {
                    "character_name": {"type": "string"},
                    "gender": {"type": "string", "enum": list(GENDERS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "minLength": 1},
                },
                ["character_name", "gender", "confidence", "evidence"],
            ),
        },
    },
]

REVIEW_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_gender_review",
        "description": "Submit an independent review.",
        "strict": True,
        "parameters": _object_schema(
            {
                "gender": {"type": "string", "enum": list(GENDERS)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "minLength": 1},
                "reasoning_summary": {"type": "string", "minLength": 1},
            },
            ["gender", "confidence", "evidence", "reasoning_summary"],
        ),
    },
}]

SYSTEM_PROMPT = """You are the evidence analyst for character voice casting.
Determine gender only from this novel. Names alone are weak evidence and must never
override explicit narration. Strong evidence includes explicit pronouns, kinship or
titles, self-identification, and unambiguous physical descriptions. Dialogue style,
personality, occupation, and stereotypes are not gender evidence.

Inspect the supplied representative contexts first. Use search/read tools when the
evidence is ambiguous, contradictory, or may have been introduced elsewhere. Cite
specific source line numbers in evidence. Return unknown when the text does not
support male or female. Never convert unknown to a guessed default. Finish only by
calling submit_gender for the requested character.
"""


def _assistant_tool_message(result: LLMResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in result.tool_calls
        ],
    }


def _execute_tool(call: ToolCall, index: NovelIndex) -> str:
    if call.name == "search_novel":
        return json.dumps(index.search(str(call.arguments.get("keyword", "")), int(call.arguments.get("limit", 20))), ensure_ascii=False)
    if call.name == "read_lines":
        return json.dumps(index.read_lines(int(call.arguments.get("start_line", 1)), int(call.arguments.get("end_line", 1))), ensure_ascii=False)
    if call.name == "get_dialogues":
        return json.dumps(index.get_dialogues(str(call.arguments.get("character_name", "")), int(call.arguments.get("limit", 20))), ensure_ascii=False)
    return "This tool is not executable; validate and submit it as the final answer."


def _validate_gender(raw: dict[str, Any], character_name: str) -> tuple[Optional[dict[str, Any]], str]:
    gender = str(raw.get("gender", "")).lower()
    if gender not in GENDERS:
        return None, f"gender must be one of {GENDERS}"
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        return None, "confidence must be numeric"
    if not 0 <= confidence <= 1:
        return None, "confidence must be between 0 and 1"
    evidence = str(raw.get("evidence", "")).strip()
    if len(evidence) < 4:
        return None, "evidence is empty or too short"
    return {
        "character_name": character_name,
        "gender": gender,
        "confidence": confidence,
        "evidence": evidence,
    }, ""


def _primary_analysis(
    character_name: str,
    index: NovelIndex,
    client: LLMClient,
    max_tool_steps: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Target character: {character_name}\nNovel length: {len(index.lines)} lines.\n\n"
                f"Cached evidence packet:\n{index.evidence_packet(character_name)}"
            ),
        },
    ]
    last_error = "No final submission"
    for step in range(1, max_tool_steps + 1):
        result = client.chat(
            messages,
            tools=TOOL_SPECS,
            temperature=0.1,
            max_tokens=1800,
            agent_role="gender_primary",
            trace_id=f"gender:{character_name}",
            agent_round=step,
        )
        if not result.tool_calls:
            messages.append({"role": "assistant", "content": result.content or ""})
            messages.append({"role": "user", "content": "Use the available tools and finish with submit_gender."})
            continue
        messages.append(_assistant_tool_message(result))
        for call in result.tool_calls:
            if call.name == "submit_gender":
                validated, last_error = _validate_gender(call.arguments, character_name)
                tool_output = "Accepted" if validated else f"Rejected: {last_error}. Correct and resubmit."
                messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_output})
                if validated:
                    return validated
            else:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _execute_tool(call, index)})
    return {
        "character_name": character_name,
        "gender": "unknown",
        "confidence": 0.0,
        "evidence": f"Primary agent did not produce a valid result: {last_error}",
    }


def _review(
    character_name: str,
    index: NovelIndex,
    primary: dict[str, Any],
    client: LLMClient,
    role: str,
) -> dict[str, Any]:
    system = (
        "You are an independent reviewer. Judge only explicit textual evidence. "
        "Actively look for unsupported name-based inference and contradictions. "
        "Return unknown when evidence is insufficient. Use submit_gender_review."
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Review role: {role}\nCharacter: {character_name}\n"
                f"Candidate decision: {json.dumps(primary, ensure_ascii=False)}\n\n"
                f"Source packet:\n{index.evidence_packet(character_name)}"
            ),
        },
    ]
    last_error = "missing tool call"
    for attempt in range(3):
        result = client.chat(
            messages,
            tools=REVIEW_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_gender_review"}},
            temperature=0.0,
            max_tokens=1800,
            agent_role="gender_" + role.replace(" ", "_"),
            trace_id=f"gender:{character_name}",
            agent_round=attempt + 1,
        )
        for call in result.tool_calls:
            if call.name == "submit_gender_review":
                validated, last_error = _validate_gender(call.arguments, character_name)
                if validated:
                    validated["reasoning_summary"] = str(call.arguments.get("reasoning_summary", "")).strip()
                    return validated
        if result.tool_calls:
            messages.append(_assistant_tool_message(result))
            for call in result.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": f"Rejected: {last_error}"})
        else:
            messages.append({"role": "assistant", "content": result.content})
        messages.append({"role": "user", "content": f"Invalid review ({last_error}). Submit a corrected tool call."})
    return {
        "character_name": character_name,
        "gender": "unknown",
        "confidence": 0.0,
        "evidence": f"Review failed validation: {last_error}",
        "reasoning_summary": "review unavailable",
    }


def identify_gender(
    character_name: str,
    text: str,
    client: Optional[LLMClient] = None,
    max_tool_steps: int = 8,
    dialogues: Optional[list[dict]] = None,
    verification_threshold: float = 0.8,
    always_verify: bool = True,
    _index: Optional[NovelIndex] = None,
) -> dict[str, Any]:
    """Identify one character, selectively using review and adjudication agents."""
    client = client or LLMClient.for_flash_lite("gender")
    index = _index or NovelIndex(text, dialogues)
    calls_before = client.usage_summary()["calls"]
    primary = _primary_analysis(character_name, index, client, max_tool_steps)
    needs_review = always_verify or primary["gender"] == "unknown" or primary["confidence"] < verification_threshold
    verification: dict[str, Any] = {"reviewed": False, "agreement": None}

    final = primary
    if needs_review:
        reviewer = _review(character_name, index, primary, client, "independent verifier")
        verification = {
            "reviewed": True,
            "primary": primary,
            "review": reviewer,
            "agreement": reviewer["gender"] == primary["gender"],
        }
        if reviewer["gender"] == primary["gender"]:
            final = dict(primary)
            final["confidence"] = round((primary["confidence"] + reviewer["confidence"]) / 2, 4)
            final["evidence"] = f"Primary: {primary['evidence']} | Review: {reviewer['evidence']}"
        else:
            adjudication_input = dict(primary)
            adjudication_input["reviewer_decision"] = reviewer
            adjudicator = _review(character_name, index, adjudication_input, client, "final adjudicator resolving disagreement")
            verification["adjudication"] = adjudicator
            final = {key: adjudicator[key] for key in ("character_name", "gender", "confidence", "evidence")}

    final["verification"] = verification
    final["agent_calls"] = client.usage_summary()["calls"] - calls_before
    return final


def identify_all_genders(
    character_names: list[str],
    text: str,
    client: Optional[LLMClient] = None,
    max_tool_steps: int = 8,
    dialogues: Optional[list[dict]] = None,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT,
    resume: bool = True,
    item_retries: int = 3,
) -> list[dict[str, Any]]:
    """Identify all characters with per-character atomic checkpointing."""
    client = client or LLMClient.for_flash_lite("gender")
    usage_at_start = _normalise_usage_summary(client.usage_summary())
    index = NovelIndex(text, dialogues)
    source_hash = gender_source_hash(text, character_names, dialogues)
    model_name = getattr(client, "sensenova_model", SENSENOVA_FLASH_LITE_MODEL)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    previous_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resume and checkpoint and checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            compatible = (
                payload.get("pipeline_version") == GENDER_PIPELINE_VERSION
                and payload.get("source_hash") == source_hash
                and payload.get("model") == model_name
            )
            if compatible:
                completed = {
                    item["character_name"]: item
                    for item in payload.get("results", [])
                    if item.get("character_name")
                    and not str(item.get("evidence", "")).startswith("Error:")
                }
                errors = dict(payload.get("errors", {}))
                previous_usage = _normalise_usage_summary(payload.get("llm_usage", {}))
            else:
                logger.warning(
                    "Ignoring incompatible gender checkpoint %s (version/source/model changed)", checkpoint
                )
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid gender checkpoint: %s", checkpoint)

    for position, name in enumerate(character_names, 1):
        if name in completed:
            continue
        last_error: Optional[Exception] = None
        item_calls_before = client.usage_summary()["calls"]
        for attempt in range(1, item_retries + 1):
            try:
                value = identify_gender(
                    name,
                    text,
                    client=client,
                    max_tool_steps=max_tool_steps,
                    dialogues=dialogues,
                    _index=index,
                )
                value["agent_calls"] = client.usage_summary()["calls"] - item_calls_before
                value["item_attempts"] = attempt
                completed[name] = value
                errors.pop(name, None)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Gender %s attempt %d/%d failed: %s", name, attempt, item_retries, exc)
        if name not in completed:
            errors[name] = str(last_error)
            if checkpoint:
                _atomic_write_json(
                    checkpoint,
                    _gender_checkpoint_payload(
                        character_names,
                        completed,
                        errors,
                        client,
                        source_hash,
                        model_name,
                        previous_usage,
                        usage_at_start,
                    ),
                )
            raise GenderBatchError(
                f"Character {name} failed after {item_retries} attempts; rerun to resume from {checkpoint}: {last_error}"
            )
        logger.info("Gender progress %d/%d: %s -> %s", position, len(character_names), name, completed[name]["gender"])
        if checkpoint:
            _atomic_write_json(
                checkpoint,
                _gender_checkpoint_payload(
                    character_names,
                    completed,
                    errors,
                    client,
                    source_hash,
                    model_name,
                    previous_usage,
                    usage_at_start,
                ),
            )
    return [completed[name] for name in character_names if name in completed]


def _normalise_usage_summary(raw: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(raw.get(key, 0) or 0)
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
    }


def _gender_checkpoint_payload(
    character_names: list[str],
    completed: dict[str, dict[str, Any]],
    errors: dict[str, str],
    client: LLMClient,
    source_hash: str,
    model_name: str,
    previous_usage: dict[str, int],
    usage_at_start: dict[str, int],
) -> dict[str, Any]:
    current_usage = _normalise_usage_summary(client.usage_summary())
    cumulative_usage = {
        key: previous_usage.get(key, 0) + max(0, current_usage[key] - usage_at_start[key])
        for key in current_usage
    }
    return {
        "pipeline_version": GENDER_PIPELINE_VERSION,
        "source_hash": source_hash,
        "model": model_name,
        "results": [completed[name] for name in character_names if name in completed],
        "errors": errors,
        "llm_usage": cumulative_usage,
    }


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
