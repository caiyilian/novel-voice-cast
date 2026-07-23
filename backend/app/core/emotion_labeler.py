"""Evidence-grounded emotion and delivery-tone labeling agents."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from app.core.llm_client import LLMClient, LLMResult, SENSENOVA_FLASH_LITE_MODEL, ToolCall

logger = logging.getLogger("emotion_labeler")

EMOTIONS = ["happy", "sad", "angry", "surprised", "calm", "nervous", "cold"]
TONES = ["loud", "soft", "stutter", "sarcastic", "gentle", "serious", "whisper"]
DEFAULT_CHECKPOINT = Path("backend/data/emotion_results.checkpoint.json")
EMOTION_PIPELINE_VERSION = 2


class EmotionBatchError(RuntimeError):
    """A dialogue remained invalid after retries; the checkpoint is resumable."""


class NovelIndex:
    def __init__(self, text: str, dialogues: Optional[list[dict]] = None):
        self.lines = text.splitlines()
        self.dialogues = dialogues or []
        self.line_to_speakers: dict[int, list[str]] = {}
        for dialogue in self.dialogues:
            line = int(dialogue.get("line", 0) or 0)
            speaker = str(dialogue.get("speaker", "")).strip()
            if line > 0 and speaker:
                self.line_to_speakers.setdefault(line, []).append(speaker)

    def _format_line(self, line_number: int, line: str) -> str:
        speakers = self.line_to_speakers.get(line_number, [])
        label = f" [speaker: {', '.join(speakers)}]" if speakers else ""
        return f"{line_number}{label}: {line.strip()}"

    def read_lines(self, start: int, end: int, limit: int = 160) -> dict[str, Any]:
        start = max(1, int(start))
        end = min(len(self.lines), int(end))
        if start > end:
            return {"text": "", "truncated": False}
        selected = self.lines[start - 1 : end]
        truncated = len(selected) > limit
        selected = selected[:limit]
        return {
            "text": "\n".join(self._format_line(start + offset, line) for offset, line in enumerate(selected)),
            "truncated": truncated,
        }

    def search(self, keyword: str, limit: int = 20) -> dict[str, Any]:
        matches = [
            {"line_number": number, "line": self._format_line(number, line)[:300]}
            for number, line in enumerate(self.lines, 1)
            if keyword and keyword in line
        ]
        return {"total_matches": len(matches), "truncated": len(matches) > limit, "matches": matches[:limit]}

    def context(self, dialogue_line: int, radius: int = 60) -> str:
        text = self.read_lines(dialogue_line - radius, dialogue_line + radius, limit=radius * 2 + 1)["text"]
        target_prefixes = (f"{dialogue_line}:", f"{dialogue_line} ")
        return "\n".join(
            f">>> TARGET SOURCE LINE {line}" if line.startswith(target_prefixes) else line
            for line in text.splitlines()
        )


def emotion_source_hash(text: str, dialogues: list[dict]) -> str:
    identity = [
        {
            "line": int(dialogue.get("line", 0) or 0),
            "speaker": str(dialogue.get("speaker", "")),
            "text": str(dialogue.get("text", "")),
        }
        for dialogue in dialogues
    ]
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_lines",
            "description": "Read additional nearby source lines when the supplied context is insufficient.",
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
            "name": "search_novel",
            "description": "Search exact words, names, or recurring expressions in the complete novel.",
            "strict": True,
            "parameters": _schema(
                {"keyword": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}},
                ["keyword", "limit"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_emotion",
            "description": "Submit exactly one emotion and one delivery tone for the target dialogue.",
            "strict": True,
            "parameters": _schema(
                {
                    "emotion": {"type": "string", "enum": EMOTIONS},
                    "tone": {"type": "string", "enum": TONES},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "minLength": 1},
                    "evidence_lines": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                    },
                },
                ["emotion", "tone", "confidence", "evidence", "evidence_lines"],
            ),
        },
    },
]

REVIEW_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_emotion_review",
        "description": "Submit an independent emotion/tone review.",
        "strict": True,
        "parameters": _schema(
            {
                "emotion": {"type": "string", "enum": EMOTIONS},
                "tone": {"type": "string", "enum": TONES},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "minLength": 1},
                "evidence_lines": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 1,
                },
            },
            ["emotion", "tone", "confidence", "evidence", "evidence_lines"],
        ),
    },
}]

SYSTEM_PROMPT = """You label one audiobook dialogue at a time from source evidence.

Emotion is the speaker's internal state:
- happy: joy, delight, amusement, or pleased excitement
- sad: grief, hurt, disappointment, or despair
- angry: anger, irritation, hostility, or indignation
- surprised: shock, sudden disbelief, or astonishment
- calm: emotionally stable, neutral, relaxed, or matter-of-fact
- nervous: fear, anxiety, unease, panic, or insecure anticipation
- cold: deliberate emotional detachment, contemptuous distance, or mercilessness

Tone is how the line is delivered:
- loud: shouted, forceful, or explicitly raised volume
- soft: quiet/low-volume delivery without being a whisper
- stutter: broken, hesitant, repeated, or stammering delivery
- sarcastic: ironic, mocking, or saying the opposite of the literal meaning
- gentle: warm, kind, soothing, or tender delivery
- serious: solemn, firm, direct, or businesslike delivery
- whisper: explicitly whispered or intentionally hushed speech

Politeness or curiosity alone is not gentle. Use gentle only with concrete warmth,
care, reassurance, or tenderness. Use serious for an otherwise unmarked direct or
businesslike line. Use soft only when low volume or subdued delivery is supported.
Do not equate punctuation with emotion. Infer who speaks, what happened immediately
before, the addressee, subtext, and narration describing delivery. Choose exactly one
label from each taxonomy. The source line number is the only location identifier: never
reinterpret it as an array index or look up a different line. Cite concrete cues and put
every cited source line in evidence_lines; evidence_lines must include the target source
line. Use calm/serious only when genuinely supported, not as a parsing fallback. Finish
with one required submission tool call for the target dialogue shown verbatim.
"""


def _assistant_message(result: LLMResult) -> dict[str, Any]:
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


def _validate(
    raw: dict[str, Any],
    dialogue_index: int,
    dialogue_line: int,
    index: NovelIndex,
) -> tuple[Optional[dict[str, Any]], str]:
    if raw.get("emotion") not in EMOTIONS:
        return None, f"emotion must be one of {EMOTIONS}"
    if raw.get("tone") not in TONES:
        return None, f"tone must be one of {TONES}"
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        return None, "confidence must be numeric"
    if not 0 <= confidence <= 1:
        return None, "confidence must be between 0 and 1"
    evidence = str(raw.get("evidence", "")).strip()
    if len(evidence) < 4:
        return None, "evidence is empty or too short"
    raw_lines = raw.get("evidence_lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        return None, "evidence_lines must be a non-empty list"
    try:
        evidence_lines = sorted({int(line) for line in raw_lines})
    except (TypeError, ValueError):
        return None, "evidence_lines must contain integers"
    if dialogue_line not in evidence_lines:
        return None, f"evidence_lines must include target source line {dialogue_line}"
    if any(line < 1 or line > len(index.lines) for line in evidence_lines):
        return None, f"evidence_lines must stay within source lines 1..{len(index.lines)}"
    return {
        "dialogue_index": dialogue_index,
        "source_line": dialogue_line,
        "emotion": raw["emotion"],
        "tone": raw["tone"],
        "confidence": confidence,
        "evidence": evidence,
        "evidence_lines": evidence_lines,
    }, ""


def _execute(call: ToolCall, index: NovelIndex) -> str:
    if call.name == "read_lines":
        payload = index.read_lines(int(call.arguments.get("start_line", 1)), int(call.arguments.get("end_line", 1)))
        return json.dumps(payload, ensure_ascii=False)
    if call.name == "search_novel":
        payload = index.search(str(call.arguments.get("keyword", "")), int(call.arguments.get("limit", 20)))
        return json.dumps(payload, ensure_ascii=False)
    return "Final submission tool; it must pass validation."


def _run_primary(
    dialogue_text: str,
    dialogue_line: int,
    dialogue_index: int,
    speaker: str,
    index: NovelIndex,
    client: LLMClient,
    max_tool_steps: int,
    memory: str,
) -> dict[str, Any]:
    del memory  # Nearby source context is the continuity source; prior labels can anchor the model.
    trace_id = f"emotion:{dialogue_index}:line:{dialogue_line}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Target source line: {dialogue_line}\n"
                f"Target speaker: {speaker or 'unknown'}\n"
                f"Target dialogue verbatim:\n<<<{dialogue_text}>>>\n\n"
                f"Nearby source context (the target is explicitly marked):\n{index.context(dialogue_line)}"
            ),
        },
    ]
    last_error = "No valid submission"
    for step in range(1, max_tool_steps + 1):
        result = client.chat(
            messages,
            tools=TOOL_SPECS,
            temperature=0.1,
            max_tokens=5000,
            agent_role="emotion_primary",
            trace_id=trace_id,
            agent_round=step,
        )
        if not result.tool_calls:
            messages.extend([
                {"role": "assistant", "content": result.content or ""},
                {"role": "user", "content": "Finish with a submit_emotion tool call."},
            ])
            continue
        messages.append(_assistant_message(result))
        for call in result.tool_calls:
            if call.name == "submit_emotion":
                validated, last_error = _validate(call.arguments, dialogue_index, dialogue_line, index)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "Accepted" if validated else f"Rejected: {last_error}. Correct and resubmit.",
                })
                if validated:
                    return validated
            else:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _execute(call, index)})
    raise ValueError(last_error)


def _run_review(
    candidate: Optional[dict[str, Any]],
    dialogue_text: str,
    dialogue_line: int,
    dialogue_index: int,
    speaker: str,
    index: NovelIndex,
    client: LLMClient,
    review_role: str = "independent verifier",
) -> dict[str, Any]:
    is_adjudicator = candidate is not None
    role_instruction = (
        "You are the final adjudicator. Compare both candidate decisions with the source, "
        "resolve the disagreement, and freely choose a third pair if both are wrong."
        if is_adjudicator
        else "You are an independent verifier. Make a fresh decision from the source only; "
        "the primary decision is intentionally hidden to prevent anchoring."
    )
    candidate_text = ""
    if candidate:
        decision_fields = ("emotion", "tone", "confidence", "evidence", "evidence_lines", "source_line")
        sanitized = {
            name: {field: decision.get(field) for field in decision_fields if field in decision}
            for name, decision in candidate.items()
        }
        candidate_text = f"\n\nCandidate decisions (source line is authoritative):\n{json.dumps(sanitized, ensure_ascii=False)}"
    trace_id = f"emotion:{dialogue_index}:line:{dialogue_line}"
    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\n"
                + role_instruction
                + " Challenge generic defaults, unsupported intensity, and labels based only on punctuation. "
                  "Submit exactly one submit_emotion_review tool call."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target source line: {dialogue_line}\n"
                f"Target speaker: {speaker or 'unknown'}\n"
                f"Target dialogue verbatim:\n<<<{dialogue_text}>>>"
                f"{candidate_text}\n\n"
                f"Nearby source context (the target is explicitly marked):\n{index.context(dialogue_line)}"
            ),
        },
    ]
    last_error = "No valid review"
    for step in range(1, 4):
        result = client.chat(
            messages,
            tools=REVIEW_TOOL,
            tool_choice={"type": "function", "function": {"name": "submit_emotion_review"}},
            temperature=0.0,
            max_tokens=5000,
            agent_role="emotion_" + review_role.replace(" ", "_"),
            trace_id=trace_id,
            agent_round=step,
        )
        validated = None
        for call in result.tool_calls:
            if call.name == "submit_emotion_review":
                validated, last_error = _validate(call.arguments, dialogue_index, dialogue_line, index)
        if validated:
            return validated
        if result.tool_calls:
            messages.append(_assistant_message(result))
            for call in result.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": f"Rejected: {last_error}"})
        else:
            messages.append({"role": "assistant", "content": result.content or ""})
        messages.append({"role": "user", "content": f"Invalid review: {last_error}. Submit a corrected review."})
    raise ValueError(last_error)


def label_emotion(
    dialogue_text: str,
    dialogue_line: int,
    dialogue_index: int,
    text: str,
    client: Optional[LLMClient] = None,
    max_tool_steps: int = 6,
    speaker: str = "",
    dialogues: Optional[list[dict]] = None,
    memory: str = "",
    verification_threshold: float = 0.68,
    always_verify: bool = True,
    _index: Optional[NovelIndex] = None,
) -> dict[str, Any]:
    """Label exactly one dialogue; uncertain labels receive a second API call."""
    client = client or LLMClient.for_flash_lite("emotion")
    index = _index or NovelIndex(text, dialogues)
    calls_before = client.usage_summary()["calls"]
    primary = _run_primary(
        dialogue_text, dialogue_line, dialogue_index, speaker, index, client, max_tool_steps, memory
    )
    final = primary
    reviewed = False
    if always_verify or primary["confidence"] < verification_threshold:
        reviewed = True
        review = _run_review(None, dialogue_text, dialogue_line, dialogue_index, speaker, index, client)
        if (review["emotion"], review["tone"]) == (primary["emotion"], primary["tone"]):
            final = dict(primary)
            final["confidence"] = round((primary["confidence"] + review["confidence"]) / 2, 4)
            final["evidence"] = f"Primary: {primary['evidence']} | Review: {review['evidence']}"
            final["evidence_lines"] = sorted(set(primary["evidence_lines"] + review["evidence_lines"]))
            final["decision_path"] = "independent_agreement"
            final["adjudicated"] = False
        else:
            dispute = {"primary": primary, "independent_review": review}
            final = _run_review(
                dispute,
                dialogue_text,
                dialogue_line,
                dialogue_index,
                speaker,
                index,
                client,
                review_role="final adjudicator resolving a disagreement between two agents",
            )
            final["adjudicated"] = True
            final["decision_path"] = "adjudicated_disagreement"
    final["primary_decision"] = primary
    if reviewed:
        final["review_decision"] = review
    final["reviewed"] = reviewed
    final["agent_calls"] = client.usage_summary()["calls"] - calls_before
    return final


def label_all_emotions(
    dialogues: list[dict],
    text: str,
    client: Optional[LLMClient] = None,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT,
    resume: bool = True,
    max_tool_steps: int = 6,
    item_retries: int = 3,
    max_items: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """Label a volume one dialogue at a time with an atomic resumable checkpoint."""
    client = client or LLMClient.for_flash_lite("emotion")
    usage_at_start = _normalise_usage_summary(client.usage_summary())
    index = NovelIndex(text, dialogues)
    source_hash = emotion_source_hash(text, dialogues)
    model_name = getattr(client, "sensenova_model", SENSENOVA_FLASH_LITE_MODEL)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    previous_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resume and checkpoint and checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            compatible = (
                payload.get("pipeline_version") == EMOTION_PIPELINE_VERSION
                and payload.get("source_hash") == source_hash
                and payload.get("model") == model_name
            )
            if compatible:
                raw_results = payload.get("results", {})
                if isinstance(raw_results, dict):
                    for key, value in raw_results.items():
                        try:
                            dialogue_index = int(key)
                            dialogue = dialogues[dialogue_index]
                        except (TypeError, ValueError, IndexError):
                            continue
                        speaker = str(dialogue.get("speaker", "")).strip()
                        if not speaker or speaker in {"旁白", "narrator", "Narrator"}:
                            continue
                        validated, _ = _validate(
                            value,
                            dialogue_index,
                            int(dialogue.get("line", 0) or 0),
                            index,
                        )
                        if validated:
                            results[str(dialogue_index)] = dict(value, **validated)
                errors = dict(payload.get("errors", {}))
                previous_usage = _normalise_usage_summary(payload.get("llm_usage", {}))
            else:
                logger.warning(
                    "Ignoring incompatible emotion checkpoint %s (version/source/model changed)",
                    checkpoint,
                )
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid emotion checkpoint: %s", checkpoint)

    newly_processed = 0
    for dialogue_index, dialogue in enumerate(dialogues):
        key = str(dialogue_index)
        speaker = str(dialogue.get("speaker", "")).strip()
        if not speaker or speaker in {"\u65c1\u767d", "narrator", "Narrator"}:
            continue
        if key in results:
            continue
        last_error: Optional[Exception] = None
        item_calls_before = client.usage_summary()["calls"]
        for attempt in range(1, item_retries + 1):
            try:
                value = label_emotion(
                    dialogue_text=str(dialogue.get("text", "")),
                    dialogue_line=int(dialogue.get("line", 0) or 0),
                    dialogue_index=dialogue_index,
                    text=text,
                    client=client,
                    max_tool_steps=max_tool_steps,
                    speaker=speaker,
                    dialogues=dialogues,
                    _index=index,
                )
                value["agent_calls"] = client.usage_summary()["calls"] - item_calls_before
                value["item_attempts"] = attempt
                results[key] = value
                errors.pop(key, None)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Emotion index %s attempt %d/%d failed: %s", key, attempt, item_retries, exc)
        if key not in results:
            errors[key] = str(last_error)
        if checkpoint:
            _write_checkpoint(
                checkpoint,
                results,
                errors,
                client,
                source_hash=source_hash,
                model_name=model_name,
                previous_usage=previous_usage,
                usage_at_start=usage_at_start,
            )
        if key not in results:
            raise EmotionBatchError(
                f"Dialogue {key} failed after {item_retries} attempts; rerun to resume from {checkpoint}: {last_error}"
            )
        logger.info("Emotion progress index=%s label=%s/%s", key, results[key]["emotion"], results[key]["tone"])
        newly_processed += 1
        if max_items is not None and newly_processed >= max_items:
            break
    return results


def _write_checkpoint(
    path: Path,
    results: dict[str, dict[str, Any]],
    errors: dict[str, str],
    client: LLMClient,
    *,
    source_hash: str,
    model_name: str,
    previous_usage: dict[str, int],
    usage_at_start: dict[str, int],
) -> None:
    current_usage = _normalise_usage_summary(client.usage_summary())
    usage_delta = {
        key: max(0, current_usage[key] - usage_at_start[key])
        for key in current_usage
    }
    payload = {
        "pipeline_version": EMOTION_PIPELINE_VERSION,
        "source_hash": source_hash,
        "model": model_name,
        "taxonomy": {"emotions": EMOTIONS, "tones": TONES},
        "results": results,
        "errors": errors,
        "llm_usage": _merge_usage_summaries(previous_usage, usage_delta),
    }
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


def _normalise_usage_summary(raw: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(raw.get(key, 0) or 0)
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
    }


def _merge_usage_summaries(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, int]:
    previous = _normalise_usage_summary(previous)
    current = _normalise_usage_summary(current)
    return {key: previous[key] + current[key] for key in previous}
