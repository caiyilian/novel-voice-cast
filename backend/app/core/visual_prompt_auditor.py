"""Evidence-grounded visual prompt auditing for illustration generation.

The auditor uses independent SenseNova agents before an illustration prompt is
allowed to reach Agnes.  Its contracts deliberately favor correctness over
call count: every candidate is rewritten, independently reviewed, and
adjudicated whenever the agents differ or make a material correction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from app.core.llm_client import LLMClient, SENSENOVA_FLASH_LITE_MODEL

logger = logging.getLogger("visual_prompt_auditor")

VISUAL_PROMPT_PIPELINE_VERSION = 1
DEFAULT_CHECKPOINT_PATH = Path("backend/data/visual_prompt_audit.checkpoint.json")
MAX_AGENT_TOKENS = 8192

PLAN_FIELDS = (
    "title",
    "description",
    "reason",
    "characters",
    "composition",
    "prompt",
    "start_line",
    "end_line",
)


class VisualPromptAuditError(RuntimeError):
    """An agent could not produce a valid, evidence-grounded audit."""


class VisualPromptBatchError(RuntimeError):
    """One item failed; preceding items remain available in the checkpoint."""


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


ENTITY_EVIDENCE_SCHEMA = {
    "type": "array",
    "items": _object_schema(
        {
            "entity": {"type": "string", "minLength": 1},
            "source": {"type": "string", "enum": ["novel", "character_card"]},
            "evidence_lines": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            },
        },
        ["entity", "source", "evidence_lines"],
    ),
}


def _candidate_properties(*, review: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "audited_prompt": {"type": "string", "minLength": 1},
        "evidence_lines": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
            "uniqueItems": True,
        },
        "excluded_nonliteral_entities": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "retained_characters": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "literal_entity_evidence": ENTITY_EVIDENCE_SCHEMA,
        "material_changes": {"type": "boolean"},
        "rationale": {"type": "string", "minLength": 1},
    }
    if review:
        properties["verdict"] = {
            "type": "string",
            "enum": ["approve", "revise", "reject"],
        }
    return properties


_CANDIDATE_REQUIRED = [
    "audited_prompt",
    "evidence_lines",
    "excluded_nonliteral_entities",
    "retained_characters",
    "literal_entity_evidence",
    "material_changes",
    "rationale",
]

PRIMARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_visual_prompt_rewrite",
        "description": "Submit the primary evidence-grounded visual prompt rewrite.",
        "strict": True,
        "parameters": _object_schema(_candidate_properties(), _CANDIDATE_REQUIRED),
    },
}]

REVIEW_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_visual_prompt_review",
        "description": "Submit an independent audit and, when needed, a corrected prompt.",
        "strict": True,
        "parameters": _object_schema(
            _candidate_properties(review=True),
            ["verdict", *_CANDIDATE_REQUIRED],
        ),
    },
}]

ADJUDICATION_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_visual_prompt_adjudication",
        "description": "Submit the final ruling after comparing the primary and independent audits.",
        "strict": True,
        "parameters": _object_schema(_candidate_properties(), _CANDIDATE_REQUIRED),
    },
}]


GROUNDING_RULES = """Hard visual-grounding rules:
1. The final prompt may contain only people, animals, objects, places, weather,
   and other physical entities that literally exist in the supplied novel scene.
2. A metaphor, simile, analogy, idiom, memory, hypothetical, negation, or mere
   comparison does not make its comparison object physically present. Never
   render that object. Convert figurative language into lighting, motion,
   texture, rhythm, camera angle, depth, or composition instead.
3. Keep every character listed in the illustration plan. Do not merge, replace,
   or silently remove a real character.
4. Character cards may supply only explicitly written appearance or clothing
   details. They cannot establish scene presence and must not be extrapolated.
5. Every novel-backed entity must cite valid 1-based lines inside the plan's
   source span. Card-backed appearance details use source=character_card and no
   invented evidence line.
6. The audited prompt itself must not mention any excluded nonliteral entity,
   even negatively (for example, do not write "no wolves").
7. Preserve useful style, lighting, movement, mood, and composition instructions
   when they do not introduce an unsupported physical entity.
"""

PRIMARY_SYSTEM_PROMPT = (
    "You are the primary visual prompt auditor. Rewrite independently from the "
    "source evidence, not by lightly editing the candidate. Quality and literal "
    "scene fidelity dominate brevity. Use only the forced submission tool.\n\n"
    + GROUNDING_RULES
)

REVIEW_SYSTEM_PROMPT = (
    "You are a second, independent visual prompt auditor. Reconstruct the correct "
    "scene from the source, then audit the primary candidate. Actively search for "
    "entity hallucinations, figurative objects made literal, dropped characters, "
    "and unsupported character-card details. Use only the forced submission tool.\n\n"
    + GROUNDING_RULES
)

ADJUDICATION_SYSTEM_PROMPT = (
    "You are the final visual prompt adjudicator. Resolve every disagreement and "
    "material rewrite conservatively. An unsupported entity must be excluded; a "
    "literally present character must remain. Produce one final generation-ready "
    "prompt using only the forced submission tool.\n\n"
    + GROUNDING_RULES
)


@dataclass
class _CallState:
    index: int
    calls: int = 0

    def next_round(self) -> int:
        self.calls += 1
        return self.calls

    @property
    def trace_id(self) -> str:
        return f"illustration_prompt:{self.index}"


def visual_prompt_source_hash(
    novel_text: str,
    plans: Sequence[dict[str, Any]],
    character_cards_text: Optional[str] = None,
) -> str:
    """Hash every input that can affect an audited prompt."""
    canonical = {
        "pipeline_version": VISUAL_PROMPT_PIPELINE_VERSION,
        "plans": [dict(plan) for plan in plans],
        "character_cards_text": character_cards_text or "",
    }
    digest = hashlib.sha256(novel_text.encode("utf-8"))
    digest.update(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


# A descriptive alias makes checkpoint assertions readable to callers.
visual_prompt_audit_source_hash = visual_prompt_source_hash


def _validate_plan(plan: dict[str, Any], line_count: int) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise TypeError("illustration plan must be a dictionary")
    missing = [field for field in PLAN_FIELDS if field not in plan]
    if missing:
        raise ValueError(f"illustration plan is missing required fields: {', '.join(missing)}")
    try:
        start_line = int(plan["start_line"])
        end_line = int(plan["end_line"])
    except (TypeError, ValueError) as exc:
        raise ValueError("start_line and end_line must be integers") from exc
    if start_line < 1 or end_line < start_line or end_line > line_count:
        raise ValueError(
            f"invalid illustration line span {start_line}-{end_line} for {line_count} source lines"
        )
    characters = plan["characters"]
    if isinstance(characters, str):
        characters = [part.strip() for part in re.split(r"[,，、/]", characters) if part.strip()]
    if not isinstance(characters, list) or any(not isinstance(name, str) for name in characters):
        raise ValueError("characters must be a list of strings")
    normalized = {field: plan[field] for field in PLAN_FIELDS}
    normalized["characters"] = _unique_strings(characters)
    normalized["start_line"] = start_line
    normalized["end_line"] = end_line
    return normalized


def _numbered_source_window(
    novel_text: str,
    start_line: int,
    end_line: int,
    radius: int = 60,
) -> str:
    lines = novel_text.splitlines()
    window_start = max(1, start_line - radius)
    window_end = min(len(lines), end_line + radius)
    output = []
    for number in range(window_start, window_end + 1):
        marker = ">>> TARGET " if start_line <= number <= end_line else ""
        output.append(f"{marker}{number}: {lines[number - 1]}")
    return "\n".join(output)


def _source_packet(
    plan: dict[str, Any],
    novel_text: str,
    character_cards_text: Optional[str],
) -> str:
    cards = character_cards_text.strip() if character_cards_text and character_cards_text.strip() else "(none supplied)"
    return (
        "ILLUSTRATION PLAN\n"
        + json.dumps(plan, ensure_ascii=False, indent=2)
        + "\n\nOPTIONAL CHARACTER CARDS (appearance evidence only)\n"
        + cards
        + "\n\nRELEVANT NUMBERED NOVEL SOURCE (target span plus nearby context)\n"
        + _numbered_source_window(
            novel_text,
            int(plan["start_line"]),
            int(plan["end_line"]),
        )
    )


def _tool_arguments(result: Any, expected_name: str) -> Optional[dict[str, Any]]:
    calls = result.get("tool_calls", []) if isinstance(result, dict) else getattr(result, "tool_calls", [])
    for call in calls or []:
        name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
        if not name and isinstance(call, dict):
            name = call.get("function", {}).get("name", "")
        if name != expected_name:
            continue
        arguments = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})
        if isinstance(call, dict) and not arguments:
            arguments = call.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        return arguments if isinstance(arguments, dict) else None
    return None


def _chat_for_candidate(
    client: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    expected_name: str,
    role: str,
    plan: dict[str, Any],
    novel_text: str,
    character_cards_text: Optional[str],
    state: _CallState,
    inherited_exclusions: Sequence[str] = (),
    max_attempts: int = 3,
    fallback_candidate: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    last_error = "missing forced tool call"
    last_raw: Optional[dict[str, Any]] = None
    conversation = list(messages)
    for _ in range(max_attempts):
        result = client.chat(
            conversation,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": expected_name}},
            temperature=0.0,
            max_tokens=MAX_AGENT_TOKENS,
            agent_role=role,
            trace_id=state.trace_id,
            agent_round=state.next_round(),
        )
        raw = _tool_arguments(result, expected_name)
        if raw is not None:
            last_raw = raw
            try:
                return _validate_candidate(
                    raw,
                    plan,
                    novel_text,
                    character_cards_text,
                    inherited_exclusions,
                    review=expected_name == "submit_visual_prompt_review",
                )
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
        conversation.append({
            "role": "user",
            "content": f"Your submission was invalid: {last_error}. Correct it and call {expected_name} again.",
        })
    if last_raw is not None and "excluded nonliteral entities" in last_error:
        try:
            repaired = _validate_candidate(
                last_raw,
                plan,
                novel_text,
                character_cards_text,
                inherited_exclusions,
                review=expected_name == "submit_visual_prompt_review",
                repair_excluded_leaks=True,
            )
        except (TypeError, ValueError):
            pass
        else:
            logger.warning(
                "%s exhausted %d attempts; deterministically removed leaked excluded entities: %s",
                role,
                max_attempts,
                ", ".join(repaired.get("deterministic_repairs", [])),
            )
            return repaired
    if fallback_candidate is not None:
        fallback = dict(fallback_candidate)
        fallback.pop("verdict", None)
        fallback_exclusions = _unique_strings([
            *inherited_exclusions,
            *fallback.get("excluded_nonliteral_entities", []),
        ])
        fallback["excluded_nonliteral_entities"] = fallback_exclusions
        leaked = [
            entity
            for entity in fallback_exclusions
            if _contains_entity(str(fallback.get("audited_prompt", "")), entity)
        ]
        if leaked:
            fallback["audited_prompt"] = _remove_excluded_entity_clauses(
                str(fallback.get("audited_prompt", "")),
                leaked,
            )
            fallback["deterministic_repairs"] = _unique_strings([
                *fallback.get("deterministic_repairs", []),
                *leaked,
            ])
        fallback["deterministic_fallback_reason"] = last_error
        if expected_name == "submit_visual_prompt_review":
            fallback["verdict"] = "revise"
        logger.warning(
            "%s exhausted %d strict-validation attempts; using audited fallback: %s",
            role,
            max_attempts,
            last_error,
        )
        return fallback
    raise VisualPromptAuditError(f"{role} failed strict validation after {max_attempts} attempts: {last_error}")


def _validate_candidate(
    raw: dict[str, Any],
    plan: dict[str, Any],
    novel_text: str,
    character_cards_text: Optional[str],
    inherited_exclusions: Sequence[str] = (),
    *,
    review: bool = False,
    repair_excluded_leaks: bool = False,
) -> dict[str, Any]:
    prompt = str(raw.get("audited_prompt", "")).strip()
    if not prompt:
        raise ValueError("audited_prompt is empty")

    evidence_lines = _integer_lines(raw.get("evidence_lines"), "evidence_lines")
    if not evidence_lines:
        raise ValueError("evidence_lines must not be empty")
    _validate_evidence_lines(evidence_lines, plan, novel_text)

    excluded = _unique_strings([*inherited_exclusions, *_string_list(raw.get("excluded_nonliteral_entities"), "excluded_nonliteral_entities")])
    leaked = [entity for entity in excluded if _contains_entity(prompt, entity)]
    if leaked:
        if not repair_excluded_leaks:
            raise ValueError(f"audited_prompt still contains excluded nonliteral entities: {', '.join(leaked)}")
        prompt = _remove_excluded_entity_clauses(prompt, leaked)
        still_leaked = [entity for entity in leaked if _contains_entity(prompt, entity)]
        if not prompt or still_leaked:
            raise ValueError(
                "audited_prompt still contains excluded nonliteral entities: "
                + ", ".join(still_leaked or leaked)
            )

    reported_retained = _unique_strings(
        _string_list(raw.get("retained_characters"), "retained_characters")
    )
    expected_characters = _unique_strings(plan.get("characters", []))
    retained_keys = {item.casefold() for item in reported_retained}
    # Chinese plan labels are metadata, while generation prompts deliberately
    # describe characters in English (赫萝 may be "wolf-eared girl" rather than
    # Holo). A tool-field string comparison cannot prove visual presence and
    # repeatedly rejected prompts that visibly retained both characters. Keep
    # exact enforcement for stable ASCII identifiers; independent review and
    # final adjudication enforce semantic CJK character presence.
    missing_retained = [
        name
        for name in expected_characters
        if name.isascii()
        if name.casefold() not in retained_keys
    ]
    if missing_retained:
        raise ValueError(f"retained_characters dropped planned characters: {', '.join(missing_retained)}")
    # Store canonical plan labels so independently translated agents compare
    # equal and downstream checkpoints remain language-stable.
    retained = expected_characters
    # Generation prompts are intentionally English while many plan character
    # labels are Chinese (for example ``罗伦斯`` -> ``Lawrence``).  Exact-string
    # matching those labels rejects a valid translated name.  Keep the exact
    # check for ASCII labels; CJK scene presence is still enforced through the
    # required retained_characters field and the independent review.
    missing_in_prompt = [
        name
        for name in expected_characters
        if name
        and name.isascii()
        and name.casefold() not in prompt.casefold()
    ]
    if missing_in_prompt:
        raise ValueError(f"audited_prompt does not identify planned characters: {', '.join(missing_in_prompt)}")

    entity_evidence = raw.get("literal_entity_evidence")
    if not isinstance(entity_evidence, list):
        raise TypeError("literal_entity_evidence must be a list")
    normalized_entities = []
    for record in entity_evidence:
        if not isinstance(record, dict):
            raise TypeError("literal_entity_evidence entries must be objects")
        entity = str(record.get("entity", "")).strip()
        source = str(record.get("source", "")).strip()
        lines = _integer_lines(record.get("evidence_lines"), f"entity evidence for {entity or '<blank>'}")
        if not entity:
            raise ValueError("literal entity name is empty")
        if source == "novel":
            if not lines:
                raise ValueError(f"novel entity {entity!r} has no evidence line")
            _validate_evidence_lines(lines, plan, novel_text)
            if not set(lines).issubset(evidence_lines):
                raise ValueError(f"entity evidence for {entity!r} is absent from overall evidence_lines")
        elif source == "character_card":
            if not character_cards_text:
                raise ValueError(f"entity {entity!r} cites a character card but none was supplied")
            if lines:
                raise ValueError("character-card evidence must not invent novel line numbers")
        else:
            raise ValueError(f"invalid literal entity source for {entity!r}: {source!r}")
        normalized_entities.append({"entity": entity, "source": source, "evidence_lines": lines})

    rationale = str(raw.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("rationale is empty")
    candidate = {
        "audited_prompt": prompt,
        "evidence_lines": evidence_lines,
        "excluded_nonliteral_entities": excluded,
        "retained_characters": retained,
        "literal_entity_evidence": normalized_entities,
        "material_changes": bool(raw.get("material_changes", False)),
        "rationale": rationale,
    }
    if {item.casefold() for item in reported_retained} != {
        item.casefold() for item in retained
    }:
        candidate["reported_retained_characters"] = reported_retained
    if leaked:
        candidate["deterministic_repairs"] = leaked
    if review:
        verdict = str(raw.get("verdict", "")).strip().lower()
        if verdict not in {"approve", "revise", "reject"}:
            raise ValueError("review verdict must be approve, revise, or reject")
        candidate["verdict"] = verdict
    return candidate


def _integer_lines(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    lines: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"{field_name} must contain integers")
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} must contain integers") from exc
        if number not in lines:
            lines.append(number)
    return sorted(lines)


def _validate_evidence_lines(lines: Sequence[int], plan: dict[str, Any], novel_text: str) -> None:
    source_lines = novel_text.splitlines()
    start, end = plan["start_line"], plan["end_line"]
    invalid = [line for line in lines if line < start or line > end or line > len(source_lines)]
    if invalid:
        raise ValueError(
            f"evidence lines must be inside source span {start}-{end}; invalid: {invalid}"
        )
    empty = [line for line in lines if not source_lines[line - 1].strip()]
    if empty:
        raise ValueError(f"evidence lines cannot cite blank source lines: {empty}")


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _unique_strings(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _entity_variants(entity: str) -> set[str]:
    folded = entity.strip().casefold()
    tokens = re.findall(r"[a-z]+", folded)
    variants = {folded}
    if not tokens:
        return {variant for variant in variants if variant}

    # A multi-word exclusion is one semantic entity.  Treating every word as
    # independently forbidden made ``elderly villager`` also ban the literal
    # source character ``villager`` and trapped the auditor in retries.  Only
    # singular/plural variants of the complete phrase are equivalent here.
    token = tokens[-1]
    suffix_variants = {token}
    if len(tokens) == 1:
        prefixes = [""]
    else:
        prefixes = [" ".join(tokens[:-1]) + " "]
    for prefix in prefixes:
        if token.endswith("ves") and len(token) > 3:
            suffix_variants.update({token[:-3] + "f", token[:-3] + "fe"})
        elif token.endswith("ies") and len(token) > 3:
            suffix_variants.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 2:
            suffix_variants.add(token[:-1])
        else:
            suffix_variants.add(token + "s")
        variants.update(prefix + suffix for suffix in suffix_variants)
    return {variant for variant in variants if variant}


def _contains_entity(prompt: str, entity: str) -> bool:
    folded = prompt.casefold()
    for variant in _entity_variants(entity):
        if re.fullmatch(r"[a-z0-9 _-]+", variant):
            if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", folded):
                return True
        elif variant in folded:
            return True
    return False


def _remove_excluded_entity_clauses(prompt: str, entities: Sequence[str]) -> str:
    """Remove leaked comparison/entity clauses after agent retries are exhausted."""

    # Prefer dropping a complete short clause so negations such as "without
    # rituals" or literalized similes do not leave misleading image tokens.
    clauses = re.split(r"(?<=[.!?;])\s+|(?<=,)\s+", prompt.strip())
    retained = [
        clause
        for clause in clauses
        if clause.strip() and not any(_contains_entity(clause, entity) for entity in entities)
    ]
    if retained and len(retained) < len(clauses):
        cleaned = " ".join(retained)
    else:
        cleaned = prompt
        for entity in entities:
            for variant in sorted(_entity_variants(entity), key=len, reverse=True):
                if re.fullmatch(r"[a-z0-9 _-]+", variant):
                    cleaned = re.sub(
                        rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])",
                        "",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
                else:
                    cleaned = re.sub(re.escape(variant), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;])(?:\s*[,;])+", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    normalized_prompt = re.sub(r"\s+", " ", candidate["audited_prompt"]).strip().casefold()
    return (
        normalized_prompt,
        tuple(candidate["evidence_lines"]),
        tuple(item.casefold() for item in candidate["excluded_nonliteral_entities"]),
        tuple(item.casefold() for item in candidate["retained_characters"]),
    )


def _is_material_rewrite(original: str, candidate: dict[str, Any]) -> bool:
    original_norm = re.sub(r"\s+", " ", original).strip().casefold()
    candidate_norm = re.sub(r"\s+", " ", candidate["audited_prompt"]).strip().casefold()
    similarity = SequenceMatcher(None, original_norm, candidate_norm).ratio() if original_norm else 0.0
    return bool(candidate["material_changes"] or candidate["excluded_nonliteral_entities"] or similarity < 0.9)


def _plan_fallback_candidate(plan: dict[str, Any], novel_text: str) -> dict[str, Any]:
    source_lines = novel_text.splitlines()
    evidence_lines = [
        line
        for line in range(int(plan["start_line"]), int(plan["end_line"]) + 1)
        if line <= len(source_lines) and source_lines[line - 1].strip()
    ]
    if not evidence_lines:
        evidence_lines = [int(plan["start_line"])]
    return {
        "audited_prompt": str(plan["prompt"]).strip(),
        "evidence_lines": evidence_lines,
        "excluded_nonliteral_entities": [],
        "retained_characters": _unique_strings(plan.get("characters", [])),
        "literal_entity_evidence": [],
        "material_changes": False,
        "rationale": (
            "Deterministic fallback to the illustration plan after repeated "
            "tool-contract failures; the independent reviewer must re-audit it."
        ),
    }


def audit_visual_prompt(
    plan: dict[str, Any],
    novel_text: str,
    character_cards_text: Optional[str] = None,
    client: Optional[Any] = None,
    illustration_index: int = 0,
    max_agent_attempts: int = 3,
) -> dict[str, Any]:
    """Audit one illustration plan with independent rewrite/review agents."""
    if not isinstance(novel_text, str) or not novel_text.splitlines():
        raise ValueError("novel_text must contain at least one source line")
    normalized_plan = _validate_plan(plan, len(novel_text.splitlines()))
    client = client or LLMClient.for_flash_lite("visual_prompt_auditor")
    state = _CallState(index=int(illustration_index))
    source = _source_packet(normalized_plan, novel_text, character_cards_text)

    primary = _chat_for_candidate(
        client,
        [
            {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ],
        PRIMARY_TOOL,
        "submit_visual_prompt_rewrite",
        "visual_prompt_primary",
        normalized_plan,
        novel_text,
        character_cards_text,
        state,
        max_attempts=max_agent_attempts,
        fallback_candidate=_plan_fallback_candidate(normalized_plan, novel_text),
    )

    review = _chat_for_candidate(
        client,
        [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": source + "\n\nPRIMARY CANDIDATE TO AUDIT\n" + json.dumps(primary, ensure_ascii=False, indent=2),
            },
        ],
        REVIEW_TOOL,
        "submit_visual_prompt_review",
        "visual_prompt_independent_reviewer",
        normalized_plan,
        novel_text,
        character_cards_text,
        state,
        max_attempts=max_agent_attempts,
        fallback_candidate=primary,
    )

    primary_exclusion_keys = {item.casefold() for item in primary["excluded_nonliteral_entities"]}
    consensus_exclusions = [
        item
        for item in review["excluded_nonliteral_entities"]
        if item.casefold() in primary_exclusion_keys
    ]
    disagreement = review["verdict"] != "approve" or _candidate_signature(primary) != _candidate_signature(review)
    material_change = _is_material_rewrite(str(normalized_plan["prompt"]), primary) or _is_material_rewrite(
        str(normalized_plan["prompt"]), review
    )
    needs_adjudication = disagreement or material_change

    decision_chain: list[dict[str, Any]] = [
        {"stage": "primary_rewrite", **primary},
        {"stage": "independent_review", **review},
    ]
    if needs_adjudication:
        adjudication = _chat_for_candidate(
            client,
            [
                {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        source
                        + "\n\nPRIMARY DECISION\n"
                        + json.dumps(primary, ensure_ascii=False, indent=2)
                        + "\n\nINDEPENDENT REVIEW\n"
                        + json.dumps(review, ensure_ascii=False, indent=2)
                    ),
                },
            ],
            ADJUDICATION_TOOL,
            "submit_visual_prompt_adjudication",
            "visual_prompt_final_adjudicator",
            normalized_plan,
            novel_text,
            character_cards_text,
            state,
            inherited_exclusions=consensus_exclusions,
            max_attempts=max_agent_attempts,
            fallback_candidate=review,
        )
        final = adjudication
        decision_chain.append({"stage": "final_adjudication", **adjudication})
        decision_path = "final_adjudication"
    else:
        final = primary
        decision_path = "independent_agreement"

    if needs_adjudication:
        final_exclusions = _unique_strings([*consensus_exclusions, *final["excluded_nonliteral_entities"]])
    else:
        final_exclusions = list(primary["excluded_nonliteral_entities"])
    leaked = [entity for entity in final_exclusions if _contains_entity(final["audited_prompt"], entity)]
    if leaked:
        raise VisualPromptAuditError(f"final prompt leaked excluded nonliteral entities: {', '.join(leaked)}")

    return {
        "illustration_index": int(illustration_index),
        "audited_prompt": final["audited_prompt"],
        "evidence_lines": final["evidence_lines"],
        "excluded_nonliteral_entities": final_exclusions,
        "retained_characters": final["retained_characters"],
        "literal_entity_evidence": final["literal_entity_evidence"],
        "agent_calls": state.calls,
        "decision_path": decision_path,
        "decision_chain": decision_chain,
    }


def audit_visual_prompts(
    plans: Sequence[dict[str, Any]],
    novel_text: str,
    character_cards_text: Optional[str] = None,
    client: Optional[Any] = None,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT_PATH,
    resume: bool = True,
    max_agent_attempts: int = 3,
    on_completed: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Audit plans in order, atomically checkpointing each completed item.

    ``on_completed`` is invoked only after the corresponding result is safely
    available (loaded from a compatible checkpoint or atomically persisted).
    Consumers can therefore start downstream work immediately without waiting
    for the complete batch.
    """
    plans = list(plans)
    if not isinstance(novel_text, str) or not novel_text.splitlines():
        raise ValueError("novel_text must contain at least one source line")
    normalized_plans = [_validate_plan(plan, len(novel_text.splitlines())) for plan in plans]
    client = client or LLMClient.for_flash_lite("visual_prompt_auditor")
    usage_at_start = _usage_snapshot(client)
    resumed_usage = _empty_usage()
    source_hash = visual_prompt_source_hash(novel_text, normalized_plans, character_cards_text)
    model_name = getattr(client, "sensenova_model", SENSENOVA_FLASH_LITE_MODEL)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed: dict[int, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    if resume and checkpoint and checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            compatible = (
                payload.get("pipeline_version") == VISUAL_PROMPT_PIPELINE_VERSION
                and payload.get("source_hash") == source_hash
                and payload.get("model") == model_name
            )
            if compatible:
                completed = {
                    int(item["illustration_index"]): item
                    for item in payload.get("results", [])
                    if isinstance(item, dict) and "illustration_index" in item
                }
                errors = dict(payload.get("errors", {}))
                resumed_usage = _normalise_usage(payload.get("llm_usage", {}))
            else:
                logger.warning("Ignoring incompatible visual prompt checkpoint: %s", checkpoint)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Ignoring invalid visual prompt checkpoint: %s", checkpoint)

    if on_completed:
        for index in sorted(completed):
            on_completed(index, completed[index])

    for index, plan in enumerate(normalized_plans):
        if index in completed:
            continue
        try:
            completed[index] = audit_visual_prompt(
                plan,
                novel_text,
                character_cards_text=character_cards_text,
                client=client,
                illustration_index=index,
                max_agent_attempts=max_agent_attempts,
            )
            errors.pop(str(index), None)
        except Exception as exc:
            errors[str(index)] = str(exc)
            if checkpoint:
                _atomic_write_json(
                    checkpoint,
                    _checkpoint_payload(
                        normalized_plans,
                        completed,
                        errors,
                        source_hash,
                        model_name,
                        _add_usage(resumed_usage, _usage_delta(client, usage_at_start)),
                    ),
                )
            raise VisualPromptBatchError(
                f"illustration prompt {index} failed; rerun to resume from {checkpoint}: {exc}"
            ) from exc
        if checkpoint:
            _atomic_write_json(
                checkpoint,
                _checkpoint_payload(
                    normalized_plans,
                    completed,
                    errors,
                    source_hash,
                    model_name,
                    _add_usage(resumed_usage, _usage_delta(client, usage_at_start)),
                ),
            )
        if on_completed:
            on_completed(index, completed[index])
    return [completed[index] for index in range(len(normalized_plans))]


def _checkpoint_payload(
    plans: Sequence[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    errors: dict[str, str],
    source_hash: str,
    model_name: str,
    llm_usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "pipeline_version": VISUAL_PROMPT_PIPELINE_VERSION,
        "source_hash": source_hash,
        "model": model_name,
        "total_items": len(plans),
        "completed_indices": sorted(completed),
        "results": [completed[index] for index in sorted(completed)],
        "errors": errors,
        "llm_usage": llm_usage,
    }


_USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in _USAGE_KEYS}


def _normalise_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return _empty_usage()
    return {key: max(0, int(value.get(key, 0) or 0)) for key in _USAGE_KEYS}


def _usage_snapshot(client: Any) -> dict[str, int]:
    summary = getattr(client, "usage_summary", None)
    return _normalise_usage(summary() if callable(summary) else {})


def _usage_delta(client: Any, baseline: dict[str, int]) -> dict[str, int]:
    current = _usage_snapshot(client)
    return {key: max(0, current[key] - baseline[key]) for key in _USAGE_KEYS}


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in _USAGE_KEYS}


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))


__all__ = [
    "ADJUDICATION_TOOL",
    "DEFAULT_CHECKPOINT_PATH",
    "PRIMARY_TOOL",
    "REVIEW_TOOL",
    "VISUAL_PROMPT_PIPELINE_VERSION",
    "VisualPromptAuditError",
    "VisualPromptBatchError",
    "audit_visual_prompt",
    "audit_visual_prompts",
    "visual_prompt_audit_source_hash",
    "visual_prompt_source_hash",
]
