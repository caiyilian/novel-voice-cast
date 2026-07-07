"""Phase 1 — Illustration Planning Agent.

The LLM reads the novel via tool calls and proposes illustrations for
scene changes, topic shifts, character reactions, and visual moments.

The novel is processed in chunks so each LLM session stays focused.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from app.core.llm_client import LLMClient, LLMResult, ToolCall
from app.core.parser import parse as parse_novel

logger = logging.getLogger("illustration_planner")

DEFAULT_OUTPUT_PATH = Path("output/illustration_plan.json")
DEFAULT_CHARACTER_CARD_PATH = Path("docs/角色卡.md")
DEFAULT_VISUAL_MEMORY_PATH = Path("output/character_visual_memory.json")
CHUNK_SIZE = 120
CHUNK_OVERLAP = 8
VISUAL_MEMORY_CHUNK_SIZE = 240
MAX_CHUNK_STEPS = 120
TEMPERATURE = 0.25
DIRECT_TEMPERATURE = 0.18
DIRECT_MAX_TOKENS = 8192
VISUAL_MEMORY_TEMPERATURE = 0.12
VISUAL_MEMORY_MAX_TOKENS = 8192
NO_TOOL_STREAK_LIMIT = 2
MIN_LINES_PER_ILLUSTRATION = 15
TARGET_LINES_PER_ILLUSTRATION = 5
CALL_LOG_PREVIEW_CHARS = 1200
CONTEXT_CHARS_PER_TOKEN = 4
CONTEXT_WARN_TOKENS = 24000
CONTEXT_HIGH_TOKENS = 48000
VISUAL_MEMORY_CONTEXT_CHAR_LIMIT = 5000


class PlanningError(Exception):
    pass


# ── Novel index ────────────────────────────────────────────────────────

class NovelIndex:
    def __init__(self, text: str, labels: Optional[list[str]] = None):
        self.text = text
        self.lines = text.splitlines()
        self.labels = labels or []

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    def get_annotated_text(self, start_line: int, end_line: int, limit: int = 350) -> dict:
        start, end = self._norm(start_line, end_line)
        sel = self.lines[start - 1 : end]
        trunc = False
        if len(sel) > limit:
            sel = sel[:limit]
            end = start + limit - 1
            trunc = True
        anno = []
        for off, line in enumerate(sel):
            ln = start + off
            lb = self.labels[ln - 1].strip() if ln <= len(self.labels) and self.labels[ln - 1].strip() else ""
            s = line.strip()
            if lb and s:
                anno.append(f"{ln}: [{lb}]{s}")
            else:
                anno.append(f"{ln}: {s}")
        return {"start_line": start, "end_line": end, "truncated": trunc, "text": "\n".join(anno)}

    def get_raw_text(self, start_line: int, end_line: int, limit: int = 350) -> dict:
        start, end = self._norm(start_line, end_line)
        sel = self.lines[start - 1 : end]
        trunc = False
        if len(sel) > limit:
            sel = sel[:limit]
            end = start + limit - 1
            trunc = True
        return {"start_line": start, "end_line": end, "truncated": trunc,
                "text": "\n".join(f"{start + off}: {line.strip()}" for off, line in enumerate(sel))}

    def get_character_list(self) -> list[dict]:
        seen = {}
        for lb in self.labels:
            for sp in _split_label_names(lb):
                if sp and sp != "旁白" and sp not in seen:
                    seen[sp] = {"name": sp}
        return list(seen.values())

    def _norm(self, a: int, b: int) -> tuple[int, int]:
        if self.total_lines <= 0:
            return 1, 0
        if a > b:
            a, b = b, a
        return max(1, min(a, self.total_lines)), min(max(1, b), self.total_lines)


# ── Tool specs ─────────────────────────────────────────────────────────

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_annotated_text",
        "description": "Read novel text with speaker labels embedded as [角色]. Lines are 1-based. Read broadly first, then drill into interesting sections.",
        "parameters": {"type": "object", "properties": {
            "start_line": {"type": "integer", "description": "Start line, 1-based"},
            "end_line": {"type": "integer", "description": "End line, inclusive"},
        }, "required": ["start_line", "end_line"]},
    }},
    {"type": "function", "function": {
        "name": "get_raw_text",
        "description": "Read raw novel text without speaker labels.",
        "parameters": {"type": "object", "properties": {
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        }, "required": ["start_line", "end_line"]},
    }},
    {"type": "function", "function": {
        "name": "get_character_list",
        "description": "Return all characters in the current section.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "propose_illustration",
        "description": "Propose one illustration for a specific moment. Call this for EVERY scene change, topic shift, character reaction, and visual detail you find.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Short Chinese title (5-15 chars)"},
            "start_line": {"type": "integer", "description": "Scene start line (1-based within this section)"},
            "end_line": {"type": "integer", "description": "Scene end line (inclusive)"},
            "description": {"type": "string", "description": "What happens, in Chinese"},
            "reason": {"type": "string", "description": "Why this deserves an illustration, e.g. scene change, character reaction, emotional beat"},
            "characters": {"type": "array", "items": {"type": "string"}, "description": "Characters in the illustration"},
            "composition": {"type": "string", "description": "Framing, e.g. 远景, 中景, 近景, 特写"},
            "prompt": {"type": "string", "description": "English prompt: character appearance, setting, lighting, mood, framing. End with: anime style, masterpiece, high quality"},
        }, "required": ["title", "start_line", "end_line", "description", "reason", "characters", "composition", "prompt"]},
    }},
    {"type": "function", "function": {
        "name": "list_proposals",
        "description": "List a compact review table of proposals so far. Long prompts are omitted to keep context small.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "finish_planning",
        "description": "Call when ALL illustrations for this section have been submitted.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


# ── System prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Illustration Planning Agent for a novel-to-audiobook project.

## Your job
Read the assigned section and propose dense, camera-shot-level illustrations.
This is not a "pick only the most important scenes" task. The audiobook may
show hundreds of illustrations, so every visible beat that can be drawn should
be submitted.

## What to illustrate
- A character appears, speaks, listens, hesitates, or reacts
- The topic, setting, time, distance, or camera angle changes
- A gesture, expression, object, prop, exchange, or action happens
- Tension rises, trust changes, a plan changes, or information is revealed
- A conversation of 40-60 lines can reasonably need 8-12 illustrations
- Target roughly one illustration per 4-8 meaningful lines when the text is visual

## Guidelines
- Use get_annotated_text to read with speaker labels (embedded as [角色]).
- Use propose_illustration to submit each moment.
- Prefer many small adjacent visual beats over one broad summary image.
- Submit at most 6 propose_illustration tool calls in one assistant turn; wait
  for tool confirmations, then continue with the next small batch.
- Avoid duplicate shots for the same line span and same visible action. If a
  duplicate slips in, continue; the program will dedupe after planning.
- Do not hard-code knowledge from any specific novel; use only the supplied text and labels.
- If character cards are provided, use them for stable identity cues.
- In every English prompt, include visible appearance, clothing, pose, expression,
  setting, lighting, and framing. Character names are allowed, but names alone are
  not enough for image generation.
- Review with list_proposals only to find missing beats, then add missing beats.
  Do not try to renumber, rewrite, or delete earlier proposals.
- Call finish_planning when the section is fully covered."""


DIRECT_SYSTEM_PROMPT = """You are an Illustration Planning Agent.

Return a dense illustration plan as JSON only. Do not explain.
The project is generic: use only the supplied section text and speaker labels.
Do not rely on any outside knowledge of a specific novel, character, or scene.

Output shape:
{
  "illustrations": [
    {
      "title": "short Chinese title",
      "start_line": 1,
      "end_line": 3,
      "description": "Chinese description of the visible beat",
      "reason": "why this beat deserves an illustration",
      "characters": ["speaker or visible character names"],
      "composition": "camera framing in Chinese",
      "prompt": "English image prompt with character names plus visible appearance/clothing/pose/expression/setting/lighting/framing, ending with: anime style, masterpiece, high quality"
    }
  ]
}"""


VISUAL_MEMORY_SYSTEM_PROMPT = """You are a Visual Memory Agent for illustration continuity.

Read the supplied novel section and extract only visual facts that are useful
for later image prompts: stable appearance, identity cues, clothing, props,
injuries/condition, current outfit/state, and explicit state changes.

Rules:
- Return JSON only. Do not explain.
- Use global line numbers from the supplied text.
- Do not invent visual facts. If the text or character card does not support a
  visual detail, omit it.
- Names are useful, but every visual fact must be a drawable detail.
- Stable facts belong in "stable". Temporary or changing facts belong in
  "states" with start_line/end_line/evidence_lines.
- If a character changes outfit, condition, carried object, or visible mood,
  add a new state instead of overwriting the old one.

Output shape:
{
  "characters": [
    {
      "name": "character name",
      "aliases": ["optional aliases"],
      "stable": {
        "identity": "stable role or species if visually useful",
        "appearance": "stable visible appearance",
        "visual_anchor": "compact English phrase for prompt reuse"
      },
      "states": [
        {
          "start_line": 12,
          "end_line": 20,
          "clothing": "temporary outfit",
          "props": ["visible prop"],
          "condition": "visible state",
          "expression": "visible emotion",
          "location": "where this state applies",
          "evidence_lines": [12, 15],
          "confidence": 0.8
        }
      ]
    }
  ]
}"""


# ── Chunk management ───────────────────────────────────────────────────

def _chunk_ranges(total_lines: int) -> list[tuple[int, int]]:
    if total_lines <= CHUNK_SIZE:
        return [(1, total_lines)]
    n = max(1, (total_lines + CHUNK_SIZE - 1) // CHUNK_SIZE)
    base = total_lines // n
    extra = total_lines % n
    chunks = []
    cur = 1
    for i in range(n):
        sz = base + (1 if i < extra else 0)
        cs = cur
        ce = min(total_lines, cur + sz - 1 + (CHUNK_OVERLAP if i < n - 1 else 0))
        chunks.append((cs, ce))
        cur += sz
    chunks[-1] = (chunks[-1][0], total_lines)
    return chunks


def _visual_memory_ranges(total_lines: int) -> list[tuple[int, int]]:
    if total_lines <= 0:
        return []
    ranges = []
    cur = 1
    while cur <= total_lines:
        end = min(total_lines, cur + VISUAL_MEMORY_CHUNK_SIZE - 1)
        ranges.append((cur, end))
        cur = end + 1
    return ranges


# ── Main orchestrator ─────────────────────────────────────────────────

def plan_illustrations(
    text: str,
    labels: Optional[list[str]] = None,
    client: Optional[LLMClient] = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    resume: bool = False,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    character_card_path: str | Path | None = DEFAULT_CHARACTER_CARD_PATH,
    visual_memory_path: str | Path | None = DEFAULT_VISUAL_MEMORY_PATH,
    enable_visual_memory: bool = True,
) -> list[dict]:
    if not text.strip():
        raise PlanningError("Novel text is empty")
    client = client or LLMClient()
    lines = text.splitlines()
    total = len(lines)
    label_lines = _line_labels_from_input(text, labels)
    chunks = _chunk_ranges(total)
    ckpt_path = Path(str(output_path).replace(".json", ".checkpoint.json"))
    raw_debug_dir = _resolve_debug_dir(output_path, debug_dir) if debug else None
    character_cards = _load_character_cards(character_card_path) if character_card_path else {}

    all_proposals: list[dict] = []
    completed: set[int] = set()
    story_summary = ""
    call_log: list[dict] = []
    if resume and ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        all_proposals = ck.get("proposals", [])
        completed = set(ck.get("completed_chunks", []))
        story_summary = ck.get("story_summary", "")
        call_log = ck.get("call_log", [])

    print(f"Novel: {total} lines, {len(chunks)} chunks")
    if completed:
        print(f"Resume: {len(completed)}/{len(chunks)} chunks done")

    visual_memory: dict[str, Any] = {}
    if enable_visual_memory and visual_memory_path:
        visual_memory_file = Path(visual_memory_path)
        if resume and visual_memory_file.exists():
            visual_memory = _load_visual_memory(visual_memory_file)
            print(
                f"Visual memory: loaded {len(visual_memory.get('characters', {}))} characters "
                f"from {visual_memory_file}"
            )
        else:
            print("Visual memory: scanning novel for appearance/state continuity")
            visual_memory = build_visual_memory(
                text=text,
                labels=labels,
                client=client,
                output_path=visual_memory_file,
                call_log=call_log,
                debug_dir=raw_debug_dir,
                character_cards=character_cards,
                line_labels=label_lines,
            )
    elif not enable_visual_memory:
        print("Visual memory: disabled")

    for ci, (cs, ce) in enumerate(chunks):
        if ci in completed:
            print(f"  Chunk {ci+1}/{len(chunks)} (lines {cs}-{ce}) — skip")
            continue

        chunk_text = "\n".join(lines[cs - 1 : ce])
        chunk_labels = label_lines[cs - 1 : ce] if label_lines else []
        ctx = [p for p in all_proposals if p.get("start_line", 0) < cs][-3:]

        print(f"\nChunk {ci+1}/{len(chunks)} (lines {cs}-{ce}, {ce-cs+1} lines)")
        print("-" * 40)

        result = _plan_chunk(
            chunk_text,
            chunk_labels,
            cs,
            client,
            ci,
            story_summary,
            ctx,
            call_log=call_log,
            debug_dir=raw_debug_dir,
            character_cards=character_cards,
            visual_memory=visual_memory,
        )

        if not result:
            _save_checkpoint(ckpt_path, all_proposals, completed, story_summary, call_log)
            raise PlanningError(
                f"Chunk {ci + 1} produced 0 proposals; see call_log/debug responses before resuming"
            )

        for p in result:
            p["start_line"] += cs - 1
            p["end_line"] += cs - 1
            all_proposals.append(p)

        events = [p.get("title", "?") for p in result[:5]]
        story_summary = " -> ".join(events) + (f" (+{len(result)-5} more)" if len(result) > 5 else "") if events else story_summary

        completed.add(ci)
        _save_checkpoint(ckpt_path, all_proposals, completed, story_summary, call_log)
        print(f"  -> {len(result)} proposals")

    merged = _filter_non_story_proposals(_dedupe_proposals(all_proposals), lines)
    _save_plan(
        output_path,
        merged,
        call_log=call_log,
        chunks=len(chunks),
        character_card_path=character_card_path,
        character_cards=character_cards,
        visual_memory_path=visual_memory_path,
        visual_memory=visual_memory,
    )
    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"\nDone: {len(merged)} illustrations")
    return merged


# ── Per-chunk planning agent ──────────────────────────────────────────

def _plan_chunk(
    chunk_text: str, chunk_labels: list[str], base_line: int,
    client: LLMClient, chunk_index: int,
    story_summary: str = "", context_prompts: Optional[list[dict]] = None,
    call_log: Optional[list[dict]] = None,
    debug_dir: Optional[Path] = None,
    character_cards: Optional[dict[str, dict[str, str]]] = None,
    visual_memory: Optional[dict[str, Any]] = None,
) -> list[dict]:
    """Run the tool-calling agent for one chunk."""
    index = NovelIndex(chunk_text, chunk_labels)
    proposals: list[dict] = []
    chars = [c["name"] for c in index.get_character_list()]
    character_context = _format_character_cards(chars, character_cards or {})
    global_end = base_line + index.total_lines - 1
    visual_context = _format_visual_memory_for_chunk(chars, visual_memory or {}, base_line, global_end)
    min_expected = _minimum_expected_proposals(index.total_lines)
    target_expected = _target_expected_proposals(index.total_lines)

    ctx = ""
    if story_summary:
        ctx += f"## Story so far:\n{story_summary}\n\n"
    if context_prompts:
        ctx += "## Previous illustrations (just before this section):\n"
        for cp in context_prompts:
            ctx += f"- {cp.get('title','?')}: {cp.get('description','')[:100]}\n"
        ctx += "\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"This section has {index.total_lines} local lines, starting at local line 1 "
            f"(global novel lines {base_line}-{global_end}).\n"
            f"{len(index.labels)} lines have speaker labels.\n"
            f"Characters: {chars}\n\n"
            f"{character_context}"
            f"{visual_context}"
            f"Density target: about {target_expected} illustrations for this section; "
            f"never finish with fewer than {min_expected} unless the section is truly non-visual.\n\n"
            f"{ctx}"
            "First call get_annotated_text for the whole section or a broad range, "
            "then call propose_illustration for every visual beat. "
            "Call finish_planning only after the section is densely covered."
        )},
    ]

    no_tool_streak = 0
    step = 0

    for step in range(1, MAX_CHUNK_STEPS + 1):
        # ── Call LLM ──
        result = None
        for retry in range(5):
            try:
                result = _chat_with_logging(
                    client,
                    messages,
                    tools=TOOL_SPECS,
                    temperature=TEMPERATURE,
                    call_log=call_log,
                    debug_dir=debug_dir,
                    chunk_index=chunk_index,
                    step=step,
                    phase="tool",
                    agent="illustration_planner",
                )
                break
            except Exception as e:
                logger.warning("Chunk %d step %d retry %d: %s", chunk_index, step, retry + 1, e)
                if retry >= 4:
                    raise PlanningError(f"Chunk {chunk_index} failed: {e}")
                time.sleep(5 * (retry + 1))

        if result is None:
            break

        # ── Debug log ──
        if result.content:
            logger.debug("Chunk %d step %d content: %s", chunk_index, step, result.content[:200])
        if result.tool_calls:
            names = [tc.name for tc in result.tool_calls]
            logger.debug("Chunk %d step %d tools: %s", chunk_index, step, names)

        # ── Handle tool calls ──
        if result.tool_calls:
            no_tool_streak = 0
            messages.append(_assistant_msg(result))
            finished = _execute(result.tool_calls, index, proposals, messages)
            if finished is not None:
                proposals = _sorted(proposals)
                if len(proposals) >= min_expected:
                    return proposals
                logger.warning(
                    "Chunk %d finished with only %d proposals (minimum %d); trying direct JSON fallback",
                    chunk_index,
                    len(proposals),
                    min_expected,
                )
                break
            continue

        # ── No tools — guide ──
        parsed = _extract_proposals_from_content(result.content, index)
        if parsed:
            proposals.extend(parsed)
            proposals = _dedupe_proposals(proposals)
            if len(proposals) >= min_expected:
                return _sorted(proposals)

        no_tool_streak += 1
        if no_tool_streak == 1:
            msg = "Call get_annotated_text to read the text, then propose_illustration for each visual beat you find."
        else:
            msg = "You must use the tools: get_annotated_text, propose_illustration. Call finish_planning when the section is done."
        messages.append({"role": "user", "content": msg})
        if no_tool_streak >= NO_TOOL_STREAK_LIMIT:
            logger.warning("Chunk %d returned no tool calls twice; trying direct JSON fallback", chunk_index)
            break

    direct = _plan_chunk_direct(
        index,
        client,
        chunk_index,
        story_summary=story_summary,
        context_prompts=context_prompts,
        call_log=call_log,
        debug_dir=debug_dir,
        character_cards=character_cards,
        visual_memory=visual_memory,
        base_line=base_line,
        start_step=step + 1,
    )
    if len(direct) >= len(proposals):
        return _sorted(direct)
    return _sorted(proposals)


def _plan_chunk_direct(
    index: NovelIndex,
    client: LLMClient,
    chunk_index: int,
    story_summary: str = "",
    context_prompts: Optional[list[dict]] = None,
    call_log: Optional[list[dict]] = None,
    debug_dir: Optional[Path] = None,
    character_cards: Optional[dict[str, dict[str, str]]] = None,
    visual_memory: Optional[dict[str, Any]] = None,
    base_line: int = 1,
    start_step: int = 1,
) -> list[dict]:
    """Fallback planner that includes the chunk text and asks for JSON only."""
    min_expected = _minimum_expected_proposals(index.total_lines)
    target_expected = _target_expected_proposals(index.total_lines)
    annotated = index.get_annotated_text(1, index.total_lines, limit=max(index.total_lines, 1))["text"]
    chars = [c["name"] for c in index.get_character_list()]
    character_context = _format_character_cards(chars, character_cards or {})
    global_end = base_line + index.total_lines - 1
    visual_context = _format_visual_memory_for_chunk(chars, visual_memory or {}, base_line, global_end)

    ctx = ""
    if story_summary:
        ctx += f"Story so far:\n{story_summary}\n\n"
    if context_prompts:
        ctx += "Previous illustrations just before this section:\n"
        for cp in context_prompts:
            ctx += f"- {cp.get('title', '?')}: {cp.get('description', '')[:100]}\n"
        ctx += "\n"

    messages = [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"This section has {index.total_lines} local lines "
            f"(global novel lines {base_line}-{global_end}). Speaker labels are embedded as [name].\n"
            f"Density target: about {target_expected} illustrations; minimum acceptable: {min_expected}.\n"
            "Prefer adjacent camera beats over broad summaries. Use only the supplied text.\n\n"
            f"{character_context}"
            f"{visual_context}"
            f"{ctx}"
            f"Section text:\n{annotated}\n\n"
            "Return JSON now."
        )},
    ]

    best: list[dict] = []
    for attempt in range(3):
        step = start_step + attempt
        try:
            result = _chat_with_logging(
                client,
                messages,
                temperature=DIRECT_TEMPERATURE,
                max_tokens=DIRECT_MAX_TOKENS,
                call_log=call_log,
                debug_dir=debug_dir,
                chunk_index=chunk_index,
                step=step,
                phase="direct",
                agent="direct_planner",
            )
        except Exception as exc:
            logger.warning("Chunk %d direct fallback attempt %d failed: %s", chunk_index, attempt + 1, exc)
            continue

        proposals = _extract_proposals_from_content(result.content, index)
        if len(proposals) > len(best):
            best = proposals
        if len(proposals) >= min_expected:
            return _sorted(proposals)

        messages.append({"role": "assistant", "content": result.content or ""})
        messages.append({"role": "user", "content": (
            f"Only {len(proposals)} valid illustrations were parsed. "
            f"Return corrected JSON with at least {min_expected} dense visual beats."
        )})

    if best:
        logger.warning(
            "Chunk %d direct fallback produced only %d proposals (minimum %d)",
            chunk_index,
            len(best),
            min_expected,
        )
    return _sorted(best)


# Visual memory agent --------------------------------------------------------

def build_visual_memory(
    text: str,
    labels: Optional[list[str]] = None,
    client: Optional[LLMClient] = None,
    output_path: str | Path | None = DEFAULT_VISUAL_MEMORY_PATH,
    call_log: Optional[list[dict]] = None,
    debug_dir: Optional[Path] = None,
    character_cards: Optional[dict[str, dict[str, str]]] = None,
    line_labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Scan the novel once for reusable visual continuity facts."""
    client = client or LLMClient()
    lines = text.splitlines()
    memory: dict[str, Any] = {
        "characters": {},
        "meta": {
            "total_lines": len(lines),
            "chunk_size": VISUAL_MEMORY_CHUNK_SIZE,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    if not lines:
        return memory

    labels_by_line = line_labels if line_labels is not None else _line_labels_from_input(text, labels)
    ranges = _visual_memory_ranges(len(lines))
    card_context = _format_character_cards(sorted((character_cards or {}).keys()), character_cards or {})

    for mi, (start_line, end_line) in enumerate(ranges):
        annotated = _annotated_global_text(lines, labels_by_line, start_line, end_line)
        known_context = _format_visual_memory_for_memory_agent(memory)
        messages = [
            {"role": "system", "content": VISUAL_MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Novel section: global lines {start_line}-{end_line}.\n"
                "Speaker labels are embedded as [name] when available.\n\n"
                f"{card_context}"
                f"{known_context}"
                "Extract new or changed visual memory from this section.\n\n"
                f"Section text:\n{annotated}\n\n"
                "Return JSON now."
            )},
        ]

        update: dict[str, Any] = {}
        for attempt in range(3):
            step = attempt + 1
            try:
                result = _chat_with_logging(
                    client,
                    messages,
                    temperature=VISUAL_MEMORY_TEMPERATURE,
                    max_tokens=VISUAL_MEMORY_MAX_TOKENS,
                    call_log=call_log,
                    debug_dir=debug_dir,
                    chunk_index=mi,
                    step=step,
                    phase="scan",
                    agent="visual_memory",
                )
                update = _extract_visual_memory_from_content(result.content)
                break
            except Exception as exc:
                logger.warning(
                    "Visual memory chunk %d/%d attempt %d failed: %s",
                    mi + 1,
                    len(ranges),
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        if update:
            _merge_visual_memory(memory, update)
        _save_visual_memory(output_path, memory)
        if output_path and call_log is not None:
            _save_call_log_snapshot(Path(str(output_path).replace(".json", ".call_log.json")), call_log)
        print(
            f"  Visual memory {mi+1}/{len(ranges)} "
            f"(lines {start_line}-{end_line}): {len(memory.get('characters', {}))} characters"
        )

    return memory


def _annotated_global_text(lines: list[str], labels: list[str], start_line: int, end_line: int) -> str:
    annotated = []
    for line_number in range(start_line, end_line + 1):
        text = lines[line_number - 1].strip() if 1 <= line_number <= len(lines) else ""
        label = labels[line_number - 1].strip() if labels and line_number <= len(labels) else ""
        if label and text:
            annotated.append(f"{line_number}: [{label}]{text}")
        else:
            annotated.append(f"{line_number}: {text}")
    return "\n".join(annotated)


def _load_visual_memory(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    memory_path = Path(path)
    if not memory_path.exists():
        return {}
    try:
        raw = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load visual memory %s: %s", memory_path, exc)
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("characters"), dict):
        return raw
    memory = {"characters": {}, "meta": raw.get("meta", {}) if isinstance(raw, dict) else {}}
    _merge_visual_memory(memory, raw)
    return memory


def _save_visual_memory(path: str | Path | None, memory: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    memory.setdefault("meta", {})["updated_at"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_call_log_snapshot(path: Path, call_log: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "agent_usage": _summarize_call_log_by_agent(call_log),
                "call_log": call_log,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _extract_visual_memory_from_content(content: str | None) -> dict[str, Any]:
    if not content:
        return {"characters": []}
    for candidate in _json_candidates(content):
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, list):
            return {"characters": raw}
        if isinstance(raw, dict):
            chars = raw.get("characters") or raw.get("visual_memory") or raw.get("items") or []
            if isinstance(chars, dict):
                chars = list(chars.values())
            if isinstance(chars, list):
                return {"characters": chars}
    return {"characters": []}


def _merge_visual_memory(memory: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    memory.setdefault("characters", {})
    characters = update.get("characters", []) if isinstance(update, dict) else []
    if isinstance(characters, dict):
        characters = list(characters.values())
    if not isinstance(characters, list):
        return memory

    for raw_char in characters:
        if not isinstance(raw_char, dict):
            continue
        name = str(raw_char.get("name") or raw_char.get("character") or "").strip()
        if not name:
            continue
        entry = memory["characters"].setdefault(
            name,
            {"name": name, "aliases": [], "stable": {}, "states": []},
        )
        entry.setdefault("aliases", [])
        entry.setdefault("stable", {})
        entry.setdefault("states", [])

        for alias in _normalise_str_list(raw_char.get("aliases", [])):
            if alias and alias != name and alias not in entry["aliases"]:
                entry["aliases"].append(alias)

        for key, value in _normalise_stable_visual_facts(raw_char).items():
            _merge_visual_fact(entry["stable"], key, value)

        existing_keys = {_visual_state_key(state) for state in entry["states"] if isinstance(state, dict)}
        for state in _normalise_visual_states(raw_char):
            state_key = _visual_state_key(state)
            if state_key in existing_keys:
                continue
            entry["states"].append(state)
            existing_keys.add(state_key)
        entry["states"] = sorted(entry["states"], key=lambda s: _safe_int(s.get("start_line")))
    return memory


def _normalise_stable_visual_facts(raw_char: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    stable = raw_char.get("stable") or raw_char.get("stable_facts") or {}
    if isinstance(stable, str):
        facts["description"] = stable.strip()
    elif isinstance(stable, dict):
        for key, value in stable.items():
            clean = _clean_visual_fact(value)
            if clean:
                facts[str(key).strip()] = clean

    for key in (
        "identity",
        "role",
        "species",
        "gender",
        "age",
        "appearance",
        "clothing",
        "visual_anchor",
        "description",
    ):
        clean = _clean_visual_fact(raw_char.get(key))
        if clean:
            facts[key] = clean
    return facts


def _normalise_visual_states(raw_char: dict[str, Any]) -> list[dict[str, Any]]:
    raw_states: list[Any] = []
    for key in ("states", "state_updates", "visual_states", "outfits"):
        value = raw_char.get(key)
        if isinstance(value, list):
            raw_states.extend(value)
        elif isinstance(value, dict):
            raw_states.append(value)
    if not raw_states and any(raw_char.get(k) for k in ("start_line", "from_line", "line")):
        raw_states.append(raw_char)

    states = []
    for raw_state in raw_states:
        if not isinstance(raw_state, dict):
            continue
        state: dict[str, Any] = {}
        start = _safe_int(raw_state.get("start_line") or raw_state.get("from_line") or raw_state.get("line"))
        end_value = raw_state.get("end_line") if raw_state.get("end_line") not in (None, "") else raw_state.get("to_line")
        end = _safe_int(end_value)
        if start:
            state["start_line"] = start
        if end:
            state["end_line"] = end

        for key in (
            "appearance",
            "clothing",
            "pose",
            "expression",
            "condition",
            "props",
            "location",
            "lighting",
            "visual_anchor",
            "notes",
        ):
            clean = _clean_visual_fact(raw_state.get(key))
            if clean:
                state[key] = clean

        evidence = _normalise_line_numbers(raw_state.get("evidence_lines") or raw_state.get("evidence"))
        if evidence:
            state["evidence_lines"] = evidence
        confidence = _normalise_confidence(raw_state.get("confidence"))
        if confidence is not None:
            state["confidence"] = confidence

        if any(k not in {"start_line", "end_line", "evidence_lines", "confidence"} for k in state):
            states.append(state)
    return states


def _format_visual_memory_for_memory_agent(memory: dict[str, Any]) -> str:
    characters = memory.get("characters", {}) if isinstance(memory, dict) else {}
    if not characters:
        return ""
    lines = ["## Known visual memory so far", "Use this to avoid repeating unchanged facts; add only new evidence or state changes."]
    for name in sorted(characters):
        line = _format_visual_memory_entry(name, characters[name], compact=True)
        if line:
            lines.append(line)
    return _clip_context_block("\n".join(lines) + "\n\n", VISUAL_MEMORY_CONTEXT_CHAR_LIMIT)


def _format_visual_memory_for_chunk(
    names: list[str],
    memory: dict[str, Any],
    start_line: int,
    end_line: int,
) -> str:
    characters = memory.get("characters", {}) if isinstance(memory, dict) else {}
    if not characters:
        return ""

    selected = [name for name in names if name in characters]
    if not selected:
        selected = [
            name for name, entry in characters.items()
            if _relevant_visual_states(entry.get("states", []), start_line, end_line)
        ]
    if not selected:
        return ""

    lines = [
        "## Visual memory for continuity",
        "Use these stable facts/current states in prompts. If the local text changes them, follow the local text.",
    ]
    for name in selected[:16]:
        line = _format_visual_memory_entry(name, characters[name], start_line, end_line)
        if line:
            lines.append(line)
    return _clip_context_block("\n".join(lines) + "\n\n", VISUAL_MEMORY_CONTEXT_CHAR_LIMIT)


def _format_visual_memory_entry(
    name: str,
    entry: dict[str, Any],
    start_line: int | None = None,
    end_line: int | None = None,
    compact: bool = False,
) -> str:
    parts = []
    stable = _format_visual_facts(entry.get("stable", {}))
    if stable:
        parts.append(f"stable: {stable}")

    states = entry.get("states", [])
    if start_line is not None and end_line is not None:
        states = _relevant_visual_states(states, start_line, end_line)
    elif compact:
        states = states[-2:]
    if states:
        state_bits = [_format_visual_state(state) for state in states[-4:]]
        state_bits = [bit for bit in state_bits if bit]
        if state_bits:
            parts.append("states: " + " | ".join(state_bits))
    if not parts:
        return ""
    return f"- {name}: " + "; ".join(parts)


def _relevant_visual_states(states: list[dict[str, Any]], start_line: int, end_line: int) -> list[dict[str, Any]]:
    before: list[dict[str, Any]] = []
    overlapping: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, dict):
            continue
        state_start = _safe_int(state.get("start_line"))
        state_end = _safe_int(state.get("end_line"))
        if state_start and state_start <= end_line and (not state_end or state_end >= start_line):
            overlapping.append(state)
        elif state_start and state_start < start_line:
            before.append(state)
    return (before[-1:] + overlapping[-4:])[-5:]


def _format_visual_state(state: dict[str, Any]) -> str:
    line = _safe_int(state.get("start_line"))
    prefix = f"line {line}" if line else "line ?"
    facts = _format_visual_facts(
        {k: v for k, v in state.items() if k not in {"start_line", "end_line", "evidence_lines", "confidence"}}
    )
    evidence = _normalise_line_numbers(state.get("evidence_lines"))
    suffix = f" evidence={evidence}" if evidence else ""
    return f"{prefix}: {facts}{suffix}".strip()


def _format_visual_facts(facts: Any) -> str:
    if not isinstance(facts, dict):
        clean = _clean_visual_fact(facts)
        return _format_fact_value(clean) if clean else ""
    bits = []
    for key, value in facts.items():
        clean = _clean_visual_fact(value)
        if clean:
            bits.append(f"{key}={_format_fact_value(clean)}")
    return "; ".join(bits)


def _clean_visual_fact(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        cleaned = {}
        for key, nested in value.items():
            nested_clean = _clean_visual_fact(nested)
            if nested_clean:
                cleaned[str(key).strip()] = nested_clean
        return cleaned
    text = str(value).strip()
    if text.lower() in {"unknown", "none", "n/a", "null"}:
        return ""
    return text


def _format_fact_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}:{_format_fact_value(v)}" for k, v in value.items())
    return str(value)


def _merge_visual_fact(facts: dict[str, Any], key: str, value: Any) -> None:
    if not value:
        return
    existing = facts.get(key)
    if not existing:
        facts[key] = value
        return
    existing_text = _format_fact_value(existing)
    value_text = _format_fact_value(value)
    if not value_text:
        return
    pieces = [piece.strip() for piece in existing_text.split("/") if piece.strip()]
    normalized = {_compact(piece) for piece in pieces}
    for piece in [piece.strip() for piece in value_text.split("/") if piece.strip()]:
        compact_piece = _compact(piece)
        if not compact_piece or compact_piece in normalized:
            continue
        if any(compact_piece in old or old in compact_piece for old in normalized):
            continue
        pieces.append(piece)
        normalized.add(compact_piece)
    facts[key] = " / ".join(pieces)


def _visual_state_key(state: dict[str, Any]) -> str:
    comparable = {k: v for k, v in state.items() if k not in {"confidence"}}
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True)


def _normalise_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[,/|]", str(value)) if part.strip()]


def _normalise_line_numbers(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        numbers = [_safe_int(item) for item in value]
    else:
        numbers = [_safe_int(item) for item in re.findall(r"\d+", str(value))]
    return sorted({number for number in numbers if number > 0})


def _normalise_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return None


def _clip_context_block(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[visual memory truncated]\n\n"


# LLM call accounting --------------------------------------------------------

def _chat_with_logging(
    client: LLMClient,
    messages: list[dict],
    *,
    call_log: Optional[list[dict]],
    debug_dir: Optional[Path],
    chunk_index: int,
    step: int,
    phase: str,
    agent: str,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    tool_choice: str = "auto",
) -> LLMResult:
    context_stats = _context_stats(messages, tools)
    if context_stats["estimated_tokens"] >= CONTEXT_WARN_TOKENS:
        logger.warning(
            "Agent %s chunk %d step %d context estimate is %d tokens (%s)",
            agent,
            chunk_index + 1,
            step,
            context_stats["estimated_tokens"],
            context_stats["risk"],
        )
    try:
        result = client.chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
    except Exception as exc:
        _record_llm_error(call_log, chunk_index, step, phase, exc, agent=agent, context=context_stats)
        raise
    _record_llm_call(call_log, debug_dir, result, chunk_index, step, phase, agent=agent, context=context_stats)
    return result


def _context_stats(messages: list[dict], tools: Optional[list[dict]] = None) -> dict[str, Any]:
    message_chars = sum(len(json.dumps(message, ensure_ascii=False, default=str)) for message in messages)
    tool_chars = len(json.dumps(tools, ensure_ascii=False, default=str)) if tools else 0
    total_chars = message_chars + tool_chars
    estimated_tokens = max(1, (total_chars + CONTEXT_CHARS_PER_TOKEN - 1) // CONTEXT_CHARS_PER_TOKEN)
    return {
        "message_count": len(messages),
        "tool_count": len(tools or []),
        "message_chars": message_chars,
        "tool_chars": tool_chars,
        "total_chars": total_chars,
        "estimated_tokens": estimated_tokens,
        "chars_per_token_estimate": CONTEXT_CHARS_PER_TOKEN,
        "occupancy_vs_warning": round(estimated_tokens / CONTEXT_WARN_TOKENS, 4),
        "risk": _context_risk(estimated_tokens),
    }


def _context_risk(estimated_tokens: int) -> str:
    if estimated_tokens >= CONTEXT_HIGH_TOKENS:
        return "high"
    if estimated_tokens >= CONTEXT_WARN_TOKENS:
        return "medium"
    return "low"


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "call"


# ── Tool execution ─────────────────────────────────────────────────────

def _execute(tool_calls: list[ToolCall], index: NovelIndex, proposals: list[dict], messages: list[dict]) -> Optional[list[dict]]:
    finished = None
    for pos, tc in enumerate(tool_calls, start=1):
        try:
            content = _run_tool(tc, index, proposals)
        except Exception as e:
            content = f"Tool error: {e}"
            logger.warning("Tool error: %s", e)
        messages.append({"role": "tool", "tool_call_id": _tool_call_id(tc, pos), "content": content})
        if tc.name == "finish_planning":
            finished = proposals[:]
    return finished


def _run_tool(tc: ToolCall, index: NovelIndex, proposals: list[dict]) -> str:
    a = tc.arguments
    n = tc.name
    if n == "get_annotated_text":
        start = _int(a, "start_line", 1)
        end = _int(a, "end_line", start)
        return json.dumps(index.get_annotated_text(start, end), ensure_ascii=False)
    if n == "get_raw_text":
        start = _int(a, "start_line", 1)
        end = _int(a, "end_line", start)
        return json.dumps(index.get_raw_text(start, end), ensure_ascii=False)
    if n == "get_character_list":
        return json.dumps(index.get_character_list(), ensure_ascii=False)
    if n == "propose_illustration":
        p = _normalise_proposal(a, index)
        proposals.append(p)
        return f"OK proposal {len(proposals)}: {p['title']}"
    if n == "list_proposals":
        return json.dumps(_proposal_review_table(proposals), ensure_ascii=False, indent=2)
    if n == "update_proposal":
        idx = _int(a, "proposal_index", 0)
        if idx < 1 or idx > len(proposals):
            return f"Invalid index {idx}"
        cur = dict(proposals[idx - 1])
        for k in ("title", "start_line", "end_line", "description", "reason", "characters", "composition", "prompt"):
            if k in a and a[k] is not None:
                cur[k] = a[k]
        proposals[idx - 1] = _normalise_proposal(cur, index)
        return f"Updated {idx}"
    if n == "delete_proposal":
        idx = _int(a, "proposal_index", 0)
        if idx < 1 or idx > len(proposals):
            return f"Invalid index {idx}"
        proposals.pop(idx - 1)
        return f"Deleted {idx}"
    if n == "finish_planning":
        if not proposals:
            return "No proposals yet. Read the text and submit illustrations first."
        return json.dumps({"illustrations": _sorted(proposals)}, ensure_ascii=False, indent=2)
    return f"Unknown tool: {n}"


# ── Helpers ────────────────────────────────────────────────────────────

def _assistant_msg(result: LLMResult) -> dict:
    return {
        "role": "assistant", "content": result.content or "",
        "tool_calls": [
            {
                "id": _tool_call_id(tc, pos),
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for pos, tc in enumerate(result.tool_calls, start=1)
        ],
    }


def _sorted(proposals: list[dict]) -> list[dict]:
    def _ln(p):
        try:
            return int(p.get("start_line", 0)) if p.get("start_line", "") != "" else 0
        except (TypeError, ValueError):
            return 0
    return sorted(proposals, key=_ln)


def _proposal_review_table(proposals: list[dict]) -> list[dict]:
    rows = []
    for idx, proposal in enumerate(_sorted(proposals), start=1):
        rows.append({
            "index": idx,
            "start_line": _safe_int(proposal.get("start_line")),
            "end_line": _safe_int(proposal.get("end_line")),
            "title": str(proposal.get("title", ""))[:40],
            "characters": proposal.get("characters", []),
            "composition": str(proposal.get("composition", ""))[:40],
            "description": _clip_text(proposal.get("description", ""), 120),
            "reason": _clip_text(proposal.get("reason", ""), 80),
        })
    return rows


def _clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _int(d: dict, key: str, default: int = 0) -> int:
    try:
        v = d.get(key, default)
        return int(v) if v != "" else default
    except (TypeError, ValueError):
        return default


def _minimum_expected_proposals(line_count: int) -> int:
    if line_count <= 0:
        return 0
    return max(1, (line_count + MIN_LINES_PER_ILLUSTRATION - 1) // MIN_LINES_PER_ILLUSTRATION)


def _line_labels_from_input(text: str, labels: Optional[list[str]]) -> list[str]:
    if not labels:
        return []
    total_lines = len(text.splitlines())
    if len(labels) == total_lines:
        return labels

    logger.warning(
        "labels.txt has %d lines but novel has %d lines; treating labels as dialogue-order labels",
        len(labels),
        total_lines,
    )
    line_labels = ["" for _ in range(total_lines)]
    dialogues, _ = parse_novel(text, labels)
    for dialogue in dialogues:
        line = _safe_int(dialogue.get("line"))
        speaker = str(dialogue.get("speaker") or "").strip()
        if line < 1 or line > total_lines or not speaker or speaker == "旁白":
            continue
        existing = _split_label_names(line_labels[line - 1])
        if speaker not in existing:
            existing.append(speaker)
            line_labels[line - 1] = " / ".join(existing)
    return line_labels


def _split_label_names(value: Any) -> list[str]:
    if value is None:
        return []
    return [p.strip() for p in re.split(r"[/|,，、]", str(value)) if p.strip()]


def _load_character_cards(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    card_path = Path(path)
    if not card_path.exists():
        logger.warning("Character card not found: %s", card_path)
        return {}

    cards: dict[str, dict[str, str]] = {}
    headers: list[str] = []
    for raw_line in card_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(re.fullmatch(r"-+", cell.replace(" ", "")) for cell in cells):
            continue
        if not headers:
            headers = [_normalise_card_header(cell) for cell in cells]
            continue
        if len(cells) < 1:
            continue
        row = {}
        for idx, value in enumerate(cells):
            key = headers[idx] if idx < len(headers) else f"extra_{idx}"
            row[key] = value.strip()
        name = row.get("name", "").strip()
        if name:
            cards[name] = row
    return cards


def _normalise_card_header(header: str) -> str:
    mapping = {
        "角色名": "name",
        "角色": "name",
        "name": "name",
        "character": "name",
        "对话数": "dialogue_count",
        "台词数": "dialogue_count",
        "dialogues": "dialogue_count",
        "性别": "gender",
        "gender": "gender",
        "置信度": "confidence",
        "confidence": "confidence",
        "外观": "appearance",
        "外貌": "appearance",
        "长相": "appearance",
        "appearance": "appearance",
        "服装": "clothing",
        "衣着": "clothing",
        "clothing": "clothing",
        "年龄": "age",
        "age": "age",
        "身份": "role",
        "职业": "role",
        "role": "role",
        "description": "description",
        "描述": "description",
        "参考图": "reference_image",
        "ref_image": "reference_image",
    }
    return mapping.get(header.strip().lower(), header.strip())


def _format_character_cards(names: list[str], cards: dict[str, dict[str, str]]) -> str:
    if not names or not cards:
        return ""
    lines = []
    for name in names:
        card = cards.get(name)
        if not card:
            continue
        facts = []
        for key, label in [
            ("gender", "gender"),
            ("dialogue_count", "dialogues"),
            ("confidence", "confidence"),
            ("role", "role"),
            ("age", "age"),
            ("appearance", "appearance"),
            ("clothing", "clothing"),
            ("description", "description"),
            ("reference_image", "reference_image"),
        ]:
            value = str(card.get(key, "")).strip()
            if value:
                facts.append(f"{label}={value}")
        if facts:
            lines.append(f"- {name}: " + "; ".join(facts))
    if not lines:
        return ""
    return (
        "## Character cards\n"
        + "\n".join(lines)
        + "\nUse these as stable identity hints. If appearance is missing, infer visible details from the section text and include them in prompts.\n\n"
    )


def _target_expected_proposals(line_count: int) -> int:
    if line_count <= 0:
        return 0
    return max(1, (line_count + TARGET_LINES_PER_ILLUSTRATION - 1) // TARGET_LINES_PER_ILLUSTRATION)


def _normalise_proposal(raw: dict, index: NovelIndex) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("proposal must be an object")
    start = _bounded_line(_int(raw, "start_line", 1), index.total_lines)
    end = _bounded_line(_int(raw, "end_line", start), index.total_lines)
    if end < start:
        start, end = end, start

    title = str(raw.get("title") or raw.get("name") or "").strip()
    description = str(raw.get("description") or raw.get("scene") or raw.get("summary") or "").strip()
    if not title and description:
        title = description[:12]
    if not title:
        title = f"插图{start}"
    if not description:
        description = title

    prompt = str(raw.get("prompt") or "").strip()
    if prompt and "anime style" not in prompt.lower():
        prompt = prompt.rstrip(" .,") + ", anime style, masterpiece, high quality"
    if not prompt:
        prompt = (
            "cinematic novel illustration, expressive characters, detailed setting, "
            "anime style, masterpiece, high quality"
        )

    return {
        "title": title,
        "start_line": start,
        "end_line": end,
        "description": description,
        "reason": str(raw.get("reason") or "").strip(),
        "characters": _normalise_characters(raw.get("characters", [])),
        "composition": str(raw.get("composition") or raw.get("framing") or "").strip(),
        "prompt": prompt,
    }


def _normalise_characters(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，、/|]", value)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _bounded_line(value: int, total_lines: int) -> int:
    if total_lines <= 0:
        return 1
    return max(1, min(value, total_lines))


def _extract_proposals_from_content(content: str, index: NovelIndex) -> list[dict]:
    if not content:
        return []

    for candidate in _json_candidates(content):
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            raw = raw.get("illustrations") or raw.get("proposals") or raw.get("items") or []
        if not isinstance(raw, list):
            continue
        proposals = []
        for item in raw:
            try:
                proposals.append(_normalise_proposal(item, index))
            except (TypeError, ValueError):
                continue
        if proposals:
            return _dedupe_proposals(proposals)
    return []


def _json_candidates(content: str) -> list[str]:
    candidates = [content.strip()]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE):
        candidates.insert(0, match.group(1).strip())
    match = re.search(r"(\{.*\}|\[.*\])", content, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1).strip())
    return [c for c in candidates if c]


def _dedupe_proposals(all_proposals: list[dict]) -> list[dict]:
    """Remove exact duplicates without merging adjacent or overlapping camera beats."""
    deduped = []
    seen = set()
    by_range: dict[tuple[int, int], list[dict]] = {}
    for proposal in _sorted(all_proposals):
        key = (
            _int(proposal, "start_line", 0),
            _int(proposal, "end_line", 0),
            _compact(proposal.get("title", "")),
            _compact(proposal.get("description", ""))[:80],
        )
        if key in seen:
            continue
        range_key = (_int(proposal, "start_line", 0), _int(proposal, "end_line", 0))
        if any(_is_near_duplicate_proposal(proposal, existing) for existing in by_range.get(range_key, [])):
            continue
        seen.add(key)
        deduped.append(proposal)
        by_range.setdefault(range_key, []).append(proposal)
    return deduped


def _merge_proposals(all_proposals: list[dict]) -> list[dict]:
    return _dedupe_proposals(all_proposals)


def _filter_non_story_proposals(proposals: list[dict], source_lines: list[str]) -> list[dict]:
    afterword_start = _first_marker_line(source_lines, {"后记"})
    filtered = []
    for proposal in proposals:
        start = _safe_int(proposal.get("start_line"))
        end = _safe_int(proposal.get("end_line"))
        if afterword_start and start >= afterword_start:
            continue
        span = source_lines[max(0, start - 1): min(len(source_lines), end)]
        if span and all(_is_non_story_marker_line(line) for line in span if str(line).strip()):
            continue
        filtered.append(proposal)
    return filtered


def _first_marker_line(source_lines: list[str], markers: set[str]) -> int:
    for idx, line in enumerate(source_lines, start=1):
        if str(line).strip() in markers:
            return idx
    return 0


def _is_non_story_marker_line(line: Any) -> bool:
    text = str(line).strip()
    if not text:
        return True
    if text in {"插图", "完", "后记", "终幕"}:
        return True
    if re.fullmatch(r"第?[一二三四五六七八九十零〇百千]+幕", text):
        return True
    if re.fullmatch(r"[一二三四五六七八九十零〇百千第幕终后记]+", text) and len(text) <= 4:
        return True
    return False


def _is_near_duplicate_proposal(a: dict, b: dict) -> bool:
    text_a = _compact(f"{a.get('title', '')} {a.get('description', '')}")
    text_b = _compact(f"{b.get('title', '')} {b.get('description', '')}")
    if not text_a or not text_b:
        return False
    if text_a in text_b or text_b in text_a:
        return True
    return SequenceMatcher(None, text_a, text_b).ratio() >= 0.62


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _tool_call_id(tool_call: ToolCall, pos: int) -> str:
    return tool_call.id or f"call_{pos}"


def _resolve_debug_dir(output_path: str | Path, debug_dir: str | Path | None) -> Path:
    if debug_dir is not None:
        return Path(debug_dir)
    path = Path(output_path)
    if path.suffix:
        return path.with_suffix(".debug")
    return Path(str(path) + ".debug")


def _record_llm_call(
    call_log: Optional[list[dict]],
    debug_dir: Optional[Path],
    result: LLMResult,
    chunk_index: int,
    step: int,
    phase: str,
    agent: str = "illustration_planner",
    context: Optional[dict[str, Any]] = None,
) -> None:
    if call_log is None:
        return
    usage = result.usage or {}
    tool_names = [tc.name for tc in result.tool_calls]
    context_data = dict(context or {})
    context_data["reported_prompt_tokens"] = _safe_int(usage.get("prompt_tokens"))
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk": chunk_index + 1,
        "step": step,
        "agent": agent,
        "phase": phase,
        "status": "success",
        "model": result.model,
        "account_index": result.account_index,
        "usage": {
            "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
            "completion_tokens": _safe_int(usage.get("completion_tokens")),
            "total_tokens": _safe_int(usage.get("total_tokens")),
        },
        "context": context_data,
        "tool_calls": tool_names,
        "tool_calls_count": len(tool_names),
        "content_preview": (result.content or "")[:CALL_LOG_PREVIEW_CHARS],
    }
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_name = _safe_filename(f"{agent}_{phase}")
        raw_path = debug_dir / f"chunk_{chunk_index + 1:03d}_step_{step:03d}_{raw_name}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "entry": entry,
                    "content": result.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in result.tool_calls
                    ],
                    "raw": result.raw,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        entry["raw_path"] = str(raw_path)
    call_log.append(entry)


def _record_llm_error(
    call_log: Optional[list[dict]],
    chunk_index: int,
    step: int,
    phase: str,
    exc: Exception,
    agent: str = "illustration_planner",
    context: Optional[dict[str, Any]] = None,
) -> None:
    if call_log is None:
        return
    call_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk": chunk_index + 1,
        "step": step,
        "agent": agent,
        "phase": phase,
        "status": "error",
        "model": "",
        "account_index": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "context": context or {},
        "tool_calls": [],
        "tool_calls_count": 0,
        "error": str(exc),
    })


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _summarize_call_log_by_agent(call_log: list[dict]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for entry in call_log:
        agent = str(entry.get("agent") or entry.get("phase") or "unknown")
        bucket = summary.setdefault(
            agent,
            {
                "calls": 0,
                "errors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_context_tokens": 0,
                "max_estimated_context_tokens": 0,
                "max_reported_prompt_tokens": 0,
            },
        )
        if entry.get("status") == "success":
            bucket["calls"] += 1
        elif entry.get("status") == "error":
            bucket["errors"] += 1
        usage = entry.get("usage", {})
        context = entry.get("context", {})
        bucket["prompt_tokens"] += _safe_int(usage.get("prompt_tokens"))
        bucket["completion_tokens"] += _safe_int(usage.get("completion_tokens"))
        bucket["total_tokens"] += _safe_int(usage.get("total_tokens"))
        estimated = _safe_int(context.get("estimated_tokens"))
        reported_prompt = _safe_int(context.get("reported_prompt_tokens"))
        bucket["estimated_context_tokens"] += estimated
        bucket["max_estimated_context_tokens"] = max(bucket["max_estimated_context_tokens"], estimated)
        bucket["max_reported_prompt_tokens"] = max(bucket["max_reported_prompt_tokens"], reported_prompt)
    return summary


def _save_checkpoint(
    path: Path,
    proposals: list[dict],
    completed: set[int],
    story_summary: str,
    call_log: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "proposals": proposals,
                "completed_chunks": sorted(completed),
                "story_summary": story_summary,
                "call_log": call_log,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _save_plan(
    path: str | Path,
    proposals: list[dict],
    call_log: Optional[list[dict]] = None,
    chunks: int = 0,
    character_card_path: str | Path | None = None,
    character_cards: Optional[dict[str, dict[str, str]]] = None,
    visual_memory_path: str | Path | None = None,
    visual_memory: Optional[dict[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    call_log = call_log or []
    total_tokens = sum(_safe_int(entry.get("usage", {}).get("total_tokens")) for entry in call_log)
    context_estimated_tokens = [
        _safe_int(entry.get("context", {}).get("estimated_tokens"))
        for entry in call_log
        if entry.get("context")
    ]
    visual_characters = len((visual_memory or {}).get("characters", {}))
    p.write_text(
        json.dumps(
            {
                "illustrations": proposals,
                "summary": {
                    "total": len(proposals),
                    "total_illustrations": len(proposals),
                    "total_llm_calls": sum(1 for entry in call_log if entry.get("status") == "success"),
                    "total_tokens": total_tokens,
                    "total_estimated_context_tokens": sum(context_estimated_tokens),
                    "max_estimated_context_tokens": max(context_estimated_tokens, default=0),
                    "chunks": chunks,
                    "character_card_path": str(character_card_path) if character_card_path else "",
                    "character_cards_loaded": len(character_cards or {}),
                    "visual_memory_path": str(visual_memory_path) if visual_memory_path else "",
                    "visual_memory_characters": visual_characters,
                    "agent_usage": _summarize_call_log_by_agent(call_log),
                },
                "call_log": call_log,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Saved: %s (%d proposals)", p, len(proposals))
    return p
