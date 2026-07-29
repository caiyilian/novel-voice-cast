"""Deterministic quality controls for long-form VoxCPM audiobook synthesis."""

from __future__ import annotations

import re
from typing import Any


TTS_CHUNKING_VERSION = 1
TTS_CONTROL_VERSION = 2

_SEMANTIC_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
_SENTENCE_BREAKS = frozenset("。！？!?；;")
_CLAUSE_BREAKS = frozenset("，,：:、")


def semantic_char_count(text: str) -> int:
    """Count characters that normally consume spoken time."""
    return len(_SEMANTIC_RE.findall(str(text)))


def _split_position(text: str, max_chars: int, min_chars: int) -> int:
    semantic = 0
    sentence_candidates: list[tuple[int, int]] = []
    clause_candidates: list[tuple[int, int]] = []
    hard_position = len(text)
    for position, character in enumerate(text, 1):
        if _SEMANTIC_RE.match(character):
            semantic += 1
        if character in _SENTENCE_BREAKS:
            sentence_candidates.append((semantic, position))
        elif character in _CLAUSE_BREAKS or character.isspace():
            clause_candidates.append((semantic, position))
        if semantic >= max_chars:
            hard_position = position
            break

    for candidates in (sentence_candidates, clause_candidates):
        usable = [position for count, position in candidates if min_chars <= count <= max_chars]
        if usable:
            return usable[-1]
    return max(1, hard_position)


def split_tts_text(text: str, *, max_chars: int = 48, min_chars: int = 8) -> list[str]:
    """Split at semantic punctuation while preserving every source character."""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not value:
        return [""]
    max_chars = max(8, int(max_chars))
    min_chars = max(1, min(int(min_chars), max_chars // 2))
    pieces: list[str] = []
    remaining = value
    while semantic_char_count(remaining) > max_chars:
        position = _split_position(remaining, max_chars, min_chars)
        pieces.append(remaining[:position])
        remaining = remaining[position:]
    if remaining:
        pieces.append(remaining)

    # Avoid a tiny final fragment when it can safely join its predecessor.
    if (
        len(pieces) >= 2
        and semantic_char_count(pieces[-1]) < min_chars
        and semantic_char_count(pieces[-2] + pieces[-1]) <= max_chars
    ):
        pieces[-2:] = [pieces[-2] + pieces[-1]]
    if "".join(pieces) != value:
        raise ValueError("TTS splitter did not preserve source text")
    return pieces


_STYLE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("沉稳", ("沉稳", "深沉", "厚重")),
    ("平稳", ("平稳", "平静", "自然带出", "客观")),
    ("温和", ("温和", "温柔", "柔和", "亲切")),
    ("克制", ("克制", "不夸张", "不煽情", "不滥情")),
    ("冷静", ("冷静", "冷淡", "冷漠")),
    ("严肃", ("严肃", "正式", "事务性", "权威")),
    ("轻快", ("轻快", "欢快", "活泼", "俏皮")),
    ("喜悦", ("喜悦", "开心", "高兴", "满足")),
    ("忧伤", ("忧伤", "悲伤", "低落", "哀伤", "乡愁", "寂寥", "失望", "无奈")),
    ("紧张", ("紧张", "焦虑", "警惕", "戒备")),
    ("恐惧", ("恐惧", "惊恐", "害怕", "畏惧")),
    ("愤怒", ("愤怒", "恼怒", "生气", "怒意")),
    ("讽刺", ("讽刺", "嘲讽", "挖苦")),
    ("轻蔑", ("轻蔑", "鄙夷", "不屑")),
    ("亲密", ("亲密", "暧昧", "宠溺", "爱意")),
    ("神秘", ("神秘", "悬疑", "不安", "诡异")),
    ("威严", ("威严", "庄重", "宏大")),
)


def _pace(control: str, pace_hint: str = "") -> str:
    structured = str(pace_hint or "").strip().lower()
    if structured in {"very_slow", "slow"}:
        return "语速舒缓"
    if structured in {"fast", "brisk"}:
        return "语速稍快"
    if structured in {"measured", "natural", "variable"}:
        return "节奏适中"
    if any(token in control for token in ("舒缓", "缓慢", "放慢", "稍缓", "从容", "不急促")):
        return "语速舒缓"
    cleaned = control.replace("不急促", "")
    if any(token in cleaned for token in ("急促", "快速", "加快", "稍快")):
        return "语速稍快"
    return "节奏适中"


def compact_performance_control(
    control: str,
    *,
    speaker: str = "",
    emotion: str = "",
    pace_hint: str = "",
    max_chars: int = 32,
) -> str:
    """Reduce verbose directing prose to a safe, non-quoting VoxCPM prefix."""
    source = re.sub(r"\s+", "", str(control or ""))
    styles: list[str] = []
    for canonical, tokens in _STYLE_GROUPS:
        if any(token in source for token in tokens):
            styles.append(canonical)
        if len(styles) >= 3:
            break
    if not styles:
        emotion_defaults = {
            "happy": "轻快",
            "sad": "忧伤",
            "angry": "愤怒",
            "surprised": "惊讶",
            "nervous": "紧张",
            "cold": "冷静",
        }
        styles.append(emotion_defaults.get(str(emotion), "沉稳" if speaker == "旁白" else "自然"))
    if "轻声" in source or "低声" in source or "耳语" in source:
        styles.append("轻声")
    elif "大声" in source or "高声" in source or "响亮" in source:
        styles.append("音量稍高")
    # The director emits a constrained structured pace field. Prefer it over
    # prose such as "节奏稍缓" or "不急促": those phrases appeared in many
    # otherwise ``measured`` directions and previously collapsed 75.8% of the
    # novel into the much stronger "语速舒缓" instruction.
    styles.append(_pace(source, pace_hint))
    output = "，".join(dict.fromkeys(styles))
    limit = max(12, int(max_chars))
    while len(output) > limit and len(styles) > 2:
        styles.pop(-2)
        output = "，".join(dict.fromkeys(styles))
    return output[:limit].rstrip("，")


def control_variants(control: str) -> list[str]:
    """Return progressively safer controls for quality retries."""
    value = str(control or "").strip()
    pace = _pace(value)
    values = [value]
    if "语速稍快" in pace:
        values.extend(("自然清晰，语速稍快", "吐字清晰，节奏适中"))
    elif "语速舒缓" in pace:
        values.extend(("自然清晰，语速舒缓", "吐字清晰，语速舒缓"))
    else:
        values.extend(("自然清晰，节奏适中", "吐字清晰，节奏适中"))
    values.append("")
    return list(dict.fromkeys(values))


def duration_quality_bounds(
    text: str,
    control: str,
    *,
    min_ratio: float = 0.92,
    max_ratio: float = 2.5,
) -> dict[str, Any]:
    """Estimate a spoken-duration envelope without prescribing exact timing.

    ``expected`` remains useful for diagnostics and the upper leak guard.  The
    lower bound is intentionally only an anomaly detector: VoxCPM owns the
    performance tempo, and a valid take must never be time-stretched merely to
    match this estimate.
    """
    value = str(text)
    semantic = max(1, semantic_char_count(value))
    pace = _pace(str(control))
    seconds_per_char = 0.17 if pace == "语速稍快" else 0.25 if pace == "语速舒缓" else 0.21
    punctuation_seconds = (
        sum(value.count(mark) for mark in "，,、：:") * 0.16
        + sum(value.count(mark) for mark in "；;") * 0.24
        + sum(value.count(mark) for mark in "。！？!?") * 0.30
        + (value.count("……") + value.count("...")) * 0.45
    )
    expected = semantic * seconds_per_char + punctuation_seconds
    minimum = max(0.35, expected * max(0.5, float(min_ratio)))
    maximum = max(2.4, expected * max(1.2, float(max_ratio)) + 0.8)
    return {
        "semantic_chars": semantic,
        "pace": pace,
        "expected_duration_seconds": round(expected, 4),
        "min_duration_seconds": round(minimum, 4),
        "max_duration_seconds": round(maximum, 4),
    }


def duration_quality_problems(duration: float, bounds: dict[str, Any]) -> list[str]:
    problems = []
    minimum = float(bounds["min_duration_seconds"])
    maximum = float(bounds["max_duration_seconds"])
    if float(duration) < minimum:
        problems.append(f"audio is too fast: {duration:.3f}s < {minimum:.3f}s")
    if float(duration) > maximum:
        problems.append(f"audio is too slow or leaked control text: {duration:.3f}s > {maximum:.3f}s")
    return problems
