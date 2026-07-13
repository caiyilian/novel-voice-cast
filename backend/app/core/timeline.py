"""Shared timing rules for TTS splicing, BGM mixing, and subtitles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


GAP_DIALOGUE = 300
GAP_PARAGRAPH = 1000
GAP_CHAPTER = 2000
FADE_DURATION = 50


def _value(segment: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(segment, Mapping):
        return segment.get(name, default)
    return getattr(segment, name, default)


def gap_between_segments(
    current: Mapping[str, Any] | Any,
    following: Mapping[str, Any] | Any,
    *,
    gap_dialogue: int = GAP_DIALOGUE,
    gap_paragraph: int = GAP_PARAGRAPH,
    gap_chapter: int = GAP_CHAPTER,
) -> int:
    """Return the silence inserted between two ordered TTS segments.

    Chapter changes take precedence over explicit paragraph changes.  Existing
    segment manifests only carry ``chapter``, so they retain their historical
    300/2000 ms behavior; future manifests can opt into the documented 1000 ms
    paragraph gap by providing a stable ``paragraph`` identifier.
    """

    current_chapter = str(_value(current, "chapter", "") or "")
    following_chapter = str(_value(following, "chapter", "") or "")
    if current_chapter != following_chapter:
        return gap_chapter

    current_paragraph = _value(current, "paragraph")
    following_paragraph = _value(following, "paragraph")
    if (
        current_paragraph is not None
        and following_paragraph is not None
        and current_paragraph != following_paragraph
    ):
        return gap_paragraph
    return gap_dialogue


def build_contiguous_intervals(
    group_ids: Sequence[Any],
    starts_ms: Sequence[int],
    ends_ms: Sequence[int],
) -> list[dict[str, Any]]:
    """Group an ordered segment timeline without dropping inter-group gaps."""

    if not (len(group_ids) == len(starts_ms) == len(ends_ms)):
        raise ValueError("group/start/end counts must match")
    if not group_ids:
        return []

    for index, (start, end) in enumerate(zip(starts_ms, ends_ms)):
        if end <= start:
            raise ValueError(f"invalid interval at segment {index}: {start}..{end}")
        if index and start < starts_ms[index - 1]:
            raise ValueError("segment starts must be ordered")

    intervals: list[dict[str, Any]] = []
    current_group = group_ids[0]
    current_start = int(starts_ms[0])
    first_index = 0
    for index in range(1, len(group_ids)):
        if group_ids[index] == current_group:
            continue
        intervals.append(
            {
                "group_id": current_group,
                "start_ms": current_start,
                # Use the following segment's start, not the preceding
                # segment's end, so the splice silence remains in the output.
                "end_ms": int(starts_ms[index]),
                "first_index": first_index,
                "last_index": index - 1,
            }
        )
        current_group = group_ids[index]
        current_start = int(starts_ms[index])
        first_index = index

    intervals.append(
        {
            "group_id": current_group,
            "start_ms": current_start,
            "end_ms": int(ends_ms[-1]),
            "first_index": first_index,
            "last_index": len(group_ids) - 1,
        }
    )
    return intervals
