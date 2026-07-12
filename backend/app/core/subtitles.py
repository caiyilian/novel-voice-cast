"""Deterministic subtitle generation for narrated novels.

The module deliberately has no model or network dependency.  It turns the same
novel entries and WAV files used by the TTS pipeline into SRT cues, keeping each
visible line short and preferring Chinese punctuation as a wrapping boundary.
"""

from __future__ import annotations

import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .parser import parse as parse_novel
from .timeline import (
    GAP_CHAPTER as DEFAULT_GAP_CHAPTER_MS,
    GAP_DIALOGUE as DEFAULT_GAP_DIALOGUE_MS,
    GAP_PARAGRAPH as DEFAULT_GAP_PARAGRAPH_MS,
    gap_between_segments,
)


DEFAULT_MAX_CHARS = 16
DEFAULT_MAX_LINES = 2
# Keep sentence-ending punctuation on the preceding line.  Commas and colons
# are included because the source sentences are often longer than 16 chars.
STRONG_BREAK_CHARS = frozenset("。！？!?…")
MEDIUM_BREAK_CHARS = frozenset("；;：:")
WEAK_BREAK_CHARS = frozenset("，,、.")
SEMANTIC_BREAK_CHARS = STRONG_BREAK_CHARS | MEDIUM_BREAK_CHARS | WEAK_BREAK_CHARS
CLOSING_CHARS = frozenset("”’」』）》】〕〉〗")
OPENING_CHARS = frozenset("“‘「『（《【〔〈〖")
_WORD_JOINER = "\u2060"


@dataclass(frozen=True)
class SubtitleEntry:
    """One spoken TTS entry, in segment-file order."""

    text: str
    speaker: str = ""
    chapter: str = ""
    source_line: int | None = None
    paragraph: int | None = None


@dataclass(frozen=True)
class SubtitleChunk:
    """A piece of one entry that fits in a single two-line subtitle cue."""

    content: str
    display: str


def _normalise_text(text: str) -> str:
    """Flatten physical source lines while preserving intentional spacing."""

    return " ".join(str(text).replace("\t", " ").splitlines()).strip()


def _entry_value(entry: SubtitleEntry | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def load_subtitle_entries(
    novel_path: str | Path,
    labels_path: str | Path | None = None,
    *,
    label_mode: str = "auto",
    expected_segment_count: int | None = None,
) -> list[SubtitleEntry]:
    """Load subtitle entries in the same order as TTS segment files.

    Two label layouts are supported:

    * one label per source line (the issue #95 attachment format);
    * one label per parsed dialogue (the repository's existing full-novel
      format).  The latter is routed through :mod:`app.core.parser`, exactly as
      the TTS pipeline is.
    """

    novel_file = Path(novel_path)
    text = novel_file.read_text(encoding="utf-8-sig")
    source_lines = text.splitlines()

    labels: list[str] = []
    if labels_path is not None:
        label_file = Path(labels_path)
        labels = label_file.read_text(encoding="utf-8-sig").splitlines()

    if label_mode not in {"auto", "line", "parsed-line", "dialogue"}:
        raise ValueError(f"unsupported subtitle label mode: {label_mode}")

    def clean_label(value: str) -> str:
        label = value.strip()
        return "" if label == "非人物发声" else label

    line_entries: list[SubtitleEntry] | None = None
    line_labels_by_number: dict[int, str] = {}
    if labels and len(labels) == len(source_lines):
        line_labels_by_number = {
            index + 1: clean_label(label) for index, label in enumerate(labels)
        }
        line_entries = [
            SubtitleEntry(
                text=line.strip(),
                speaker=line_labels_by_number[index + 1],
                source_line=index + 1,
            )
            for index, line in enumerate(source_lines)
            if line.strip()
        ]

    nonempty_lines = [
        (index + 1, line.strip()) for index, line in enumerate(source_lines) if line.strip()
    ]
    if line_entries is None and labels and len(labels) == len(nonempty_lines):
        line_labels_by_number = {
            line_number: clean_label(labels[index])
            for index, (line_number, _) in enumerate(nonempty_lines)
        }
        line_entries = [
            SubtitleEntry(
                text=line,
                speaker=line_labels_by_number[line_number],
                source_line=line_number,
            )
            for index, (line_number, line) in enumerate(nonempty_lines)
        ]

    def entries_from_dialogues(
        dialogues: Sequence[Mapping[str, Any]],
        *,
        speakers_by_line: Mapping[int, str] | None = None,
    ) -> list[SubtitleEntry]:
        entries: list[SubtitleEntry] = []
        for dialogue in dialogues:
            dialogue_text = _normalise_text(dialogue.get("text", ""))
            if not dialogue_text:
                continue
            source_line = _optional_int(dialogue.get("line"))
            speaker = str(dialogue.get("speaker", "") or "").strip()
            if speakers_by_line is not None and source_line is not None:
                line_speaker = speakers_by_line.get(source_line)
                if line_speaker is not None:
                    speaker = line_speaker
            entries.append(
                SubtitleEntry(
                    text=dialogue_text,
                    speaker=speaker,
                    chapter=str(dialogue.get("chapter", "") or "").strip(),
                    source_line=source_line,
                )
            )
        return entries

    dialogue_entries = entries_from_dialogues(parse_novel(text, labels or None)[0])
    parsed_line_entries: list[SubtitleEntry] | None = None
    if line_labels_by_number:
        parsed_line_entries = entries_from_dialogues(
            parse_novel(text, None)[0],
            speakers_by_line=line_labels_by_number,
        )

    candidates = {
        "line": line_entries,
        "parsed-line": parsed_line_entries,
        "dialogue": dialogue_entries,
    }
    if label_mode != "auto":
        selected = candidates[label_mode]
        if selected is None:
            raise ValueError(f"labels are not compatible with {label_mode!r} mode")
        return selected

    if expected_segment_count is not None:
        matching = {
            name: entries
            for name, entries in candidates.items()
            if entries is not None and len(entries) == expected_segment_count
        }
        # A literal line mapping is unsafe in auto mode when the project parser
        # skips structural headings.  Equal line/WAV counts can otherwise hide
        # a shifted transcript (the issue attachment has exactly this defect).
        # Intentional line-by-line TTS inputs can still select ``line`` mode.
        if (
            "line" in matching
            and parsed_line_entries is not None
            and len(parsed_line_entries) != len(line_entries or [])
        ):
            matching.pop("line")
        signatures = {
            name: tuple(
                (entry.text, entry.speaker, entry.source_line) for entry in entries
            )
            for name, entries in matching.items()
        }
        if len(set(signatures.values())) > 1:
            raise ValueError(
                "subtitle label layout is ambiguous; select --subtitle-label-mode "
                + ", ".join(sorted(matching))
            )
        # Equivalent candidates are safe; prefer parser-backed layouts.
        for preferred in ("parsed-line", "dialogue", "line"):
            if preferred in matching:
                return matching[preferred]
        counts = ", ".join(
            f"{name}={len(entries)}"
            for name, entries in candidates.items()
            if entries is not None
        )
        raise ValueError(
            f"no subtitle label interpretation matches {expected_segment_count} WAV files "
            f"({counts})"
        )

    if line_entries is not None:
        return line_entries
    return dialogue_entries


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _semantic_break_positions(text: str, limit: int | None = None) -> list[int]:
    """Return legal semantic break offsets at or before ``limit``."""

    positions: list[int] = []
    capped = len(text) if limit is None else min(len(text), limit)
    index = 0
    while index < capped:
        char = text[index]
        if char in SEMANTIC_BREAK_CHARS or char.isspace():
            end = index + 1
            if char in SEMANTIC_BREAK_CHARS:
                # Scan the complete punctuation token even when it crosses the
                # width limit.  A partial token is not a legal boundary.
                while end < len(text) and text[end] in SEMANTIC_BREAK_CHARS:
                    end += 1
                while end < len(text) and text[end] in CLOSING_CHARS:
                    end += 1
            else:
                while end < len(text) and text[end].isspace():
                    end += 1
            if end <= capped:
                positions.append(end)
            index = end
        else:
            index += 1
    return positions


def _break_rank(text: str, position: int) -> int:
    index = position - 1
    while index >= 0 and (text[index] in CLOSING_CHARS or text[index].isspace()):
        index -= 1
    if index < 0:
        return 3
    if text[index] in STRONG_BREAK_CHARS:
        return 0
    if text[index] in MEDIUM_BREAK_CHARS:
        return 1
    if text[index] in WEAK_BREAK_CHARS:
        return 2
    return 3


def _is_unsafe_boundary(text: str, position: int) -> bool:
    if position <= 0 or position >= len(text):
        return False
    previous = text[position - 1]
    following = text[position]
    previous_visible_index = position - 1
    while previous_visible_index >= 0 and text[previous_visible_index].isspace():
        previous_visible_index -= 1
    following_visible_index = position
    while following_visible_index < len(text) and text[following_visible_index].isspace():
        following_visible_index += 1
    previous_visible = text[previous_visible_index] if previous_visible_index >= 0 else ""
    following_visible = (
        text[following_visible_index] if following_visible_index < len(text) else ""
    )
    previous_codepoint = ord(previous) if previous else 0
    following_codepoint = ord(following) if following else 0
    regional_indicator_pair = (
        0x1F1E6 <= previous_codepoint <= 0x1F1FF
        and 0x1F1E6 <= following_codepoint <= 0x1F1FF
    )
    return (
        following_visible in CLOSING_CHARS
        or following_visible in SEMANTIC_BREAK_CHARS
        or previous_visible in OPENING_CHARS
        or (previous == following and following in "…—")
        or previous == "\u200d"
        or following == "\u200d"
        or unicodedata.category(following) in {"Mn", "Mc", "Me"}
        or 0x1F3FB <= following_codepoint <= 0x1F3FF
        or regional_indicator_pair
    )


def _safe_hard_break(text: str, position: int, lower: int, upper: int) -> int:
    """Avoid splitting punctuation or a grapheme when a nearby break exists."""

    position = min(max(position, lower), upper)
    candidates = [
        candidate
        for candidate in range(lower, upper + 1)
        if not _is_unsafe_boundary(text, candidate)
    ]
    if not candidates:
        return position
    # When two positions are equally balanced, keep the first line slightly
    # shorter.  Chinese subtitle layout conventionally prefers the lower line
    # to carry the extra character (for example 8+9 rather than 9+8).
    return min(candidates, key=lambda candidate: (abs(candidate - position), candidate))


def _choose_break(text: str, lower: int, upper: int, target: int) -> int:
    candidates = [
        position
        for position in _semantic_break_positions(text, upper)
        if lower <= position <= upper
    ]
    if candidates:
        # Balance first, then prefer stronger punctuation and a slightly
        # earlier boundary so an odd extra character lands on the lower line.
        return min(
            candidates,
            key=lambda position: (
                abs(position - target),
                _break_rank(text, position),
                position,
            ),
        )
    return _safe_hard_break(text, target, lower, upper)


def _partition_text(
    text: str,
    part_count: int,
    capacity: int,
    *,
    first_part_minimum: int = 1,
) -> list[str]:
    """Partition into a fixed count of balanced, capacity-limited spans."""

    if part_count <= 0 or capacity <= 0:
        raise ValueError("part_count and capacity must be positive")
    if not text:
        return []

    parts: list[str] = []
    remaining = text
    for part_index in range(part_count):
        parts_left = part_count - part_index
        if parts_left == 1:
            take = len(remaining)
        else:
            lower = max(1, len(remaining) - capacity * (parts_left - 1))
            if part_index == 0:
                lower = max(lower, first_part_minimum)
            upper = min(capacity, len(remaining) - (parts_left - 1))
            if lower > upper:
                raise ValueError("cannot partition subtitle within the requested limits")
            target = len(remaining) // parts_left
            take = _choose_break(remaining, lower, upper, target)
        parts.append(remaining[:take])
        remaining = remaining[take:]
    if remaining:
        raise ValueError("subtitle partition did not consume all source text")
    return parts


def _layout_display(
    display: str,
    prefix_length: int,
    max_chars: int,
    max_lines: int,
) -> tuple[list[str], int, int]:
    """Return lines plus hard-break count and semantic-boundary rank."""

    line_count = (len(display) + max_chars - 1) // max_chars
    if line_count > max_lines:
        raise ValueError("subtitle cue exceeds the configured line count")
    first_line_minimum = (
        prefix_length + 1
        if line_count > 1 and prefix_length < max_chars
        else max(1, prefix_length)
    )
    lines = _partition_text(
        display,
        line_count,
        max_chars,
        first_part_minimum=first_line_minimum,
    )

    semantic_positions = set(_semantic_break_positions(display))
    meaningful_positions = {
        position for position in semantic_positions if position > prefix_length
    }
    hard_breaks = 0
    boundary_rank = 0
    offset = 0
    for line in lines[:-1]:
        offset += len(line)
        if _is_unsafe_boundary(display, offset):
            raise ValueError("subtitle line would split paired punctuation")
        if offset in semantic_positions:
            boundary_rank += _break_rank(display, offset)
        elif meaningful_positions:
            hard_breaks += 1
    return lines, hard_breaks, boundary_rank


def _split_content_for_cues(
    text: str,
    prefix: str,
    max_chars: int,
    max_lines: int,
) -> list[tuple[str, list[str]]]:
    """Find balanced cues while prioritizing semantic over hard breaks."""

    cue_capacity = max_chars * max_lines - len(prefix)
    if cue_capacity <= 0:
        raise ValueError("speaker prefix leaves no room in a subtitle cue")

    length = len(text)
    semantic_positions = set(_semantic_break_positions(text))
    # A hard cue boundary is more disruptive than a hard line wrap.  The small
    # per-cue cost prevents unpunctuated text from becoming many one-line cues,
    # while still allowing an extra cue when that removes a mid-sentence wrap.
    best_cost: list[tuple[int, int, int, int] | None] = [None] * (length + 1)
    best_end: list[int | None] = [None] * (length + 1)
    best_lines: list[list[str] | None] = [None] * (length + 1)
    best_cost[length] = (0, 0, 0, 0)

    for start in range(length - 1, -1, -1):
        upper = min(length, start + cue_capacity)
        for end in range(start + 1, upper + 1):
            if _is_unsafe_boundary(text, end):
                continue
            following_cost = best_cost[end]
            if following_cost is None:
                continue
            content = text[start:end]
            display = prefix + content
            try:
                lines, line_hard_breaks, line_rank = _layout_display(
                    display,
                    len(prefix),
                    max_chars,
                    max_lines,
                )
            except ValueError:
                continue
            cue_hard_break = int(end < length and end not in semantic_positions)
            cue_rank = _break_rank(text, end) if end < length and not cue_hard_break else 0
            slack = cue_capacity - len(content)
            candidate_cost = (
                line_hard_breaks * 2 + cue_hard_break * 3 + 1 + following_cost[0],
                1 + following_cost[1],
                slack * slack + following_cost[2],
                line_rank + cue_rank + following_cost[3],
            )
            if best_cost[start] is None or candidate_cost < best_cost[start]:
                best_cost[start] = candidate_cost
                best_end[start] = end
                best_lines[start] = lines

    if best_cost[0] is None:
        raise ValueError("cannot split subtitle within the requested limits")

    result: list[tuple[str, list[str]]] = []
    start = 0
    while start < length:
        end = best_end[start]
        lines = best_lines[start]
        if end is None or lines is None or end <= start:
            raise ValueError("subtitle cue splitter made no progress")
        result.append((text[start:end], lines))
        start = end
    return result


def split_subtitle_chunks(
    text: str,
    speaker: str = "",
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[SubtitleChunk]:
    """Split text into display-ready cues.

    Each cue contains at most ``max_lines`` lines and every visible line is at
    most ``max_chars`` code points.  Punctuation is preferred as a boundary;
    a hard split is used only when no semantic boundary exists within the
    allowed width.  The speaker prefix is repeated for every cue so a cue is
    always understandable on its own.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

    normalised = _normalise_text(text)
    if not normalised:
        return []

    clean_speaker = _normalise_text(speaker)
    prefix = f"[{clean_speaker}] " if clean_speaker else ""
    if len(prefix) > max_chars:
        raise ValueError(
            f"speaker prefix {prefix!r} exceeds a {max_chars}-character line"
        )

    chunks: list[SubtitleChunk] = []
    for content, lines in _split_content_for_cues(
        normalised,
        prefix,
        max_chars,
        max_lines,
    ):
        chunks.append(
            SubtitleChunk(
                content=content,
                display="\n".join(lines),
            )
        )
    return chunks


def split_subtitle_text(
    text: str,
    speaker: str = "",
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[str]:
    """Convenience wrapper returning only the visible cue strings."""

    return [
        chunk.display
        for chunk in split_subtitle_chunks(
            text,
            speaker,
            max_chars=max_chars,
            max_lines=max_lines,
        )
    ]


def get_wav_duration_ms(wav_path: str | Path) -> int:
    """Read a WAV duration without decoding the whole audio file."""

    path = Path(wav_path)
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                raise ValueError(f"invalid WAV sample rate: {frame_rate}")
            return int(round(wav_file.getnframes() * 1000 / frame_rate))
    except (wave.Error, EOFError):
        # Some valid WAV variants are unsupported by Python's wave module.
        # pydub/ffmpeg is already a project dependency and handles those files.
        try:
            from pydub import AudioSegment

            return len(AudioSegment.from_file(str(path)))
        except Exception as exc:  # pragma: no cover - only unusual WAV codecs
            raise ValueError(f"cannot read WAV duration: {path}") from exc


def discover_segment_files(segments_dir: str | Path) -> list[Path]:
    """Return WAV files in numeric segment order."""

    directory = Path(segments_dir)
    files = list(directory.glob("*.wav"))

    def key(path: Path) -> tuple[int, int | str]:
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.casefold())

    return sorted(files, key=key)


def build_segment_timestamps(
    entries: Sequence[SubtitleEntry | Mapping[str, Any]],
    segments_dir: str | Path,
    *,
    gap_dialogue_ms: int = DEFAULT_GAP_DIALOGUE_MS,
    gap_paragraph_ms: int = DEFAULT_GAP_PARAGRAPH_MS,
    gap_chapter_ms: int = DEFAULT_GAP_CHAPTER_MS,
) -> list[dict[str, Any]]:
    """Compute exact start/end timestamps from ordered WAV files.

    A count mismatch is fatal: continuing would attach correct words to the
    wrong audio, which is worse than failing before the six-hour render starts.
    """

    files = discover_segment_files(segments_dir)
    if not files:
        raise FileNotFoundError(f"no WAV segments found in {Path(segments_dir)}")
    if len(files) != len(entries):
        raise ValueError(
            "subtitle/audio count mismatch: "
            f"{len(entries)} text entries but {len(files)} WAV segments"
        )
    numeric_indices = [int(path.stem) for path in files if path.stem.isdigit()]
    if len(numeric_indices) != len(files) or numeric_indices != list(range(len(files))):
        names = ", ".join(path.name for path in files[:5])
        raise ValueError(
            "WAV segment names must be a contiguous zero-based sequence "
            f"(00000.wav, 00001.wav, ...); found {names}"
        )

    timestamps: list[dict[str, Any]] = []
    offset_ms = 0
    for index, (entry, wav_path) in enumerate(zip(entries, files)):
        duration_ms = get_wav_duration_ms(wav_path)
        if duration_ms <= 0:
            raise ValueError(f"empty audio segment: {wav_path}")
        end_ms = offset_ms + duration_ms
        source_line = _entry_value(entry, "source_line", index + 1)
        timestamps.append(
            {
                "line": index + 1 if source_line is None else source_line,
                "segment_file": wav_path.name,
                "duration_ms": duration_ms,
                "start_ms": offset_ms,
                "end_ms": end_ms,
            }
        )
        offset_ms = end_ms
        if index + 1 < len(entries):
            offset_ms += gap_between_segments(
                entry,
                entries[index + 1],
                gap_dialogue=gap_dialogue_ms,
                gap_paragraph=gap_paragraph_ms,
                gap_chapter=gap_chapter_ms,
            )
    return timestamps


def _timestamp_value(timestamp: Mapping[str, Any] | Any, key: str) -> int:
    value = timestamp.get(key) if isinstance(timestamp, Mapping) else getattr(timestamp, key)
    return int(value)


def _allocate_chunk_times(
    start_ms: int,
    end_ms: int,
    chunks: Sequence[SubtitleChunk],
) -> list[tuple[int, int]]:
    if end_ms <= start_ms:
        raise ValueError(f"invalid subtitle timestamp: {start_ms}..{end_ms}")
    if not chunks:
        return []

    duration_ms = end_ms - start_ms
    if duration_ms < len(chunks):
        raise ValueError("subtitle duration is too short for the number of chunks")

    weights = [max(1, len(chunk.content.strip())) for chunk in chunks]
    total_weight = sum(weights)
    result: list[tuple[int, int]] = []
    cumulative = 0
    cue_start = start_ms
    for index, weight in enumerate(weights):
        cumulative += weight
        if index == len(weights) - 1:
            cue_end = end_ms
        else:
            proportional_end = start_ms + (
                duration_ms * cumulative + total_weight // 2
            ) // total_weight
            latest_end = end_ms - (len(weights) - index - 1)
            cue_end = min(max(cue_start + 1, proportional_end), latest_end)
        result.append((cue_start, cue_end))
        cue_start = cue_end
    return result


def ms_to_srt(ms: int) -> str:
    """Format integer milliseconds as an SRT timestamp."""

    if ms < 0:
        raise ValueError("SRT timestamp cannot be negative")
    seconds, milliseconds = divmod(int(ms), 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def escape_srt_text(text: str) -> str:
    """Keep literal novel text from becoming HTML/ASS markup in FFmpeg.

    FFmpeg converts SubRip through its HTML-to-ASS parser.  A zero-width word
    joiner makes a ``<tag>`` or ``&entity;`` syntactically invalid without
    changing how it is displayed; the same character breaks ASS commands such
    as ``\\N``.  The left-brace escape mirrors FFmpeg's own plain-text ASS
    escaping strategy.
    This is applied *after* line layout, so safety characters do not count
    against the 16-character source limit.
    """

    escaped = text.replace("&", "&" + _WORD_JOINER)
    escaped = escaped.replace("\\", "\\" + _WORD_JOINER)
    escaped = escaped.replace("{", r"\{{}")
    return escaped.replace("<", "<" + _WORD_JOINER)


def build_srt(
    entries: Sequence[SubtitleEntry | Mapping[str, Any]],
    timestamps: Sequence[Mapping[str, Any] | Any],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> str:
    """Build SRT text and proportionally time any split cues."""

    if len(entries) != len(timestamps):
        raise ValueError(
            f"subtitle entry/timestamp mismatch: {len(entries)} entries, "
            f"{len(timestamps)} timestamps"
        )

    blocks: list[str] = []
    cue_index = 1
    for entry, timestamp in zip(entries, timestamps):
        text = str(_entry_value(entry, "text", "") or "")
        speaker = str(_entry_value(entry, "speaker", "") or "")
        chunks = split_subtitle_chunks(
            text,
            speaker,
            max_chars=max_chars,
            max_lines=max_lines,
        )
        if not chunks:
            continue

        start_ms = _timestamp_value(timestamp, "start_ms")
        end_ms = _timestamp_value(timestamp, "end_ms")
        for chunk, (chunk_start, chunk_end) in zip(
            chunks,
            _allocate_chunk_times(start_ms, end_ms, chunks),
        ):
            blocks.append(
                f"{cue_index}\n"
                f"{ms_to_srt(chunk_start)} --> {ms_to_srt(chunk_end)}\n"
                f"{escape_srt_text(chunk.display)}"
            )
            cue_index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(
    path: str | Path,
    entries: Sequence[SubtitleEntry | Mapping[str, Any]],
    timestamps: Sequence[Mapping[str, Any] | Any],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> Path:
    """Write a UTF-8 SRT file and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_srt(entries, timestamps, max_chars=max_chars, max_lines=max_lines),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def validate_visible_cues(
    cues: Iterable[str],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> None:
    """Raise when a generated cue violates the line-count/width contract."""

    for cue in cues:
        lines = cue.splitlines()
        if not 1 <= len(lines) <= max_lines:
            raise ValueError(f"subtitle cue has {len(lines)} lines: {cue!r}")
        if any(len(line) > max_chars for line in lines):
            raise ValueError(f"subtitle cue exceeds {max_chars} characters: {cue!r}")
