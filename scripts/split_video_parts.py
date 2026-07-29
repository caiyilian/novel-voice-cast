"""Split finished portrait/landscape videos at shared natural boundaries.

This is an optional post-processing utility.  It is deliberately independent
from ``run_full.py``: the long-form masters stay untouched and the generated
parts are written below a separate output directory.

The splitter combines four signals:

* TTS segment joins and gaps (with configurable frame-time tolerance),
* chapter/act headings in the source novel,
* BGM and illustration scene boundaries,
* keyframes shared by every input video (stream-copy safe).

The selected boundaries are therefore both semantically useful and safe for
lossless ``-c copy`` splitting.  A manifest records the plan and verifies that
every emitted file remains below the platform limit.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.parser import parse  # noqa: E402
from app.core.subtitles import build_segment_timestamps  # noqa: E402


DEFAULT_INPUTS = {
    "portrait": ROOT / "output/illustration_video_local_portrait_7x9_subtitled.mp4",
    "landscape": ROOT / "output/illustration_video_local_landscape_16x9_subtitled.mp4",
}
DEFAULT_OUTPUT_DIR = ROOT / "output/video_parts_under_60min"
DEFAULT_NOVEL = ROOT / "novels/novel.txt"
DEFAULT_LABELS = ROOT / "novels/labels.txt"
DEFAULT_SEGMENTS_DIR = ROOT / "output/segments"
DEFAULT_ILLUSTRATION_PLAN = ROOT / "output/illustration_plan.json"
DEFAULT_BGM_SEGMENTS = ROOT / "backend/data/bgm_segments.json"
DEFAULT_FFMPEG = Path(
    shutil.which("ffmpeg") or "D:/ffmpeg-7.1.1-full_build/bin/ffmpeg.exe"
)
DEFAULT_FFPROBE = Path(
    shutil.which("ffprobe") or "D:/ffmpeg-7.1.1-full_build/bin/ffprobe.exe"
)

HEADING_PATTERN = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百千万零〇○\d]+[章节部集卷篇幕]"
    r"|序章|序幕|前言|楔子|引子|终章|终幕|尾声|后记|番外(?:篇)?|特别篇|完结"
    r"|(?:Chapter|Chap\.?|Part|Volume|Book)\s*\d+"
    r")(?:[\s:：.．\-—]+.*)?$",
    re.IGNORECASE,
)
STRONG_ENDINGS = tuple("。！？!?…」』”’）)】]》〉")


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration_ms: int
    width: int
    height: int
    video_codec: str
    audio_codec: str
    size_bytes: int


@dataclass(frozen=True)
class BoundaryCandidate:
    time_ms: int
    score: float
    reasons: tuple[str, ...]
    label: str
    previous_line: int
    next_line: int
    previous_text: str
    next_text: str
    silence_ms: int


@dataclass(frozen=True)
class PlannedPart:
    number: int
    start_ms: int
    end_ms: int
    duration_ms: int
    start_context: str
    end_context: str
    cut_label: str
    cut_reasons: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_clock(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    total_seconds, millis = divmod(milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _short_text(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {detail[-2000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc


def probe_media(ffprobe: Path, path: Path) -> MediaInfo:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = raw.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise ValueError(f"input must contain video and audio streams: {path}")
    format_data = raw.get("format") or {}
    duration_ms = int(round(float(format_data.get("duration") or 0) * 1000))
    if duration_ms <= 0:
        raise ValueError(f"invalid media duration: {path}")
    return MediaInfo(
        path=str(path.resolve()),
        duration_ms=duration_ms,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        video_codec=str(video.get("codec_name") or ""),
        audio_codec=str(audio.get("codec_name") or ""),
        size_bytes=int(format_data.get("size") or path.stat().st_size),
    )


def probe_keyframes(ffprobe: Path, path: Path) -> list[int]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot read keyframes from {path}: {completed.stderr[-2000:]}")
    result: list[int] = []
    for raw_line in completed.stdout.splitlines():
        value = raw_line.strip().split(",", 1)[0]
        if not value:
            continue
        try:
            result.append(int(round(float(value) * 1000)))
        except ValueError:
            continue
    result = sorted(set(result))
    if not result:
        raise ValueError(f"no video keyframes found: {path}")
    return result


def shared_keyframes(groups: Sequence[Sequence[int]], tolerance_ms: int = 50) -> list[int]:
    """Return timestamps that have a matching keyframe in every video."""

    if not groups:
        return []
    others = [sorted(int(value) for value in group) for group in groups[1:]]
    shared: list[int] = []
    for base in sorted(int(value) for value in groups[0]):
        matches = [base]
        for group in others:
            position = bisect.bisect_left(group, base)
            nearby = group[max(0, position - 1) : position + 1]
            if not nearby:
                break
            match = min(nearby, key=lambda value: abs(value - base))
            if abs(match - base) > tolerance_ms:
                break
            matches.append(match)
        else:
            shared.append(int(round(sum(matches) / len(matches))))
    return sorted(set(shared))


def detect_headings(lines: Sequence[str]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        title = line.strip()
        if not title or not HEADING_PATTERN.fullmatch(title):
            continue
        if headings and headings[-1][1] == title and line_number - headings[-1][0] <= 2:
            continue
        headings.append((line_number, title))
    return headings


def _load_markers(path: Path, key: str) -> list[tuple[int, str]]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get(key) or []
    if not isinstance(raw, list):
        return []
    markers: list[tuple[int, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            line = int(item.get("start_line") or 0)
        except (TypeError, ValueError):
            continue
        if line > 1:
            markers.append((line, _short_text(item.get("title") or key, 50)))
    return markers


def _markers_between(
    markers: Sequence[tuple[int, str]], previous_line: int, next_line: int
) -> list[str]:
    return [title for line, title in markers if previous_line < line <= next_line]


def build_timeline(
    novel_path: Path,
    labels_path: Path,
    segments_dir: Path,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    lines = novel_path.read_text(encoding="utf-8").splitlines()
    labels = (
        [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
        if labels_path.is_file()
        else []
    )
    entries, _ = parse("\n".join(lines), labels)
    timestamps = build_segment_timestamps(entries, segments_dir)
    if len(entries) != len(timestamps):
        raise ValueError("dialogue/timestamp count mismatch")
    return lines, entries, timestamps


def build_boundary_candidates(
    lines: Sequence[str],
    entries: Sequence[Mapping[str, Any]],
    timestamps: Sequence[Mapping[str, Any]],
    keyframes: Sequence[int],
    *,
    headings: Sequence[tuple[int, str]] = (),
    bgm_markers: Sequence[tuple[int, str]] = (),
    illustration_markers: Sequence[tuple[int, str]] = (),
    frame_tolerance_ms: int = 50,
) -> list[BoundaryCandidate]:
    if len(entries) != len(timestamps):
        raise ValueError("entries and timestamps must have equal length")
    keyframes = sorted(int(value) for value in keyframes)
    candidates: dict[int, BoundaryCandidate] = {}
    for index in range(len(entries) - 1):
        current = entries[index]
        following = entries[index + 1]
        current_ts = timestamps[index]
        next_ts = timestamps[index + 1]
        gap_start = int(current_ts["end_ms"])
        gap_end = int(next_ts["start_ms"])
        if gap_end < gap_start:
            continue
        # The composed video's frame clock and the audio timeline can differ
        # by a few milliseconds.  Accept only the configured alignment window
        # around a complete TTS segment join, never an arbitrary point in the
        # middle of a source segment.
        low = gap_start - frame_tolerance_ms
        high = gap_end + frame_tolerance_ms
        left = bisect.bisect_left(keyframes, low)
        right = bisect.bisect_right(keyframes, high)
        nearby = keyframes[left:right]
        if not nearby:
            continue
        midpoint = (gap_start + gap_end) // 2
        cut_time = min(nearby, key=lambda value: abs(value - midpoint))
        previous_line = int(current.get("line") or current_ts.get("line") or index + 1)
        next_line = int(following.get("line") or next_ts.get("line") or index + 2)
        if next_line < previous_line:
            continue

        silence_ms = max(0, gap_end - gap_start)
        reasons: list[str] = [f"语音间隙 {silence_ms}ms"]
        score = min(8.0, silence_ms / 100.0)
        if next_line > previous_line:
            score += 3.0
        if next_line - previous_line > 1:
            score += 8.0
            reasons.append("跨自然段")
            interior = lines[previous_line: max(previous_line, next_line - 1)]
            if any(not line.strip() for line in interior):
                score += 8.0
                reasons.append("原文空行")
        previous_text = _short_text(current.get("text"))
        next_text = _short_text(following.get("text"))
        if previous_text.endswith(STRONG_ENDINGS):
            score += 2.0
            reasons.append("完整句末")
        if current.get("speaker") != following.get("speaker"):
            score += 1.0

        chapter_titles = _markers_between(headings, previous_line, next_line)
        bgm_titles = _markers_between(bgm_markers, previous_line, next_line)
        illustration_titles = _markers_between(
            illustration_markers, previous_line, next_line
        )
        label = ""
        if chapter_titles:
            label = chapter_titles[-1]
            score += 80.0
            reasons.append(f"章节边界：{label}")
        if bgm_titles:
            label = label or bgm_titles[-1]
            score += 18.0
            reasons.append(f"BGM 场景边界：{bgm_titles[-1]}")
        if illustration_titles:
            label = label or illustration_titles[-1]
            score += 12.0
            reasons.append(f"插图边界：{illustration_titles[-1]}")

        candidate = BoundaryCandidate(
            time_ms=cut_time,
            score=round(score, 3),
            reasons=tuple(reasons),
            label=label,
            previous_line=previous_line,
            next_line=next_line,
            previous_text=previous_text,
            next_text=next_text,
            silence_ms=silence_ms,
        )
        existing = candidates.get(cut_time)
        if existing is None or candidate.score > existing.score:
            candidates[cut_time] = candidate
    return sorted(candidates.values(), key=lambda item: item.time_ms)


def choose_boundaries(
    candidates: Sequence[BoundaryCandidate],
    total_duration_ms: int,
    *,
    max_duration_ms: int,
    safety_margin_ms: int,
    minimum_part_ms: int = 5 * 60 * 1000,
) -> list[BoundaryCandidate]:
    """Choose the minimum number of balanced parts using dynamic programming."""

    effective_max = max_duration_ms - safety_margin_ms
    if effective_max <= 0:
        raise ValueError("safety margin must be smaller than maximum duration")
    if total_duration_ms <= effective_max:
        return []
    part_count = math.ceil(total_duration_ms / effective_max)
    ideal = total_duration_ms / part_count
    minimum = min(int(ideal * 0.45), int(minimum_part_ms))

    usable = [
        item
        for item in candidates
        if minimum <= item.time_ms <= total_duration_ms - minimum
    ]
    if len(usable) < part_count - 1:
        raise ValueError(
            f"only {len(usable)} safe boundaries for {part_count} required parts"
        )

    times = [0] + [item.time_ms for item in usable] + [total_duration_ms]
    scores = [0.0] + [item.score for item in usable] + [0.0]
    end_index = len(times) - 1
    previous: list[dict[int, tuple[float, int]]] = [dict() for _ in range(part_count + 1)]
    previous[0][0] = (0.0, -1)

    for parts_used in range(1, part_count + 1):
        target_is_end = parts_used == part_count
        j_values: Iterable[int] = [end_index] if target_is_end else range(1, end_index)
        for j in j_values:
            best: tuple[float, int] | None = None
            for i, (accumulated, _) in previous[parts_used - 1].items():
                duration = times[j] - times[i]
                if duration < minimum or duration > effective_max:
                    continue
                remaining_parts = part_count - parts_used
                remaining_duration = total_duration_ms - times[j]
                if remaining_parts:
                    if not (
                        remaining_parts * minimum
                        <= remaining_duration
                        <= remaining_parts * effective_max
                    ):
                        continue
                elif j != end_index:
                    continue
                deviation_minutes = (duration - ideal) / 60_000.0
                # A strong chapter/scene boundary is worth accepting a less
                # even part length, provided the hard duration constraints
                # remain satisfied.  Platform parts do not need to be equal;
                # they need to end naturally.
                value = accumulated + scores[j] - 0.35 * deviation_minutes**2
                if best is None or value > best[0]:
                    best = (value, i)
            if best is not None:
                previous[parts_used][j] = best

    if end_index not in previous[part_count]:
        raise ValueError(
            "cannot build safe parts below the maximum duration from available keyframes"
        )
    selected_indices: list[int] = []
    current = end_index
    for parts_used in range(part_count, 0, -1):
        predecessor = previous[parts_used][current][1]
        if current != end_index:
            selected_indices.append(current)
        current = predecessor
    selected_indices.reverse()
    return [usable[index - 1] for index in selected_indices]


def build_parts(
    boundaries: Sequence[BoundaryCandidate], total_duration_ms: int
) -> list[PlannedPart]:
    points = [0] + [item.time_ms for item in boundaries] + [total_duration_ms]
    parts: list[PlannedPart] = []
    for index in range(len(points) - 1):
        ending = boundaries[index] if index < len(boundaries) else None
        starting = boundaries[index - 1] if index > 0 else None
        parts.append(
            PlannedPart(
                number=index + 1,
                start_ms=points[index],
                end_ms=points[index + 1],
                duration_ms=points[index + 1] - points[index],
                start_context=starting.next_text if starting else "作品开头",
                end_context=ending.previous_text if ending else "作品结尾",
                cut_label=ending.label if ending else "作品结尾",
                cut_reasons=ending.reasons if ending else ("完整作品结尾",),
            )
        )
    return parts


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must use NAME=PATH")
    name, path = value.split("=", 1)
    # ``\w`` is Unicode-aware, so user-facing variant names such as “横屏”
    # remain readable while filesystem separators and punctuation stay out.
    name = re.sub(r"[^\w-]", "_", name.strip(), flags=re.UNICODE)
    if not name or not path.strip():
        raise argparse.ArgumentTypeError("input must use non-empty NAME=PATH")
    return name, Path(path.strip())


def _clear_existing_parts(directory: Path) -> None:
    for path in directory.glob("part_*.mp4"):
        if path.is_file():
            path.unlink()


def split_one_video(
    ffmpeg: Path,
    input_path: Path,
    output_dir: Path,
    boundaries: Sequence[BoundaryCandidate],
    *,
    force: bool,
    segment_time_delta_seconds: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("part_*.mp4"))
    if existing and not force:
        raise FileExistsError(
            f"split outputs already exist in {output_dir}; pass --force to replace them"
        )
    if force:
        _clear_existing_parts(output_dir)
    common_command = [
        str(ffmpeg),
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
    ]
    if not boundaries:
        # A source that already fits below the limit still gets the same
        # predictable post-processing layout.  The master remains untouched.
        completed = subprocess.run(
            common_command
            + ["-movflags", "+faststart", str(output_dir / "part_01.mp4")],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg stream copy failed ({completed.returncode}): {input_path}"
            )
        return sorted(output_dir.glob("part_*.mp4"))

    pattern = output_dir / "part_%02d.mp4"
    command = common_command + [
        "-f",
        "segment",
        "-segment_times",
        ",".join(f"{item.time_ms / 1000:.3f}" for item in boundaries),
        "-segment_time_delta",
        f"{segment_time_delta_seconds:.3f}",
        "-reset_timestamps",
        "1",
        "-segment_start_number",
        "1",
        "-segment_format",
        "mp4",
        "-segment_format_options",
        "movflags=+faststart",
        str(pattern),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed ({completed.returncode}): {input_path}")
    return sorted(output_dir.glob("part_*.mp4"))


def validate_outputs(
    ffprobe: Path,
    paths: Sequence[Path],
    *,
    expected_count: int,
    max_duration_ms: int,
) -> list[dict[str, Any]]:
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} parts, found {len(paths)}")
    results: list[dict[str, Any]] = []
    for path in paths:
        info = probe_media(ffprobe, path)
        if info.duration_ms >= max_duration_ms:
            raise ValueError(
                f"part exceeds strict limit: {path} = {format_clock(info.duration_ms)}"
            )
        results.append(
            {
                "path": info.path,
                "duration_ms": info.duration_ms,
                "duration": format_clock(info.duration_ms),
                "size_bytes": info.size_bytes,
                "width": info.width,
                "height": info.height,
                "video_codec": info.video_codec,
                "audio_codec": info.audio_codec,
            }
        )
    return results


def _write_summary(
    path: Path,
    parts: Sequence[PlannedPart],
    outputs: Mapping[str, Any],
    *,
    max_duration_ms: int,
) -> None:
    limit_minutes = max_duration_ms / 60_000
    limit_text = f"{limit_minutes:g}"
    lines = [f"小于 {limit_text} 分钟的视频分段", "=" * 32, ""]
    for part in parts:
        lines.extend(
            [
                f"第 {part.number:02d} 部分  {format_clock(part.start_ms)} -> {format_clock(part.end_ms)}",
                f"计划时长：{format_clock(part.duration_ms)}",
                f"自然切点：{part.cut_label}",
                f"结尾内容：{part.end_context}",
            ]
        )
        for name, records in outputs.items():
            record = records[part.number - 1]
            lines.append(f"{name}: {record['path']}  ({record['duration']})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=_parse_input,
        help="Video input as NAME=PATH; repeat for variants. Defaults to portrait and landscape.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--novel", type=Path, default=DEFAULT_NOVEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--segments-dir", type=Path, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument("--illustration-plan", type=Path, default=DEFAULT_ILLUSTRATION_PLAN)
    parser.add_argument("--bgm-segments", type=Path, default=DEFAULT_BGM_SEGMENTS)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument(
        "--safety-seconds",
        type=float,
        default=30.0,
        help="Keep every planned part this far below the strict platform limit.",
    )
    parser.add_argument("--keyframe-tolerance-ms", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_minutes <= 0:
        raise SystemExit("--max-minutes must be positive")
    if args.safety_seconds < 0:
        raise SystemExit("--safety-seconds must be non-negative")
    inputs = dict(args.input or DEFAULT_INPUTS.items())
    inputs = {name: path.resolve() for name, path in inputs.items()}
    ffmpeg = args.ffmpeg.resolve()
    ffprobe = args.ffprobe.resolve()
    for executable in (ffmpeg, ffprobe):
        if not executable.is_file():
            raise FileNotFoundError(executable)

    print(f"Inputs: {len(inputs)}")
    media = {name: probe_media(ffprobe, path) for name, path in inputs.items()}
    durations = [item.duration_ms for item in media.values()]
    if max(durations) - min(durations) > 1000:
        raise ValueError(f"input video durations differ by more than one second: {durations}")
    total_duration_ms = min(durations)
    print(f"Duration: {format_clock(total_duration_ms)}")

    keyframe_groups: list[list[int]] = []
    for name, path in inputs.items():
        print(f"Reading keyframes: {name}")
        keyframe_groups.append(probe_keyframes(ffprobe, path))
    common_keyframes = shared_keyframes(
        keyframe_groups, tolerance_ms=args.keyframe_tolerance_ms
    )
    print(f"Shared keyframes: {len(common_keyframes)}")

    lines, entries, timestamps = build_timeline(
        args.novel.resolve(), args.labels.resolve(), args.segments_dir.resolve()
    )
    timeline_duration_ms = int(timestamps[-1]["end_ms"])
    if abs(timeline_duration_ms - total_duration_ms) > 1000:
        raise ValueError(
            "TTS timeline does not match video duration: "
            f"timeline={format_clock(timeline_duration_ms)}, "
            f"video={format_clock(total_duration_ms)}"
        )
    headings = detect_headings(lines)
    bgm_markers = _load_markers(args.bgm_segments.resolve(), "segments")
    illustration_markers = _load_markers(
        args.illustration_plan.resolve(), "illustrations"
    )
    candidates = build_boundary_candidates(
        lines,
        entries,
        timestamps,
        common_keyframes,
        headings=headings,
        bgm_markers=bgm_markers,
        illustration_markers=illustration_markers,
        frame_tolerance_ms=args.keyframe_tolerance_ms,
    )
    print(f"Safe natural candidates: {len(candidates)}")
    max_duration_ms = int(round(args.max_minutes * 60_000))
    safety_margin_ms = int(round(args.safety_seconds * 1000))
    boundaries = choose_boundaries(
        candidates,
        total_duration_ms,
        max_duration_ms=max_duration_ms,
        safety_margin_ms=safety_margin_ms,
    )
    parts = build_parts(boundaries, total_duration_ms)
    print(f"Parts: {len(parts)}")
    for part in parts:
        print(
            f"  {part.number:02d}: {format_clock(part.start_ms)} -> "
            f"{format_clock(part.end_ms)} ({format_clock(part.duration_ms)}) "
            f"[{part.cut_label}]"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_payload: dict[str, Any] = {
        "version": 1,
        "created_at": utc_now(),
        "mode": "stream-copy-shared-keyframe-natural-boundaries",
        "strict_max_duration_ms": max_duration_ms,
        "safety_margin_ms": safety_margin_ms,
        "source_duration_ms": total_duration_ms,
        "source_duration": format_clock(total_duration_ms),
        "inputs": {name: asdict(info) for name, info in media.items()},
        "shared_keyframe_count": len(common_keyframes),
        "safe_candidate_count": len(candidates),
        "headings": [{"line": line, "title": title} for line, title in headings],
        "parts": [asdict(part) for part in parts],
        "outputs": {},
    }
    plan_path = output_dir / "split_manifest.json"
    if args.plan_only:
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Plan written: {plan_path}")
        return 0

    outputs: dict[str, list[dict[str, Any]]] = {}
    for name, input_path in inputs.items():
        print(f"Splitting {name} with stream copy...")
        paths = split_one_video(
            ffmpeg,
            input_path,
            output_dir / name,
            boundaries,
            force=args.force,
            segment_time_delta_seconds=max(0.021, args.keyframe_tolerance_ms / 1000.0),
        )
        outputs[name] = validate_outputs(
            ffprobe,
            paths,
            expected_count=len(parts),
            max_duration_ms=max_duration_ms,
        )
    plan_payload["outputs"] = outputs
    plan_payload["completed_at"] = utc_now()
    plan_path.write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary(
        output_dir / "分段清单.txt",
        parts,
        outputs,
        max_duration_ms=max_duration_ms,
    )
    print(f"Manifest: {plan_path}")
    print(f"Summary: {output_dir / '分段清单.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
