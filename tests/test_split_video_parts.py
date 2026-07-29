import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split_video_parts import (  # noqa: E402
    _parse_input,
    BoundaryCandidate,
    build_boundary_candidates,
    build_parts,
    choose_boundaries,
    detect_headings,
    format_clock,
    shared_keyframes,
)


def candidate(time_minutes, score=10.0, label=""):
    return BoundaryCandidate(
        time_ms=int(time_minutes * 60_000),
        score=score,
        reasons=(label or "pause",),
        label=label,
        previous_line=1,
        next_line=2,
        previous_text="before",
        next_text="after",
        silence_ms=600,
    )


def test_format_clock_supports_long_form_video():
    assert format_clock(21_665_770) == "06:01:05.770"


def test_detect_headings_supports_acts_finale_and_deduplicates_afterword():
    lines = ["开头", "第一幕", "正文", "第", "二", "幕", "第二幕", "终幕", "后记", "后记"]
    assert detect_headings(lines) == [
        (2, "第一幕"),
        (7, "第二幕"),
        (8, "终幕"),
        (9, "后记"),
    ]


def test_shared_keyframes_requires_all_variants_with_tolerance():
    assert shared_keyframes([[0, 1000, 2000, 3000], [0, 1020, 1980, 4000]], 30) == [
        0,
        1010,
        1990,
    ]


def test_boundary_candidates_only_use_keyframes_inside_speech_gaps():
    lines = ["第一句。", "第二幕", "第二句。", "第三句。"]
    entries = [
        {"line": 1, "text": "第一句。", "speaker": "旁白"},
        {"line": 3, "text": "第二句。", "speaker": "旁白"},
        {"line": 4, "text": "第三句。", "speaker": "旁白"},
    ]
    timestamps = [
        {"start_ms": 0, "end_ms": 1000},
        {"start_ms": 1600, "end_ms": 2600},
        {"start_ms": 2900, "end_ms": 3900},
    ]
    results = build_boundary_candidates(
        lines,
        entries,
        timestamps,
        [1020, 1300, 2750, 2880, 3500],
        headings=[(2, "第二幕")],
        frame_tolerance_ms=50,
    )
    assert [item.time_ms for item in results] == [1300, 2750]
    assert results[0].label == "第二幕"
    assert any("章节边界" in reason for reason in results[0].reasons)


def test_dynamic_plan_is_below_limit_and_prefers_natural_chapter_cuts():
    total = 181 * 60_000
    candidates = [candidate(value) for value in range(30, 181, 5)]
    candidates.extend(
        [
            candidate(58, 100, "第二幕"),
            candidate(119, 100, "第三幕"),
        ]
    )
    candidates = sorted({item.time_ms: item for item in candidates}.values(), key=lambda x: x.time_ms)
    selected = choose_boundaries(
        candidates,
        total,
        max_duration_ms=60 * 60_000,
        safety_margin_ms=30_000,
    )
    parts = build_parts(selected, total)
    assert len(parts) == 4
    assert all(part.duration_ms < 60 * 60_000 for part in parts)
    assert any(item.label == "第二幕" for item in selected)
    assert parts[0].start_ms == 0
    assert parts[-1].end_ms == total


def test_video_below_effective_limit_needs_no_cut():
    selected = choose_boundaries(
        [],
        42 * 60_000,
        max_duration_ms=60 * 60_000,
        safety_margin_ms=30_000,
    )
    parts = build_parts(selected, 42 * 60_000)
    assert selected == []
    assert len(parts) == 1
    assert parts[0].duration_ms == 42 * 60_000


def test_input_variant_name_supports_readable_unicode():
    name, path = _parse_input("横屏版本=output/custom master.mp4")
    assert name == "横屏版本"
    assert path == Path("output/custom master.mp4")
