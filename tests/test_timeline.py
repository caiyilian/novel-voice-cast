import sys
from pathlib import Path

import pytest
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.timeline import build_contiguous_intervals, gap_between_segments  # noqa: E402
from app.core.splicer import AudioSplicer  # noqa: E402


def test_dialogue_gap_is_default():
    assert gap_between_segments({"chapter": "第一章"}, {"chapter": "第一章"}) == 300


def test_explicit_paragraph_change_uses_paragraph_gap():
    assert gap_between_segments(
        {"chapter": "第一章", "paragraph": 1},
        {"chapter": "第一章", "paragraph": 2},
    ) == 1000


def test_chapter_change_takes_precedence():
    assert gap_between_segments(
        {"chapter": "第一章", "paragraph": 1},
        {"chapter": "第二章", "paragraph": 1},
    ) == 2000


def test_gap_overrides_are_shared_by_callers():
    assert gap_between_segments(
        {"chapter": ""},
        {"chapter": ""},
        gap_dialogue=123,
        gap_paragraph=456,
        gap_chapter=789,
    ) == 123


def test_grouped_intervals_preserve_silence_at_boundaries():
    intervals = build_contiguous_intervals(
        group_ids=[0, 0, 1, 1],
        starts_ms=[0, 400, 900, 1600],
        ends_ms=[100, 600, 1200, 2000],
    )

    assert intervals == [
        {
            "group_id": 0,
            "start_ms": 0,
            "end_ms": 900,
            "first_index": 0,
            "last_index": 1,
        },
        {
            "group_id": 1,
            "start_ms": 900,
            "end_ms": 2000,
            "first_index": 2,
            "last_index": 3,
        },
    ]
    assert intervals[0]["end_ms"] == intervals[1]["start_ms"]


def test_audio_splicer_uses_shared_gap_rules(tmp_path):
    segments = []
    for index in range(3):
        path = tmp_path / f"{index:05d}.wav"
        AudioSegment.silent(duration=100).export(path, format="wav")
        segments.append({
            "audio_path": str(path),
            "order": index,
            "chapter": "第一章" if index < 2 else "第二章",
        })

    result = AudioSplicer().splice(segments)

    assert len(result) == 100 + 300 + 100 + 2000 + 100


def test_audio_splicer_fails_instead_of_shifting_metadata(tmp_path):
    with pytest.raises(RuntimeError, match="Failed to load"):
        AudioSplicer().splice([
            {"audio_path": str(tmp_path / "missing.wav"), "order": 0, "chapter": ""}
        ])
