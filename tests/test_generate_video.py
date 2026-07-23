from scripts.generate_video import (
    build_illustration_timeline,
    build_source_line_timeline,
    write_slideshow_concat,
)


def test_source_line_timeline_merges_multiple_segments_on_one_line():
    timestamps = [
        {"line": 2, "start_ms": 0, "end_ms": 100},
        {"line": 3, "start_ms": 400, "end_ms": 600},
        {"line": 3, "start_ms": 900, "end_ms": 1200},
        {"line": 6, "start_ms": 1500, "end_ms": 1900},
    ]

    source_timeline = build_source_line_timeline(timestamps)

    assert source_timeline[3] == {"start_ms": 400, "end_ms": 1200}
    assert source_timeline[6] == {"start_ms": 1500, "end_ms": 1900}


def test_illustration_timeline_covers_audio_tail_and_missing_source_lines():
    source_timeline = {
        2: {"start_ms": 0, "end_ms": 1000},
        6: {"start_ms": 1300, "end_ms": 1800},
    }
    illustrations = [
        {"title": "opening", "start_line": 1, "end_line": 3, "image_idx": 0},
        {"title": "later", "start_line": 4, "end_line": 6, "image_idx": 1},
    ]

    timeline = build_illustration_timeline(
        illustrations,
        [],
        source_line_timeline=source_timeline,
        total_duration_ms=2200,
    )

    assert timeline[0]["start_ms"] == 0
    assert timeline[0]["end_ms"] == 1300
    assert timeline[1]["start_ms"] == 1300
    assert timeline[1]["end_ms"] == 2200
    assert sum(item["duration_ms"] for item in timeline) == 2200


def test_illustration_timeline_handles_no_illustrations():
    assert build_illustration_timeline([], [], source_line_timeline={}, total_duration_ms=1000) == []


def test_concat_repeats_last_image_so_duration_is_honored(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    concat = tmp_path / "concat.txt"
    timeline = [
        {"image_idx": 0, "duration_ms": 1250},
        {"image_idx": 1, "duration_ms": 2750},
    ]
    illustrations = [
        {"image_path": str(first)},
        {"image_path": str(last)},
    ]

    total_ms = write_slideshow_concat(timeline, illustrations, concat)
    content = concat.read_text(encoding="utf-8").splitlines()

    assert total_ms == 4000
    assert content[-1].endswith("last.png'")
    assert content[-2] == "duration 2.750"
