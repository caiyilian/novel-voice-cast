from pathlib import Path

import pytest

from scripts.generate_h3_native_clips import (
    build_native_prompt,
    build_records,
    extract_continuation_frame,
    requested_duration_for_interval,
    scene_for_line,
)


SCENES = [
    {"start_line": 1, "end_line": 10, "title": "village", "description": "quiet fields"},
    {"start_line": 11, "end_line": 20, "title": "road", "description": "a wagon road"},
]


def test_native_scene_mapping_and_prompt_mode():
    assert scene_for_line(8, SCENES)[0] == 0
    assert scene_for_line(15, SCENES)[0] == 1
    assert scene_for_line(99, SCENES)[0] == 1

    item = {"title": "arrival", "description": "a traveler arrives", "prompt": "anime field"}
    t2v = build_native_prompt(item, SCENES[0], duration_seconds=5, uses_first_frame=False)
    i2v = build_native_prompt(item, SCENES[0], duration_seconds=5, uses_first_frame=True)

    assert "Picture 1" not in t2v
    assert "Picture 1 is fully referenced" in i2v
    assert "No spoken dialogue" in i2v


def test_native_short_beat_generates_minimum_clip_but_holds_at_audio_boundary():
    requested, expected, usable = requested_duration_for_interval(
        2.25,
        minimum_duration=5,
        maximum_duration=10,
    )
    assert requested == 5
    assert expected == pytest.approx(124 / 24)
    assert usable == pytest.approx(2.25)


def test_native_records_include_every_beat_and_scene_identity(tmp_path: Path):
    plan = [
        {"start_line": 2, "title": "one", "prompt": "one"},
        {"start_line": 12, "title": "two", "prompt": "two"},
    ]
    timeline = [
        {"duration_ms": 12_000},
        {"duration_ms": 2_000},
    ]
    records = build_records(
        plan,
        timeline,
        SCENES,
        output_dir=tmp_path / "clips",
        frames_dir=tmp_path / "frames",
        minimum_duration=5,
        maximum_duration=10,
        limit=None,
    )

    assert len(records) == len(plan)
    assert [record["scene_index"] for record in records] == [0, 1]
    assert records[0]["requested_duration"] == 10
    assert records[1]["requested_duration"] == 5
    assert records[1]["usable_duration"] == pytest.approx(2.0)


def test_continuation_frame_seek_stays_inside_last_video_frame(tmp_path: Path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    output = tmp_path / "continuation.png"
    clip.write_bytes(b"video")
    seeks = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **_kwargs):
        seeks.append(float(command[command.index("-ss") + 1]))
        Path(command[-1]).write_bytes(b"png")
        return Result()

    monkeypatch.setattr("scripts.generate_h3_native_clips.subprocess.run", fake_run)
    extract_continuation_frame(
        clip,
        output,
        timestamp_seconds=5.166667,
        media_duration_seconds=5.166667,
        ffmpeg="ffmpeg",
    )

    assert output.read_bytes() == b"png"
    assert seeks == [pytest.approx(5.166667 - 2 / 24)]
