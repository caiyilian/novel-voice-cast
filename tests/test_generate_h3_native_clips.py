import json
from pathlib import Path

import pytest

from scripts.generate_h3_native_clips import (
    H3MotionQualityError,
    build_native_prompt,
    build_records,
    continuous_duration_parts,
    extract_continuation_frame,
    inspect_motion_quality,
    migrate_legacy_records,
    prepare_checkpoint,
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


def test_continuous_parts_cover_full_interval_without_short_tail():
    parts = continuous_duration_parts(
        12.0,
        minimum_duration=5,
        maximum_duration=10,
    )

    assert len(parts) == 2
    assert parts[0][3] == pytest.approx(0.0)
    assert parts[-1][4] == pytest.approx(12.0)
    assert sum(part[2] for part in parts) == pytest.approx(12.0)
    assert all(part[1] >= part[2] for part in parts)


def test_continuous_records_flatten_beats_and_reset_finite_chains(tmp_path: Path):
    plan = [
        {"start_line": 2, "title": "one", "prompt": "one"},
        {"start_line": 3, "title": "two", "prompt": "two"},
    ]
    timeline = [{"duration_ms": 22_000}, {"duration_ms": 2_000}]
    records = build_records(
        plan,
        timeline,
        SCENES,
        output_dir=tmp_path / "clips",
        frames_dir=tmp_path / "frames",
        minimum_duration=5,
        maximum_duration=10,
        limit=None,
        mode="continuous-chain",
        max_chain_length=3,
    )

    assert len(records) == 4
    assert [record["beat_index"] for record in records] == [0, 0, 0, 1]
    assert [record["part_index"] for record in records] == [0, 1, 2, 0]
    assert [record["continuity_reset"] for record in records] == [True, False, False, True]
    assert records[2]["coverage_end_seconds"] == pytest.approx(22.0)
    assert records[3]["coverage_end_seconds"] == pytest.approx(2.0)
    assert all("_p" in Path(record["output_target"]).name for record in records)


def test_native_prompt_describes_micro_shot_and_first_last_frames():
    prompt = build_native_prompt(
        {"title": "arrival", "description": "a traveler arrives", "prompt": "anime field"},
        SCENES[0],
        duration_seconds=6,
        uses_first_frame=True,
        uses_last_frame=True,
        part_index=1,
        part_count=3,
        phase="develop",
    )

    assert "Picture 1" in prompt and "Picture 2" in prompt
    assert "micro-shot 2 of 3" in prompt


def test_motion_quality_gate_rejects_long_freeze(tmp_path: Path, monkeypatch):
    class Result:
        returncode = 0
        stderr = "freeze_duration: 7.0 black_duration:0.2"

    monkeypatch.setattr("scripts.generate_h3_native_clips.subprocess.run", lambda *_a, **_k: Result())

    with pytest.raises(H3MotionQualityError) as exc:
        inspect_motion_quality(
            tmp_path / "clip.mp4",
            usable_duration=10.0,
            ffmpeg="ffmpeg",
            max_freeze_ratio=0.65,
            max_black_ratio=0.20,
        )
    assert exc.value.metrics["freeze_ratio"] == pytest.approx(0.7)


def test_continuous_checkpoint_reuses_unchanged_records_after_local_plan_change(tmp_path: Path):
    timeline = [{"duration_ms": 6_000}, {"duration_ms": 6_000}]
    original = build_records(
        [
            {"start_line": 2, "title": "one", "prompt": "one"},
            {"start_line": 3, "title": "two", "prompt": "two"},
        ],
        timeline,
        SCENES,
        output_dir=tmp_path / "clips",
        frames_dir=tmp_path / "frames",
        minimum_duration=5,
        maximum_duration=10,
        limit=None,
        mode="continuous-chain",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    old = prepare_checkpoint(
        checkpoint_path,
        original,
        endpoint="http://h3",
        width=672,
        height=864,
        minimum_duration=5,
        maximum_duration=10,
        scene_source_hash="scene",
        resume=False,
        mode="continuous-chain",
    )
    old["clips"][0]["status"] = "queued"
    old["clips"][0]["job_id"] = "keep-me"
    checkpoint_path.write_text(json.dumps(old), encoding="utf-8")
    changed = build_records(
        [
            {"start_line": 2, "title": "one", "prompt": "one"},
            {"start_line": 3, "title": "changed", "prompt": "changed"},
        ],
        timeline,
        SCENES,
        output_dir=tmp_path / "clips",
        frames_dir=tmp_path / "frames",
        minimum_duration=5,
        maximum_duration=10,
        limit=None,
        mode="continuous-chain",
    )

    resumed = prepare_checkpoint(
        checkpoint_path,
        changed,
        endpoint="http://h3",
        width=672,
        height=864,
        minimum_duration=5,
        maximum_duration=10,
        scene_source_hash="scene",
        resume=True,
        mode="continuous-chain",
    )

    assert resumed["clips"][0]["job_id"] == "keep-me"
    assert resumed["clips"][1]["status"] == "pending"


def test_legacy_running_job_is_transferred_without_resubmission(tmp_path: Path):
    records = build_records(
        [{"start_line": 2, "title": "one", "prompt": "one"}],
        [{"duration_ms": 6_000}],
        SCENES,
        output_dir=tmp_path / "clips",
        frames_dir=tmp_path / "frames",
        minimum_duration=5,
        maximum_duration=10,
        limit=None,
        mode="continuous-chain",
    )
    checkpoint = prepare_checkpoint(
        tmp_path / "continuous.json",
        records,
        endpoint="http://h3",
        width=672,
        height=864,
        minimum_duration=5,
        maximum_duration=10,
        scene_source_hash="scene",
        resume=False,
        mode="continuous-chain",
    )
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "mode": "native-chain",
                "width": 672,
                "height": 864,
                "clips": [
                    {
                        "index": 0,
                        "status": "running",
                        "attempts": 1,
                        "job_id": "remote-job-73",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    migrated = migrate_legacy_records(
        checkpoint,
        legacy_path,
        checkpoint_path=tmp_path / "continuous.json",
        keyframes_dir=tmp_path / "keyframes",
        width=672,
        height=864,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        max_freeze_ratio=0.65,
        max_black_ratio=0.20,
    )

    assert migrated == 1
    assert checkpoint["clips"][0]["job_id"] == "remote-job-73"
    assert checkpoint["clips"][0]["legacy_job_recovery"] is True


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
