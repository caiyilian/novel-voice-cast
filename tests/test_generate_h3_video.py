from pathlib import Path

from scripts.generate_h3_video import render_segment, scale_filter, segment_specs


def test_scale_filter_normalizes_frame_rate_and_time_base():
    value = scale_filter(864, 480, 24)
    assert "fps=24" in value
    assert "settb=expr=1/24" in value


def test_native_segment_uses_h3_continuation_without_legacy_image_directory(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    frame = tmp_path / "continuation.png"
    clip.write_bytes(b"video")
    frame.write_bytes(b"frame")

    specs = segment_specs(
        [{"title": "beat"}],
        [{"start_ms": 0, "end_ms": 12_000}],
        [
            {
                "index": 0,
                "status": "success",
                "output_file": str(clip),
                "duration_seconds": 5.166667,
                "continuation_frame": str(frame),
            }
        ],
        h3_mode="native-chain",
        images_dir=None,
        segments_output_dir=tmp_path / "segments",
        fps=25,
    )

    assert specs[0]["clip"] == clip.resolve()
    assert specs[0]["image"] == frame.resolve()
    assert specs[0]["placement"] == "start"
    assert specs[0]["frame_count"] == 300


def test_native_segment_requires_a_continuation_frame(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")

    try:
        segment_specs(
            [{"title": "beat"}],
            [{"start_ms": 0, "end_ms": 1_000}],
            [{"index": 0, "status": "success", "output_file": str(clip), "duration_seconds": 1}],
            h3_mode="native-chain",
            images_dir=None,
            segments_output_dir=tmp_path / "segments",
            fps=25,
        )
    except ValueError as exc:
        assert "continuation frame" in str(exc)
    else:
        raise AssertionError("missing continuation frame should fail")


def test_render_segment_forces_exact_cfr_after_concat(tmp_path: Path, monkeypatch):
    image = tmp_path / "frame.png"
    clip = tmp_path / "clip.mp4"
    output = tmp_path / "segment.mp4"
    image.write_bytes(b"image")
    clip.write_bytes(b"clip")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"rendered")

    monkeypatch.setattr("scripts.generate_h3_video.run_command", fake_run)
    render_segment(
        {
            "frame_count": 288,
            "clip": clip,
            "clip_duration": 5.167,
            "image": image,
            "placement": "start",
        },
        output=output,
        width=864,
        height=480,
        fps=24,
        crf=18,
        preset="slow",
        ffmpeg="ffmpeg",
    )

    command = commands[0]
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "settb=expr=1/24" in filter_graph
    assert "setpts=N/(24*TB)" in filter_graph
    assert command[command.index("-r") + 1] == "24"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert output.read_bytes() == b"rendered"
