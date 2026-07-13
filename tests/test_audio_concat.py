from pathlib import Path
import sys

import pytest
from pydub import AudioSegment


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.core.bgm_mixer as bgm_mixer_module  # noqa: E402
import scripts.run_full as run_full_module  # noqa: E402
from app.core.bgm_mixer import (  # noqa: E402
    BGM_MIX_CHUNK_MS,
    _concat_via_ffmpeg,
    _extract_audio_segment_ffmpeg,
    _iter_interval_chunks,
    _mix_audio_segment,
    _prepare_bgm,
    _validate_speech_timeline,
)
from app.core.splicer import (  # noqa: E402
    AudioSplicer,
    SpliceFileResult,
    concat_wav_files_ffmpeg,
)


def _write_silence(path: Path, duration_ms: int, frame_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate).export(path, format="wav")


def test_ffmpeg_concat_handles_quoted_paths_and_preserves_duration(tmp_path):
    source_dir = tmp_path / "speaker's clips"
    first = source_dir / "part 1.wav"
    second = source_dir / "part 2.wav"
    _write_silence(first, 125)
    _write_silence(second, 275)

    output = tmp_path / "joined.wav"
    duration = concat_wav_files_ffmpeg(
        [first, second],
        output,
        output_format="wav",
        sample_rate=24000,
        channels=1,
        expected_duration_seconds=0.4,
        duration_tolerance_seconds=0.01,
    )

    assert duration == pytest.approx(0.4, abs=0.002)
    assert len(AudioSegment.from_file(output)) == 400


def test_ffmpeg_concat_does_not_replace_destination_on_validation_failure(tmp_path):
    part = tmp_path / "part.wav"
    output = tmp_path / "joined.wav"
    _write_silence(part, 100)
    output.write_bytes(b"existing output")

    with pytest.raises(RuntimeError, match="duration mismatch"):
        concat_wav_files_ffmpeg(
            [part],
            output,
            expected_duration_seconds=10.0,
            duration_tolerance_seconds=0.001,
        )

    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob("ffconcat_*"))


def test_audio_splicer_streams_output_and_keeps_shared_gaps(tmp_path):
    segments = []
    for index in range(3):
        path = tmp_path / f"{index:05d}.wav"
        _write_silence(path, 100)
        segments.append({
            "audio_path": str(path),
            "order": index,
            "chapter": "第一章" if index < 2 else "第二章",
        })

    output = tmp_path / "full_volume.wav"
    result = AudioSplicer().splice(segments, output_path=str(output))

    assert isinstance(result, SpliceFileResult)
    assert len(result) == 100 + 300 + 100 + 2000 + 100
    assert len(AudioSegment.from_file(output)) == len(result)
    assert not list(tmp_path.glob("audio_splice_*"))
    assert not list(tmp_path.glob("ffconcat_*"))


def test_audio_splicer_encodes_mp3_suffix_and_ignores_encoder_padding_in_result(tmp_path):
    segments = []
    for index in range(2):
        path = tmp_path / f"voice_{index}.wav"
        _write_silence(path, 200, frame_rate=24000)
        segments.append({"audio_path": str(path), "order": index})

    output = tmp_path / "full_volume.mp3"
    result = AudioSplicer(output_bitrate="96k").splice(segments, output_path=str(output))

    assert isinstance(result, SpliceFileResult)
    assert len(result) == 200 + 300 + 200
    assert output.read_bytes()[:4] != b"RIFF"
    assert len(AudioSegment.from_file(output)) == pytest.approx(len(result), abs=100)


def test_full_pipeline_passes_configured_bitrate_to_splicer(tmp_path, monkeypatch):
    captured = {}

    class FakeResult:
        def __len__(self):
            return 1234

    class FakeSplicer:
        def __init__(self, *, output_bitrate):
            captured["bitrate"] = output_bitrate

        def splice(self, _segments, *, output_path):
            Path(output_path).write_bytes(b"encoded")
            return FakeResult()

    monkeypatch.setattr(run_full_module, "AudioSplicer", FakeSplicer)
    output_path, duration = run_full_module.step_splice(
        {
            "output": {
                "dir": str(tmp_path),
                "filename": "voice",
                "format": "mp3",
                "bitrate": "48k",
            }
        },
        [],
    )

    assert captured["bitrate"] == "48k"
    assert output_path == str(tmp_path / "voice.mp3")
    assert duration == pytest.approx(1.234)


def test_bgm_concat_streams_to_mp3(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_silence(first, 200, frame_rate=48000)
    _write_silence(second, 300, frame_rate=48000)

    output = tmp_path / "mixed.mp3"
    duration = _concat_via_ffmpeg([first, second], output, expected_duration=0.5)

    assert output.stat().st_size > 100
    assert duration == pytest.approx(0.5, abs=0.1)


def test_bgm_concat_rejects_non_mp3_destination(tmp_path):
    with pytest.raises(ValueError, match=r"\.mp3 extension"):
        _concat_via_ffmpeg([], tmp_path / "mixed.wav", expected_duration=0.0)


def test_mix_bgm_rejects_non_mp3_destination_before_processing(tmp_path):
    with pytest.raises(ValueError, match=r"\.mp3 extension"):
        bgm_mixer_module.mix_bgm(output_path=tmp_path / "mixed.wav")


@pytest.mark.parametrize("source_format", ["wav", "mp3"])
def test_bgm_extractor_auto_detects_legacy_and_real_mp3_input(tmp_path, source_format):
    # Legacy AudioSplicer wrote a WAV payload under an .mp3 filename. New
    # output is an actual MP3; the mixer must accept both during migration.
    source = tmp_path / f"speech_{source_format}.mp3"
    AudioSegment.silent(duration=500, frame_rate=24000).export(source, format=source_format)

    extracted = tmp_path / f"extracted_{source_format}.wav"
    _extract_audio_segment_ffmpeg(source, 100, 300, extracted)

    assert len(AudioSegment.from_file(extracted)) == pytest.approx(200, abs=2)


def test_prepare_bgm_builds_multi_clip_cycle_to_exact_duration():
    clips = [
        AudioSegment.silent(duration=125, frame_rate=24000),
        AudioSegment.silent(duration=175, frame_rate=24000),
    ]

    assert len(_prepare_bgm(clips, 1050)) == 1050


def test_mixed_chunks_have_identical_concat_stream_parameters(tmp_path):
    speech = tmp_path / "speech.wav"
    _write_silence(speech, 250, frame_rate=48000)
    output_without_bgm = tmp_path / "without-bgm.wav"
    output_with_bgm = tmp_path / "with-bgm.wav"

    _mix_audio_segment(speech, None, "daily", None, output_without_bgm)
    _mix_audio_segment(
        speech,
        [AudioSegment.silent(duration=250, frame_rate=32000).set_channels(2)],
        "daily",
        None,
        output_with_bgm,
    )

    for output in (output_without_bgm, output_with_bgm):
        audio = AudioSegment.from_file(output)
        assert audio.frame_rate == 44100
        assert audio.channels == 2
        assert audio.sample_width == 2
        assert len(audio) == 250


def test_prepare_bgm_offset_keeps_loop_continuous_across_chunks():
    # Deliberately use clip frame counts that produce fractional-millisecond
    # tails. A rounded-ms modulo would be one frame out by the second chunk.
    clips = [
        AudioSegment(data=b"\x01\x00" * 5513, sample_width=2, frame_rate=44100, channels=1),
        AudioSegment(data=b"\x02\x00" * 7718, sample_width=2, frame_rate=44100, channels=1),
    ]

    complete = _prepare_bgm(clips, 1050)
    first = _prepare_bgm(clips, 400, offset_ms=0)
    second = _prepare_bgm(clips, 650, offset_ms=400)

    assert first.raw_data + second.raw_data == complete.raw_data


def test_interval_chunking_is_bounded_adjacent_and_complete():
    start = 1234
    end = start + BGM_MIX_CHUNK_MS * 2 + 5678

    chunks = _iter_interval_chunks(start, end)

    assert chunks == [
        (start, start + BGM_MIX_CHUNK_MS),
        (start + BGM_MIX_CHUNK_MS, start + BGM_MIX_CHUNK_MS * 2),
        (start + BGM_MIX_CHUNK_MS * 2, end),
    ]


def test_speech_timeline_validation_accepts_small_drift_and_rejects_stale_audio(tmp_path):
    speech = tmp_path / "speech.wav"
    _write_silence(speech, 500)

    assert _validate_speech_timeline(speech, 0.55, tolerance_seconds=0.1) == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="Regenerate the spliced speech audio"):
        _validate_speech_timeline(speech, 2.0, tolerance_seconds=0.1)


def test_mix_bgm_loads_only_active_interval_and_fades_only_original_boundaries(
    tmp_path,
    monkeypatch,
):
    speech = tmp_path / "speech.mp3"
    speech.write_bytes(b"speech-placeholder")
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    active_clip = bgm_dir / "001_0.mp3"
    inactive_clip = bgm_dir / "002_0.mp3"
    active_clip.write_bytes(b"active")
    inactive_clip.write_bytes(b"inactive")
    manifest = bgm_dir / "bgm_manifest.json"
    manifest.write_text('{"clips_per_segment": 1}', encoding="utf-8")
    segments = tmp_path / "bgm_segments.json"
    segments.write_text(
        "["
        '{"start_line": 1, "end_line": 3, "bgm_type": "daily"},'
        '{"start_line": 4, "end_line": 5, "bgm_type": "epic"}'
        "]",
        encoding="utf-8",
    )
    output = tmp_path / "mixed.mp3"

    monkeypatch.setattr(
        bgm_mixer_module,
        "load_config",
        lambda _path: {"output": {"dir": str(tmp_path)}, "bgm": {"volume_db": -8.0}},
    )
    monkeypatch.setattr(
        bgm_mixer_module,
        "_build_dialogue_line_map",
        lambda _config: ([1, 2, 3], ["第一章"] * 3),
    )
    monkeypatch.setattr(
        bgm_mixer_module,
        "_compute_dialogue_timestamps",
        lambda _n, _chapters, _segments_dir: (
            [0, 200_000, 400_000],
            [100_000, 300_000, 620_000],
        ),
    )
    monkeypatch.setattr(
        bgm_mixer_module,
        "_validate_speech_timeline",
        lambda _path, expected: expected,
    )

    loaded_paths = []

    def fake_from_file(path):
        loaded_paths.append(Path(path))
        return AudioSegment.silent(duration=100)

    monkeypatch.setattr(
        bgm_mixer_module.AudioSegment,
        "from_file",
        staticmethod(fake_from_file),
    )

    extracted = []

    def fake_extract(_speech, start_ms, end_ms, destination):
        extracted.append((start_ms, end_ms))
        destination.write_bytes(b"speech")

    monkeypatch.setattr(bgm_mixer_module, "_extract_audio_segment_ffmpeg", fake_extract)

    mixed_calls = []

    def fake_mix(_speech, _clips, _type, _previous_type, destination, **kwargs):
        mixed_calls.append(kwargs)
        destination.write_bytes(b"mixed")

    monkeypatch.setattr(bgm_mixer_module, "_mix_audio_segment", fake_mix)

    def fake_concat(wav_files, destination, expected_duration):
        assert len(wav_files) == 3
        assert expected_duration == 620.0
        destination.write_bytes(b"mp3")
        return expected_duration

    monkeypatch.setattr(bgm_mixer_module, "_concat_via_ffmpeg", fake_concat)

    duration = bgm_mixer_module.mix_bgm(
        speech_path=speech,
        bgm_dir=bgm_dir,
        manifest_path=manifest,
        segments_path=segments,
        output_path=output,
        config_path=str(tmp_path / "unused.yaml"),
    )

    assert duration == 620.0
    assert loaded_paths == [active_clip]
    assert extracted == [
        (0, BGM_MIX_CHUNK_MS),
        (BGM_MIX_CHUNK_MS, BGM_MIX_CHUNK_MS * 2),
        (BGM_MIX_CHUNK_MS * 2, 620_000),
    ]
    assert [call["bgm_offset_ms"] for call in mixed_calls] == [0, 300_000, 600_000]
    assert [call["fade_in_at_start"] for call in mixed_calls] == [True, False, False]
    assert [call["fade_out_at_end"] for call in mixed_calls] == [False, False, True]
