import base64
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.core.subtitles import (  # noqa: E402
    SubtitleEntry,
    build_segment_timestamps,
    build_srt,
    escape_srt_text,
    load_subtitle_entries,
    ms_to_srt,
    split_subtitle_chunks,
    split_subtitle_text,
)
from scripts.generate_video import (  # noqa: E402
    build_illustration_timeline,
    build_source_line_timeline,
    build_subtitle_filter,
    build_video_filter_chain,
    ffmpeg_supports_filter,
    main as generate_video,
    probe_media_duration,
    validate_audio_timeline,
    write_slideshow_concat,
)


def _write_silence(path: Path, duration_ms: int, sample_rate: int = 1000) -> None:
    frame_count = duration_ms * sample_rate // 1000
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\0\0" * frame_count)


def test_split_prefers_semantic_boundaries_and_repeats_speaker():
    text = "这是第一句话，接着是第二句话。然后继续说明，最后结束。"

    chunks = split_subtitle_chunks(text, "旁白")

    assert "".join(chunk.content for chunk in chunks) == text
    assert all(chunk.display.startswith("[旁白] ") for chunk in chunks)
    assert all(len(cue.splitlines()) <= 2 for cue in [chunk.display for chunk in chunks])
    assert all(len(line) <= 16 for chunk in chunks for line in chunk.display.splitlines())
    assert chunks[0].display.splitlines()[0].endswith("，")


def test_split_falls_back_to_hard_break_without_losing_characters():
    text = "甲" * 75

    chunks = split_subtitle_chunks(text, "")

    assert "".join(chunk.content for chunk in chunks) == text
    assert all(len(line) <= 16 for chunk in chunks for line in chunk.display.splitlines())
    assert all(len(chunk.display.splitlines()) <= 2 for chunk in chunks)
    assert [len(chunk.content) for chunk in chunks] == [25, 25, 25]


@pytest.mark.parametrize(
    ("length", "expected_lines"),
    [
        (17, [8, 9]),
        (31, [15, 16]),
        (32, [16, 16]),
    ],
)
def test_unpunctuated_two_line_layout_is_lower_heavy(length, expected_lines):
    chunks = split_subtitle_chunks("甲" * length)

    assert len(chunks) == 1
    assert [len(line) for line in chunks[0].display.splitlines()] == expected_lines


def test_split_preserves_intentional_internal_spaces():
    text = "甲" * 14 + "  " + "乙" * 10

    chunks = split_subtitle_chunks(text)

    assert "".join(chunk.content for chunk in chunks) == text
    assert "".join(chunk.display.replace("\n", "") for chunk in chunks) == text


def test_split_keeps_punctuation_pairs_and_closing_quotes_together():
    text = "甲" * 20 + "……" + "乙" * 20 + "——" + "丙" * 20 + "！」"

    chunks = split_subtitle_chunks(text)
    boundaries = []
    offset = 0
    for chunk in chunks:
        for line in chunk.display.splitlines()[:-1]:
            boundaries.append(offset + len(line))
        offset += len(chunk.content)
        boundaries.append(offset)

    assert "".join(chunk.content for chunk in chunks) == text
    assert all(
        text[max(0, position - 1):position + 1] not in {"……", "——", "！」"}
        for position in boundaries
    )


def test_split_never_starts_a_line_with_a_closing_quote():
    text = "A" * 14 + "「B」" + "C" * 14

    chunks = split_subtitle_chunks(text)

    assert "".join(chunk.content for chunk in chunks) == text
    assert all(not line.startswith("」") for chunk in chunks for line in chunk.display.splitlines())


def test_split_keeps_variation_selectors_with_their_base_character():
    text = "甲" * 10 + "❤️" + "乙" * 10

    chunks = split_subtitle_chunks(text)

    assert all(
        not line.endswith("❤")
        for chunk in chunks
        for line in chunk.display.splitlines()[:-1]
    )


@pytest.mark.parametrize("cluster", ["👍🏽", "🇨🇳"])
def test_split_keeps_common_emoji_clusters_together(cluster):
    text = "甲" * 10 + cluster + "乙" * 10

    chunks = split_subtitle_chunks(text)
    boundaries = []
    offset = 0
    for chunk in chunks:
        for line in chunk.display.splitlines()[:-1]:
            boundaries.append(offset + len(line))
        offset += len(chunk.content)
        boundaries.append(offset)

    cluster_start = text.index(cluster)
    assert all(
        not (cluster_start < boundary < cluster_start + len(cluster))
        for boundary in boundaries
    )


def test_split_flattens_all_unicode_line_separators():
    text = ("甲" * 10 + "\u2028") * 3 + "乙" * 10

    chunks = split_subtitle_chunks(text)

    assert all(len(chunk.display.splitlines()) <= 2 for chunk in chunks)


def test_split_normalises_tabs_without_collapsing_regular_spaces():
    chunks = split_subtitle_chunks("甲\t乙  丙")

    assert "".join(chunk.content for chunk in chunks) == "甲 乙  丙"


@pytest.mark.parametrize("punctuation", ["，", "。", "！", "？", "；", "：", "、"])
def test_split_never_starts_a_line_with_punctuation(punctuation):
    text = "A" * 16 + punctuation + "B" * 15

    chunks = split_subtitle_chunks(text)

    assert all(
        not line.startswith(punctuation)
        for chunk in chunks
        for line in chunk.display.splitlines()
    )


def test_speaker_separator_does_not_create_a_prefix_only_line():
    cue = split_subtitle_text("甲" * 24, "旁白")[0]

    assert cue.splitlines()[0] != "[旁白]"


def test_split_rejects_speaker_prefix_that_cannot_share_a_line():
    with pytest.raises(ValueError, match="speaker prefix"):
        split_subtitle_text("正文", "一二三四五六七八九十甲乙丙丁")


def test_speaker_prefix_may_exactly_fill_the_first_line():
    # Brackets plus the trailing space make this prefix exactly 16 characters.
    speaker = "一二三四五六七八九十甲乙丙"

    cue = split_subtitle_text("正文", speaker)[0]

    assert cue.splitlines() == [f"[{speaker}] ", "正文"]


def test_load_line_aligned_attachment_format(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("第一句。\n\n第二句。\n", encoding="utf-8")
    labels.write_text("甲\n\n乙\n", encoding="utf-8")

    entries = load_subtitle_entries(novel, labels)

    assert [(entry.text, entry.speaker, entry.source_line) for entry in entries] == [
        ("第一句。", "甲", 1),
        ("第二句。", "乙", 3),
    ]


def test_load_strips_utf8_bom_from_novel_and_labels(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("第一句。\n", encoding="utf-8-sig")
    labels.write_text("甲\n", encoding="utf-8-sig")

    entries = load_subtitle_entries(novel, labels, label_mode="line")

    assert entries == [SubtitleEntry("第一句。", "甲", source_line=1)]


def test_load_dialogue_order_labels_through_project_parser(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("第一章\n这是一段足够长的旁白。\n「你好」\n", encoding="utf-8")
    labels.write_text("赫萝\n", encoding="utf-8")

    entries = load_subtitle_entries(novel, labels)

    assert [(entry.text, entry.speaker, entry.source_line) for entry in entries] == [
        ("这是一段足够长的旁白。", "旁白", 2),
        ("你好", "赫萝", 3),
    ]


def test_auto_mode_uses_parser_order_with_line_aligned_labels(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("第一章\n这是一段足够长的旁白。\n「你好」\n", encoding="utf-8")
    labels.write_text("\n旁白\n赫萝\n", encoding="utf-8")

    entries = load_subtitle_entries(
        novel,
        labels,
        label_mode="parsed-line",
        expected_segment_count=2,
    )

    assert [(entry.text, entry.speaker, entry.source_line) for entry in entries] == [
        ("这是一段足够长的旁白。", "旁白", 2),
        ("你好", "赫萝", 3),
    ]


def test_line_mode_hides_non_character_sound_sentinel(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("钟声响起。\n", encoding="utf-8")
    labels.write_text("非人物发声\n", encoding="utf-8")

    entries = load_subtitle_entries(novel, labels, label_mode="line")

    assert entries[0].speaker == ""


def test_auto_mode_rejects_equal_length_but_structurally_shifted_fixture(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("这是一段足够长的旁白。\n插图\n「你好」\n", encoding="utf-8")
    labels.write_text("旁白\n村民\n赫萝\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no subtitle label interpretation"):
        load_subtitle_entries(novel, labels, expected_segment_count=3)

    assert len(load_subtitle_entries(novel, labels, label_mode="line")) == 3


def test_auto_mode_rejects_ambiguous_equal_count_layouts(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    novel.write_text("「甲」「乙」\n短\n", encoding="utf-8")
    labels.write_text("角色甲\n角色乙\n", encoding="utf-8")

    with pytest.raises(ValueError, match="layout is ambiguous"):
        load_subtitle_entries(novel, labels, expected_segment_count=2)


def test_repository_labels_follow_parser_order_without_heading_shift():
    entries = load_subtitle_entries(
        ROOT / "novels" / "novel.txt",
        ROOT / "novels" / "labels.txt",
        expected_segment_count=3026,
    )

    assert len(entries) == 3026
    assert (entries[17].text, entries[17].speaker, entries[17].source_line) == (
        "这是最后一件了吧？",
        "村民",
        23,
    )
    assert entries[-1].source_line == 3064


def test_segment_timestamps_match_dialogue_chapter_and_paragraph_gaps(tmp_path):
    for index, duration in enumerate((100, 200, 300, 400)):
        _write_silence(tmp_path / f"{index:05d}.wav", duration)
    entries = [
        SubtitleEntry("一", chapter="第一章", paragraph=1),
        SubtitleEntry("二", chapter="第一章", paragraph=1),
        SubtitleEntry("三", chapter="第一章", paragraph=2),
        SubtitleEntry("四", chapter="第二章", paragraph=1),
    ]

    timestamps = build_segment_timestamps(entries, tmp_path)

    assert [(item["start_ms"], item["end_ms"]) for item in timestamps] == [
        (0, 100),
        (400, 600),
        (1600, 1900),
        (3900, 4300),
    ]


def test_segment_timestamps_fail_on_count_mismatch(tmp_path):
    _write_silence(tmp_path / "00000.wav", 100)

    with pytest.raises(ValueError, match="count mismatch"):
        build_segment_timestamps([SubtitleEntry("一"), SubtitleEntry("二")], tmp_path)


def test_segment_timestamps_fail_on_non_contiguous_names(tmp_path):
    _write_silence(tmp_path / "00000.wav", 100)
    _write_silence(tmp_path / "00002.wav", 100)

    with pytest.raises(ValueError, match="contiguous"):
        build_segment_timestamps([SubtitleEntry("一"), SubtitleEntry("二")], tmp_path)


def test_segment_timestamps_fall_back_to_entry_order_for_missing_source_line(tmp_path):
    _write_silence(tmp_path / "00000.wav", 100)

    timestamps = build_segment_timestamps([SubtitleEntry("一")], tmp_path)

    assert timestamps[0]["line"] == 1


def test_srt_split_cues_cover_original_interval_without_gaps():
    text = "甲" * 70
    entries = [SubtitleEntry(text, speaker="旁白")]
    timestamps = [{"start_ms": 1000, "end_ms": 8000}]

    srt = build_srt(entries, timestamps)
    timing_lines = [line for line in srt.splitlines() if " --> " in line]
    chunks = split_subtitle_chunks(text, "旁白")

    assert timing_lines[0].startswith("00:00:01,000 --> ")
    assert timing_lines[-1].endswith("00:00:08,000")
    cumulative = 0
    for timing, chunk in zip(timing_lines[:-1], chunks[:-1]):
        cumulative += len(chunk.content)
        expected_end = 1000 + (7000 * cumulative + len(text) // 2) // len(text)
        assert timing.endswith(f" --> {ms_to_srt(expected_end)}")
    for previous, following in zip(timing_lines, timing_lines[1:]):
        assert previous.split(" --> ")[1] == following.split(" --> ")[0]


def test_srt_rejects_segment_too_short_for_split_cues():
    with pytest.raises(ValueError, match="too short"):
        build_srt(
            [SubtitleEntry("甲" * 70)],
            [{"start_ms": 0, "end_ms": 1}],
        )


def test_ms_to_srt_handles_six_hour_video():
    assert ms_to_srt(6 * 60 * 60 * 1000 + 12_345) == "06:00:12,345"


def test_build_srt_requires_matching_entry_and_timestamp_counts():
    with pytest.raises(ValueError, match="mismatch"):
        build_srt([SubtitleEntry("一")], [])


def test_srt_escapes_html_and_ass_injection_after_layout():
    raw = r"<b>正文</b>{\an1}\N &lt;"

    escaped = escape_srt_text(raw)

    assert "<b>" not in escaped
    assert r"{\an1}" not in escaped
    assert r"\N" not in escaped
    assert "&lt;" not in escaped
    assert "\\u2060" not in escaped  # the real word-joiner is not a six-char escape
    round_trip = (
        escaped.replace(r"\{{}", "{")
        .replace("&\u2060", "&")
        .replace("\\\u2060", "\\")
        .replace("<\u2060", "<")
    )
    assert round_trip == raw


def test_ffmpeg_filter_chain_expands_sparse_frames_before_subtitles(tmp_path):
    subtitle_filter = build_subtitle_filter(tmp_path / "subtitles.srt")

    chain = build_video_filter_chain(subtitle_filter)

    assert chain.startswith("scale=896:1152,fps=25,subtitles=")
    assert "FontName=SimHei" in chain
    assert "FontSize=20" in chain
    assert "WrapStyle=2" in chain
    assert chain.endswith("wrap_unicode=1")
    assert build_video_filter_chain() == "scale=896:1152,fps=25"


def test_video_filters_support_landscape_canvas(tmp_path):
    subtitle_filter = build_subtitle_filter(
        tmp_path / "landscape.srt",
        video_width=1280,
        video_height=720,
    )

    assert "PlayResX\\=1280" in subtitle_filter or "PlayResX=1280" in subtitle_filter
    assert "PlayResY\\=720" in subtitle_filter or "PlayResY=720" in subtitle_filter
    assert build_video_filter_chain(
        subtitle_filter,
        video_width=1280,
        video_height=720,
    ).startswith("scale=1280:720,fps=25,subtitles=")


def test_audio_timeline_validation_rejects_accumulated_drift():
    validate_audio_timeline(582.740, 582_740)

    with pytest.raises(ValueError, match="时间轴不一致"):
        validate_audio_timeline(570.0, 582_740)


def test_illustrations_map_raw_lines_to_dialogue_timestamps():
    timestamps = [
        {"line": line, "start_ms": index * 100, "end_ms": index * 100 + 80}
        for index, line in enumerate([*range(1, 18), 23, 24])
    ]
    source_timeline = build_source_line_timeline(timestamps)
    illustrations = [
        {"start_line": 1, "end_line": 22, "image_idx": 0},
        {"start_line": 23, "end_line": 23, "image_idx": 1},
        {"start_line": 24, "end_line": 24, "image_idx": 2},
    ]

    timeline = build_illustration_timeline(
        illustrations,
        [item["start_ms"] for item in timestamps],
        source_line_timeline=source_timeline,
        total_duration_ms=1880,
    )

    assert [item["start_ms"] for item in timeline] == [0, 1700, 1800]
    assert [item["end_ms"] for item in timeline] == [1700, 1800, 1880]


def test_duplicate_illustration_starts_share_real_interval_without_drift():
    source_timeline = {
        1: {"start_ms": 0, "end_ms": 80},
        10: {"start_ms": 100, "end_ms": 180},
    }
    illustrations = [
        {"start_line": 1, "end_line": 1, "image_idx": 0},
        {"start_line": 5, "end_line": 5, "image_idx": 1},
        {"start_line": 5, "end_line": 5, "image_idx": 2},
    ]

    timeline = build_illustration_timeline(
        illustrations,
        [0, 100],
        source_line_timeline=source_timeline,
        total_duration_ms=400,
    )

    assert [(item["start_ms"], item["end_ms"]) for item in timeline] == [
        (0, 100),
        (100, 250),
        (250, 400),
    ]
    assert sum(item["duration_ms"] for item in timeline) == 400


def test_slideshow_concat_preserves_short_duration_and_escapes_quote(tmp_path):
    image_path = tmp_path / "O'Brien.png"
    image_path.write_bytes(b"png fixture")
    concat_path = tmp_path / "video.ffconcat"
    timeline = [{"image_idx": 0, "duration_ms": 10}]
    illustrations = [{"image_path": str(image_path)}]

    total_ms = write_slideshow_concat(timeline, illustrations, concat_path)
    content = concat_path.read_text(encoding="utf-8")

    assert total_ms == 10
    assert "duration 0.010" in content
    assert "O'\\''Brien.png" in content
    assert content.count("file ") == 2


def test_slideshow_concat_fails_on_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError, match="缺少插图"):
        write_slideshow_concat(
            [{"image_idx": 0, "duration_ms": 100}],
            [{"image_path": str(tmp_path / "missing.png")}],
            tmp_path / "video.ffconcat",
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or not ffmpeg_supports_filter(shutil.which("ffmpeg") or "ffmpeg", "subtitles"),
    reason="FFmpeg with the libass subtitles filter is required for the video integration test",
)
def test_generate_video_cli_burns_subtitles_end_to_end(tmp_path):
    novel = tmp_path / "novel.txt"
    labels = tmp_path / "labels.txt"
    segments_dir = tmp_path / "segments"
    illustrations_dir = tmp_path / "illustrations"
    segments_dir.mkdir()
    illustrations_dir.mkdir()
    novel.write_text("这是一段足够长的旁白。\n「你好」\n", encoding="utf-8")
    labels.write_text("甲\n", encoding="utf-8")
    for index in range(2):
        _write_silence(segments_dir / f"{index:05d}.wav", 400, sample_rate=24000)

    # 1x1 valid PNG; scaling is intentionally exercised by the real filter.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    (illustrations_dir / "0001.png").write_bytes(png)
    plan = tmp_path / "plan.json"
    plan.write_text(
        '[{"title":"测试","start_line":1,"end_line":2}]',
        encoding="utf-8",
    )

    source_wav = tmp_path / "audio.wav"
    audio = tmp_path / "audio.mp3"
    _write_silence(source_wav, 1100, sample_rate=24000)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(source_wav), "-c:a", "libmp3lame", str(audio),
        ],
        check=True,
    )

    output = tmp_path / "video.mp4"
    subtitle_output = tmp_path / "O'Brien, [subtitles]" / "captions.srt"
    fonts_dir = tmp_path / "font's [dir]"
    fonts_dir.mkdir()
    result = generate_video(
        [
            "--plan", str(plan),
            "--illustrations-dir", str(illustrations_dir),
            "--audio", str(audio),
            "--output", str(output),
            "--novel", str(novel),
            "--labels", str(labels),
            "--segments-dir", str(segments_dir),
            "--subtitle-output", str(subtitle_output),
            "--subtitle-fonts-dir", str(fonts_dir),
        ]
    )

    assert result == 0
    assert output.stat().st_size > 1000
    assert abs(probe_media_duration(output) - probe_media_duration(audio)) < 0.25
    assert not list(tmp_path.glob("*.rendering.*.mp4"))
    assert "[旁白]" in subtitle_output.read_text(encoding="utf-8")

    # The compatibility mode must not require either subtitle input.
    output_without_subtitles = tmp_path / "video-no-subtitles.mp4"
    unused_subtitle = tmp_path / "must-not-be-created.srt"
    result = generate_video(
        [
            "--plan", str(plan),
            "--illustrations-dir", str(illustrations_dir),
            "--audio", str(audio),
            "--output", str(output_without_subtitles),
            "--novel", str(tmp_path / "missing-novel.txt"),
            "--labels", str(tmp_path / "missing-labels.txt"),
            "--segments-dir", str(segments_dir),
            "--subtitle-output", str(unused_subtitle),
            "--no-subtitles",
        ]
    )

    assert result == 0
    assert output_without_subtitles.stat().st_size > 1000
    assert not unused_subtitle.exists()
    assert not list(tmp_path.glob("*.rendering.*.mp4"))
