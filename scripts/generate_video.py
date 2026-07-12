"""从插图 + 音频生成带中文字幕的视频。

流程:
1. 读取所有音频片段 (output/segments/*.wav)，计算每行对应的时间戳
2. 读取插图计划，将每张插图映射到时间范围
3. 生成每行最多 16 字、最多 2 行的中文字幕
4. 用 ffmpeg 一次性合成视频并烧录字幕

用法:
    .venv\\Scripts\\python scripts/generate_video.py
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.subtitles import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_LINES,
    build_segment_timestamps,
    discover_segment_files,
    get_wav_duration_ms,
    load_subtitle_entries,
    write_srt,
)
from app.core.timeline import GAP_DIALOGUE  # noqa: E402

SEGMENTS_DIR = Path("output/segments")
PLAN_PATH = Path("output/illustration_plan.json")
ILLUSTRATIONS_DIR = Path("output/illustrations")
AUDIO_PATH = Path("output/full_volume_bgm.mp3")
OUTPUT_PATH = Path("output/illustration_video.mp4")
NOVEL_PATH = Path("novels/novel.txt")
LABELS_PATH = Path("novels/labels.txt")
SUBTITLE_PATH = Path("output/_generated_subtitles.srt")


def get_segment_duration(wav_path: Path) -> int:
    """返回音频片段时长（毫秒）"""
    return get_wav_duration_ms(wav_path)


def build_line_timestamps(segments_dir: Path = SEGMENTS_DIR) -> list[int]:
    """计算每行对应的音频开始时间戳（毫秒）"""
    wav_files = discover_segment_files(segments_dir)
    if not wav_files:
        raise FileNotFoundError(f"没有找到音频片段: {segments_dir}")

    print(f"音频片段: {len(wav_files)} 个")

    # 计算每个片段的时长和时间戳
    seg_starts = []
    offset = 0
    for wav in wav_files:
        dur = get_segment_duration(wav)
        seg_starts.append(offset)
        # 片段间间隔（与 splicer 一致）
        gap = GAP_DIALOGUE
        offset += dur + gap

    # 每个片段对应一行（按文件索引）
    # segments 文件名 00000.wav = line 1, 00001.wav = line 2, ...
    # 但小说总行数可能大于片段数（空行/标题行无音频）
    return seg_starts


def build_source_line_timeline(
    segment_timestamps: list[dict],
) -> dict[int, dict[str, int]]:
    """Map raw novel line numbers to their first start and final end time."""

    result: dict[int, dict[str, int]] = {}
    for timestamp in segment_timestamps:
        line = timestamp.get("line")
        if line is None:
            continue
        line_number = int(line)
        start_ms = int(timestamp["start_ms"])
        end_ms = int(timestamp["end_ms"])
        if line_number not in result:
            result[line_number] = {"start_ms": start_ms, "end_ms": end_ms}
        else:
            result[line_number]["start_ms"] = min(result[line_number]["start_ms"], start_ms)
            result[line_number]["end_ms"] = max(result[line_number]["end_ms"], end_ms)
    return result


def _escape_filter_option_value(value: str) -> str:
    """Escape one value through FFmpeg's option and filtergraph parsers."""

    def escape_level(text: str, special: str) -> str:
        return "".join(f"\\{char}" if char in special else char for char in text)

    # The subtitles option parser consumes the first escape layer.  The outer
    # filtergraph parser must therefore preserve those backslashes and also
    # protect its own delimiters.  argv is passed directly, so there is no
    # third shell-escaping layer here.
    option_escaped = escape_level(value, "\\':")
    return escape_level(option_escaped, "\\'[],;")


def _escape_filter_value(path: Path) -> str:
    """Escape a path for FFmpeg's two filter parsing layers."""

    try:
        value = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        value = path.resolve().as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError(f"字幕路径不能包含换行符: {path}")
    return _escape_filter_option_value(value.replace("\\", "/"))


def build_subtitle_filter(
    subtitle_path: Path,
    *,
    font_name: str = "SimHei",
    font_size: int = 20,
    fonts_dir: Path | None = None,
) -> str:
    """Build a libass-backed subtitles filter with deterministic wrapping."""

    if font_size <= 0:
        raise ValueError("subtitle font size must be positive")
    if any(char in font_name for char in ",:\r\n"):
        raise ValueError("subtitle font name cannot contain commas, colons, or newlines")
    style = ",".join(
        [
            "PlayResX=896",
            "PlayResY=1152",
            "WrapStyle=2",
            f"FontName={font_name}",
            f"FontSize={font_size}",
            "PrimaryColour=&H00FFFFFF",
            "OutlineColour=&H00000000",
            "BorderStyle=1",
            "Outline=2",
            "Shadow=0",
            "Alignment=2",
            "MarginV=50",
        ]
    )
    options = [f"filename={_escape_filter_value(subtitle_path)}"]
    if fonts_dir is not None:
        options.append(f"fontsdir={_escape_filter_value(fonts_dir)}")
    options.extend(
        [f"force_style={_escape_filter_option_value(style)}", "wrap_unicode=1"]
    )
    return "subtitles=" + ":".join(options)


def build_video_filter_chain(subtitle_filter: str | None = None) -> str:
    """Build the ordered filter chain required by the sparse slideshow input."""

    filters = ["scale=896:1152"]
    if subtitle_filter:
        # Expand the sparse slideshow before libass so cues can change while
        # one illustration remains visible.
        filters.append("fps=25")
        filters.append(subtitle_filter)
    return ",".join(filters)


def probe_media_duration(path: Path) -> float:
    """Return media duration from ffprobe or raise with a useful error."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 无法读取 {path}: {result.stderr.strip()}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe 返回了无效时长: {result.stdout!r}") from exc
    if duration <= 0:
        raise RuntimeError(f"媒体时长必须为正数: {duration}")
    return duration


def validate_audio_timeline(
    audio_duration_s: float,
    timeline_duration_ms: int,
    *,
    tolerance_s: float = 1.0,
) -> None:
    """Reject an already-misaligned final audio track before a long render."""

    expected_s = timeline_duration_ms / 1000.0
    drift_s = audio_duration_s - expected_s
    if abs(drift_s) > tolerance_s:
        raise ValueError(
            "最终音轨与 TTS 片段时间轴不一致: "
            f"audio={audio_duration_s:.3f}s, timeline={expected_s:.3f}s, "
            f"drift={drift_s:+.3f}s。请重新生成 BGM 混音后再合成视频。"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be positive")
        return parsed

    parser = argparse.ArgumentParser(description="从插图和音频生成带中文字幕的视频")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH, help="插图计划 JSON")
    parser.add_argument(
        "--illustrations-dir",
        type=Path,
        default=ILLUSTRATIONS_DIR,
        help="插图目录",
    )
    parser.add_argument("--audio", type=Path, default=AUDIO_PATH, help="最终混音音轨")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出视频")
    parser.add_argument("--novel", type=Path, default=NOVEL_PATH, help="小说原文")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="说话人标注")
    parser.add_argument("--segments-dir", type=Path, default=SEGMENTS_DIR, help="TTS WAV 片段目录")
    parser.add_argument("--subtitle-output", type=Path, default=SUBTITLE_PATH, help="生成的 SRT 文件")
    parser.add_argument("--subtitle-font", default="SimHei", help="字幕字体 family name")
    parser.add_argument("--subtitle-font-size", type=positive_int, default=20, help="字幕字号")
    parser.add_argument("--subtitle-fonts-dir", type=Path, help="额外字体目录")
    parser.add_argument(
        "--subtitle-label-mode",
        choices=("auto", "line", "parsed-line", "dialogue"),
        default="auto",
        help="说话人标签布局（默认按 WAV 数量自动判断）",
    )
    parser.add_argument("--max-subtitle-chars", type=positive_int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-subtitle-lines", type=positive_int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--no-subtitles", action="store_true", help="生成不带字幕的视频")
    return parser.parse_args(argv)


def _ffconcat_quote(path: Path) -> str:
    """Quote an absolute path for an ffconcat file (not for a shell)."""

    value = path.resolve().as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError(f"插图路径不能包含换行符: {path}")
    # FFmpeg's lexer cannot represent a quote inside a quoted section.  Close
    # the section, escape the quote, then reopen it: 'O'\''Brien.png'.
    return "'" + value.replace("'", "'\\''") + "'"


def write_slideshow_concat(
    timeline: list[dict],
    illustrations: list[dict],
    output_path: Path,
) -> int:
    """Write an exact-duration ffconcat slideshow and return its length in ms."""

    if not timeline:
        raise ValueError("插图时间轴为空")

    lines: list[str] = []
    total_duration_ms = 0
    last_path: Path | None = None
    for item in timeline:
        image_index = int(item["image_idx"])
        if not 0 <= image_index < len(illustrations):
            raise ValueError(f"无效的插图索引: {image_index}")
        image_value = illustrations[image_index].get("image_path", "")
        image_path = Path(image_value) if image_value else None
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"时间轴缺少插图文件: {image_value or image_index}")

        duration_ms = int(item["duration_ms"])
        if duration_ms <= 0:
            raise ValueError(f"插图时长必须为正数: {duration_ms}ms")
        quoted_path = _ffconcat_quote(image_path)
        lines.extend([f"file {quoted_path}", f"duration {duration_ms / 1000:.3f}"])
        total_duration_ms += duration_ms
        last_path = image_path

    # The concat demuxer ignores the final duration without a following file.
    assert last_path is not None
    lines.append(f"file {_ffconcat_quote(last_path)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return total_duration_ms


def build_illustration_timeline(
    illustrations: list[dict],
    line_timestamps: list[int],
    *,
    source_line_timeline: dict[int, dict[str, int]] | None = None,
    total_duration_ms: int | None = None,
) -> list[dict]:
    """Map raw novel line ranges to a continuous, exact-duration slideshow."""

    if not illustrations:
        return []

    timeline: list[dict] = []
    source_lines = sorted(source_line_timeline or {})
    exact_timeline = total_duration_ms is not None and total_duration_ms > 0
    for p in illustrations:
        sl = int(p.get("start_line", 1))
        el = int(p.get("end_line", sl))
        if sl <= 0 or el < sl:
            raise ValueError(f"无效的插图行号范围: {sl}..{el}")
        title = p.get("title", "?")

        if source_lines:
            # Illustration plans use raw novel line numbers, while WAV files
            # use parsed-dialogue order.  Pick the first spoken source line in
            # the requested range (or the nearest following one for headings).
            candidates = [line for line in source_lines if line >= sl]
            chosen_line = candidates[0] if candidates else source_lines[-1]
            start_ms = source_line_timeline[chosen_line]["start_ms"]
            # The exact end is assigned from the next illustration below.  It
            # keeps uncovered lines and inter-segment silence on the current
            # image without introducing cumulative drift.
            end_ms = start_ms
        else:
            # Backward-compatible fallback for callers without source lines.
            if not line_timestamps:
                raise ValueError("音频片段时间轴为空")
            start_ms = line_timestamps[min(sl - 1, len(line_timestamps) - 1)]
            if exact_timeline:
                end_ms = start_ms
            else:
                end_ms = (
                    line_timestamps[el - 1]
                    if el <= len(line_timestamps)
                    else start_ms + 3000
                )
                if el < len(line_timestamps):
                    end_ms += 300

        timeline.append({
            "title": title,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "image_idx": p.get("image_idx", 0),
        })

    if source_lines and not exact_timeline:
        raise ValueError("需要有效的总音频时长来构建插图时间轴")

    if exact_timeline:
        starts = [int(item["start_ms"]) for item in timeline]
        if starts != sorted(starts):
            raise ValueError("插图计划必须按 start_line 递增排列")
        if starts[-1] >= int(total_duration_ms):
            raise ValueError("最后一张插图没有可用的显示时长")

        # Several headings can resolve to the same first spoken source line.
        # Split that real interval among them instead of stretching each one to
        # an arbitrary minimum and shifting everything that follows.
        index = 0
        while index < len(timeline):
            group_end_index = index + 1
            while (
                group_end_index < len(timeline)
                and starts[group_end_index] == starts[index]
            ):
                group_end_index += 1

            group_start_ms = 0 if index == 0 else starts[index]
            group_end_ms = (
                starts[group_end_index]
                if group_end_index < len(timeline)
                else int(total_duration_ms)
            )
            group_size = group_end_index - index
            span_ms = group_end_ms - group_start_ms
            if span_ms < group_size:
                raise ValueError("插图区间太短，无法无损分配到时间轴")

            for offset, item_index in enumerate(range(index, group_end_index)):
                start_ms = group_start_ms + span_ms * offset // group_size
                end_ms = group_start_ms + span_ms * (offset + 1) // group_size
                timeline[item_index]["start_ms"] = start_ms
                timeline[item_index]["end_ms"] = end_ms
                timeline[item_index]["duration_ms"] = end_ms - start_ms
            index = group_end_index
    else:
        for item in timeline:
            if item["end_ms"] <= item["start_ms"]:
                raise ValueError(f"插图 {item['title']!r} 没有可用的显示时长")

    return timeline


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    print("=" * 50)
    print("插图视频生成（自动中文字幕）")
    print("=" * 50)

    # 1. 加载数据
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"错误: 无法读取插图计划: {exc}")
        return 1
    illustrations = plan.get("illustrations", plan) if isinstance(plan, dict) else plan
    if not isinstance(illustrations, list) or not illustrations:
        print("错误: 插图计划中没有 illustrations 列表")
        return 1
    if not all(isinstance(item, dict) for item in illustrations):
        print("错误: 插图计划的每一项都必须是对象")
        return 1
    print(f"插图: {len(illustrations)} 张")

    # 2. 检查插图文件
    img_files = sorted(args.illustrations_dir.glob("*.png"))
    print(f"插图文件: {len(img_files)} 个")
    if not img_files:
        print("错误: 没有找到插图文件")
        return 1

    # 建立 插图索引 -> 文件路径 映射
    for i, p in enumerate(illustrations):
        p["image_idx"] = i
        # 尝试按序号匹配
        expected = args.illustrations_dir / f"{i+1:04d}_{p.get('title','?')}.png"
        if expected.is_file():
            p["image_path"] = str(expected)
        else:
            # fallback: 按顺序匹配
            p["image_path"] = str(img_files[i]) if i < len(img_files) else ""
    missing_images = [
        index + 1
        for index, item in enumerate(illustrations)
        if not item.get("image_path") or not Path(item["image_path"]).is_file()
    ]
    if missing_images:
        preview = ", ".join(str(index) for index in missing_images[:10])
        print(f"错误: 插图计划缺少文件（序号: {preview}）")
        return 1

    # 3. 计算时间戳
    print("\n计算时间戳...")
    subtitle_filter = None
    audio_dur_s: float | None = None
    timeline_matches_segments = False
    segment_count = len(discover_segment_files(args.segments_dir))
    if segment_count == 0:
        print(f"错误: 没有找到音频片段: {args.segments_dir}")
        return 1

    if args.no_subtitles:
        # Preserve the old no-subtitle workflow: novel/labels are optional.
        # When a novel is available, parse it without speaker labels solely to
        # recover accurate raw source-line and chapter timing.
        subtitle_timestamps = None
        if args.novel.is_file():
            try:
                timing_entries = load_subtitle_entries(
                    args.novel,
                    None,
                    expected_segment_count=segment_count,
                )
                subtitle_timestamps = build_segment_timestamps(
                    timing_entries,
                    args.segments_dir,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"  提示: 原文时间轴不可用，回退到 WAV 顺序: {exc}")

        if subtitle_timestamps is not None:
            line_ts = [item["start_ms"] for item in subtitle_timestamps]
            source_line_ts = build_source_line_timeline(subtitle_timestamps)
            total_duration_ms = subtitle_timestamps[-1]["end_ms"]
            timeline_matches_segments = True
        else:
            try:
                line_ts = build_line_timestamps(args.segments_dir)
                audio_dur_s = probe_media_duration(args.audio)
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"错误: 无法构建无字幕时间轴: {exc}")
                return 1
            source_line_ts = None
            total_duration_ms = round(audio_dur_s * 1000)
    else:
        try:
            print("  加载小说和说话人标注...")
            subtitle_entries = load_subtitle_entries(
                args.novel,
                args.labels,
                label_mode=args.subtitle_label_mode,
                expected_segment_count=segment_count,
            )
            subtitle_timestamps = build_segment_timestamps(
                subtitle_entries,
                args.segments_dir,
            )
            write_srt(
                args.subtitle_output,
                subtitle_entries,
                subtitle_timestamps,
                max_chars=args.max_subtitle_chars,
                max_lines=args.max_subtitle_lines,
            )
            subtitle_filter = build_subtitle_filter(
                args.subtitle_output,
                font_name=args.subtitle_font,
                font_size=args.subtitle_font_size,
                fonts_dir=args.subtitle_fonts_dir,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"错误: 无法构建音频/字幕时间轴: {exc}")
            return 1

        line_ts = [timestamp["start_ms"] for timestamp in subtitle_timestamps]
        source_line_ts = build_source_line_timeline(subtitle_timestamps)
        total_duration_ms = subtitle_timestamps[-1]["end_ms"]
        timeline_matches_segments = True
        cue_count = sum(
            1
            for line in args.subtitle_output.read_text(encoding="utf-8").splitlines()
            if " --> " in line
        )
        print(f"  字幕: {len(subtitle_entries)} 个音频条目 -> {cue_count} 条")
        print(f"  SRT: {args.subtitle_output}")

    print(f"音频总时长: {total_duration_ms / 1000 / 60:.0f} 分钟")

    # 4. 构建插图时间轴
    try:
        timeline = build_illustration_timeline(
            illustrations,
            line_ts,
            source_line_timeline=source_line_ts,
            total_duration_ms=total_duration_ms,
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"错误: 无法构建插图时间轴: {exc}")
        return 1
    print(f"时间轴条目: {len(timeline)}")

    # 5. 生成 ffmpeg concat 文件
    print("\n生成视频...")
    try:
        if audio_dur_s is None:
            audio_dur_s = probe_media_duration(args.audio)
        if timeline_matches_segments:
            validate_audio_timeline(audio_dur_s, total_duration_ms)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}")
        return 1

    print(f"音频时长: {audio_dur_s/60:.0f} 分钟")

    # 生成 concat 文件。每一段使用真实毫秒时长，禁止用最短显示时间
    # 拉长画面，否则所有后续插图都会逐步偏离音频。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = (args.output.parent / "_video_concat.txt").resolve()
    try:
        total_video_ms = write_slideshow_concat(timeline, illustrations, concat_file)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"错误: 无法生成 concat 文件: {exc}")
        return 1
    if total_video_ms != total_duration_ms:
        print(
            "错误: 插图时间轴未覆盖完整音频: "
            f"video={total_video_ms}ms, audio={total_duration_ms}ms"
        )
        return 1
    print(f"  插图总长: {total_video_ms / 1000 / 60:.0f} 分")

    print(f"  concat 文件: {concat_file}")

    # 6. 执行 ffmpeg
    # The concat slideshow contains one sparse frame per illustration.  Expand
    # it to CFR *before* subtitles so libass can update cues while one still
    # image remains on screen.
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-i", str(args.audio.resolve()),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", build_video_filter_chain(subtitle_filter),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(args.output),
    ]

    print(f"  输出: {args.output}")
    print(f"  命令: {' '.join(cmd)}")
    print("\n渲染中...")

    t0 = time.time()
    try:
        result = subprocess.run(cmd)
    except OSError as exc:
        print(f"\n错误: 无法启动 ffmpeg: {exc}")
        return 1
    elapsed = time.time() - t0

    if result.returncode == 0:
        size_mb = args.output.stat().st_size / 1024 / 1024
        print(f"\n完成! ({elapsed:.0f} 秒, {size_mb:.0f} MB)")
        print(f"输出: {args.output}")
    else:
        print(f"\n错误: ffmpeg 返回 {result.returncode}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
