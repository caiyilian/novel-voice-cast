"""
音频拼接引擎 — 按原文顺序拼接音频片段。

功能：
1. 按原文顺序拼接音频
2. 对话间间隔 0.3s，段落间隔 1s，章节间隔 2s
3. 淡入淡出效果（每段 50ms）
4. 逐章节拆分输出
"""
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pydub import AudioSegment

from .timeline import (
    FADE_DURATION,
    GAP_CHAPTER,
    GAP_DIALOGUE,
    GAP_PARAGRAPH,
    gap_between_segments,
)


FFMPEG_CONCAT_TIMEOUT_SECONDS = 2 * 60 * 60


@dataclass(frozen=True)
class SpliceFileResult:
    """Lightweight result returned when a splice is streamed to disk.

    Keeping only the duration and path avoids loading a multi-hour output back
    into an ``AudioSegment`` merely so callers can inspect its length. This is
    intentionally not a drop-in ``AudioSegment``: callers that need slicing,
    levels, or other sample operations must omit ``output_path`` and should do
    so only for inputs small enough to materialise safely in memory.
    """

    path: Path
    duration_ms: int

    def __len__(self) -> int:
        return self.duration_ms


def _ffconcat_path(path: Path) -> str:
    """Quote an absolute path for an ffconcat list file."""
    value = path.resolve().as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError(f"Audio path contains a newline and cannot be concatenated: {path}")
    return "'" + value.replace("'", "'\\''") + "'"


def probe_audio_duration(path: Path) -> float:
    """Return the encoded file duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[:500]}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned an invalid duration for {path}: {result.stdout!r}") from exc
    if duration < 0:
        raise RuntimeError(f"ffprobe returned a negative duration for {path}: {duration}")
    return duration


def concat_wav_files_ffmpeg(
    wav_files: Sequence[Path],
    output_path: Path,
    *,
    output_format: str = "wav",
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    sample_width: Optional[int] = None,
    bitrate: Optional[str] = None,
    expected_duration_seconds: Optional[float] = None,
    duration_tolerance_seconds: float = 0.1,
) -> float:
    """Concatenate WAV parts with ffmpeg's concat demuxer.

    ffmpeg reads each part sequentially and encodes the destination once, so
    memory use is independent of the complete programme duration. The result
    is written beside the destination and atomically replaced only after both
    ffmpeg and ffprobe succeed.
    """
    wav_files = [Path(path) for path in wav_files]
    if not wav_files:
        raise ValueError("At least one WAV file is required for concatenation")

    missing = [str(path) for path in wav_files if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"Audio parts missing before concatenation: {preview}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix="ffconcat_", dir=str(output_path.parent)))
    temp_output = temp_parent / f"result{output_path.suffix or '.' + output_format}"
    concat_list = temp_parent / "inputs.ffconcat"

    try:
        lines = ["ffconcat version 1.0"]
        lines.extend(f"file {_ffconcat_path(path)}" for path in wav_files)
        concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if output_format == "wav":
            pcm_codecs = {1: "pcm_u8", 2: "pcm_s16le", 3: "pcm_s24le", 4: "pcm_s32le"}
            try:
                pcm_codec = pcm_codecs[sample_width or 2]
            except KeyError as exc:
                raise ValueError(f"Unsupported PCM sample width: {sample_width}") from exc
            codec_args = ["-c:a", pcm_codec, "-rf64", "auto", "-f", "wav"]
        elif output_format == "mp3":
            codec_args = ["-c:a", "libmp3lame", "-b:a", bitrate or "192k", "-f", "mp3"]
        else:
            raise ValueError(f"Unsupported concat output format: {output_format}")

        conversion_args: List[str] = []
        if sample_rate is not None:
            conversion_args.extend(["-ar", str(sample_rate)])
        if channels is not None:
            conversion_args.extend(["-ac", str(channels)])

        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-map", "0:a:0",
                "-vn",
                *codec_args,
                *conversion_args,
                str(temp_output),
            ],
            capture_output=True,
            text=True,
            timeout=FFMPEG_CONCAT_TIMEOUT_SECONDS,
        )
        if result.returncode != 0 or not temp_output.is_file() or temp_output.stat().st_size < 44:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()[:1000]}")

        duration = probe_audio_duration(temp_output)
        if (
            expected_duration_seconds is not None
            and abs(duration - expected_duration_seconds) > duration_tolerance_seconds
        ):
            raise RuntimeError(
                "Concatenated audio duration mismatch: "
                f"expected {expected_duration_seconds:.3f}s, got {duration:.3f}s"
            )
        os.replace(temp_output, output_path)
        return duration
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


# ─── Audio Splicer ─────────────────────────────────────────────────

class AudioSplicer:
    """音频拼接引擎"""

    def __init__(
        self,
        gap_dialogue: int = GAP_DIALOGUE,
        gap_paragraph: int = GAP_PARAGRAPH,
        gap_chapter: int = GAP_CHAPTER,
        fade_duration: int = FADE_DURATION,
        output_bitrate: str = "96k",
    ):
        self.gap_dialogue = gap_dialogue
        self.gap_paragraph = gap_paragraph
        self.gap_chapter = gap_chapter
        self.fade_duration = fade_duration
        self.output_bitrate = output_bitrate

    def _create_silence(self, duration_ms: int) -> AudioSegment:
        """创建静音片段"""
        return AudioSegment.silent(duration=duration_ms)

    def _apply_fade(self, audio: AudioSegment) -> AudioSegment:
        """应用淡入淡出效果"""
        if len(audio) < self.fade_duration * 2:
            return audio
        return audio.fade_in(self.fade_duration).fade_out(self.fade_duration)

    def splice(
        self,
        segments: List[Dict],
        output_path: Optional[str] = None,
    ) -> Union[AudioSegment, SpliceFileResult]:
        """拼接音频片段。

        Args:
            segments: [{"audio_path": str, "chapter": str, "order": int}]
            output_path: 输出文件路径（可选）

        Returns:
            未指定输出路径时返回拼接后的 ``AudioSegment``；指定输出路径
            时以流式方式写盘并返回仅包含路径和时长的 ``SpliceFileResult``。
            后者支持 ``len(result)``，但不提供 ``AudioSegment`` 的采样操作。
        """
        if not segments:
            return AudioSegment.empty()

        # 按 order 排序
        sorted_segments = sorted(segments, key=lambda x: x.get("order", 0))

        if output_path:
            return self._splice_to_file(sorted_segments, Path(output_path))

        # The no-output API must materialise an AudioSegment. Build a list of
        # pieces and join their raw data once; repeated ``result += piece``
        # copies the complete prefix on every iteration (quadratic work).
        pieces: List[AudioSegment] = []
        for index, seg in enumerate(sorted_segments):
            try:
                audio = AudioSegment.from_file(seg["audio_path"])
            except Exception as e:
                raise RuntimeError(f"Failed to load audio segment {seg['audio_path']}: {e}") from e
            pieces.append(self._apply_fade(audio))
            if index < len(sorted_segments) - 1:
                gap = gap_between_segments(
                    seg,
                    sorted_segments[index + 1],
                    gap_dialogue=self.gap_dialogue,
                    gap_paragraph=self.gap_paragraph,
                    gap_chapter=self.gap_chapter,
                )
                pieces.append(self._create_silence(gap))

        synced = AudioSegment._sync(*pieces)
        return synced[0]._spawn(b"".join(piece.raw_data for piece in synced))

    def _splice_to_file(
        self,
        sorted_segments: List[Dict],
        output_path: Path,
    ) -> SpliceFileResult:
        """Write a splice without retaining the complete result in memory."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix in ("", ".wav", ".wave"):
            output_format = "wav"
            duration_tolerance = 0.01
        elif suffix == ".mp3":
            output_format = "mp3"
            # MP3 containers include encoder delay/padding even though
            # decoders use LAME metadata for gapless playback.
            duration_tolerance = 0.1
        else:
            raise ValueError(
                f"Unsupported splice output extension {output_path.suffix!r}; "
                "use .wav or .mp3"
            )

        # Match pydub's append semantics: all pieces are converted to the
        # greatest channel count, frame rate, and sample width in the input.
        target_channels = 0
        target_frame_rate = 0
        target_sample_width = 0
        for seg in sorted_segments:
            try:
                audio = AudioSegment.from_file(seg["audio_path"])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load audio segment {seg['audio_path']}: {exc}"
                ) from exc
            target_channels = max(target_channels, audio.channels)
            target_frame_rate = max(target_frame_rate, audio.frame_rate)
            target_sample_width = max(target_sample_width, audio.sample_width)

        tmpdir = Path(tempfile.mkdtemp(prefix="audio_splice_", dir=str(output_path.parent)))
        part_files: List[Path] = []
        expected_total_frames = 0
        try:
            for index, seg in enumerate(sorted_segments):
                try:
                    audio = AudioSegment.from_file(seg["audio_path"])
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load audio segment {seg['audio_path']}: {exc}"
                    ) from exc

                audio = self._apply_fade(audio)
                audio = (
                    audio.set_channels(target_channels)
                    .set_frame_rate(target_frame_rate)
                    .set_sample_width(target_sample_width)
                )

                if index < len(sorted_segments) - 1:
                    gap = gap_between_segments(
                        seg,
                        sorted_segments[index + 1],
                        gap_dialogue=self.gap_dialogue,
                        gap_paragraph=self.gap_paragraph,
                        gap_chapter=self.gap_chapter,
                    )
                    silence = (
                        AudioSegment.silent(duration=gap, frame_rate=target_frame_rate)
                        .set_channels(target_channels)
                        .set_sample_width(target_sample_width)
                    )
                    audio = audio + silence

                part_path = tmpdir / f"part_{index:05d}.wav"
                audio.export(part_path, format="wav")
                part_files.append(part_path)
                expected_total_frames += int(audio.frame_count())

            expected_duration = expected_total_frames / target_frame_rate
            concat_wav_files_ffmpeg(
                part_files,
                output_path,
                output_format=output_format,
                sample_rate=target_frame_rate,
                channels=target_channels,
                sample_width=target_sample_width,
                bitrate=self.output_bitrate,
                expected_duration_seconds=expected_duration,
                duration_tolerance_seconds=duration_tolerance,
            )
            return SpliceFileResult(
                path=output_path,
                # Match the old in-memory return value: expose programme
                # duration, excluding MP3 encoder padding reported by ffprobe.
                duration_ms=round(expected_duration * 1000),
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def splice_by_chapter(
        self,
        segments: List[Dict],
        output_dir: str,
        filename_pattern: str = "chapter_{index}.wav",
    ) -> List[str]:
        """按章节拆分输出。

        Args:
            segments: [{"audio_path": str, "chapter": str, "order": int}]
            output_dir: 输出目录
            filename_pattern: 文件名模式

        Returns:
            输出文件路径列表
        """
        # 按章节分组
        chapters: Dict[str, List[Dict]] = {}
        for seg in segments:
            chapter = seg.get("chapter", "")
            if chapter not in chapters:
                chapters[chapter] = []
            chapters[chapter].append(seg)

        # 按章节顺序拼接
        output_files = []
        os.makedirs(output_dir, exist_ok=True)

        for i, (chapter_name, chapter_segments) in enumerate(chapters.items()):
            if not chapter_name:
                chapter_name = f"chapter_{i + 1}"

            filename = filename_pattern.format(index=i + 1, chapter=chapter_name)
            output_path = os.path.join(output_dir, filename)
            self.splice(chapter_segments, output_path=output_path)
            output_files.append(output_path)

        return output_files

    def get_duration(self, audio: Union[AudioSegment, SpliceFileResult]) -> float:
        """获取音频时长（秒）"""
        return len(audio) / 1000.0
