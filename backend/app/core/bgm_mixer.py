"""
Stage 7-d: BGM mixing engine.

Strategy (robust against timing errors):
  1. Parse the novel structure to get the dialogue-to-line mapping.
  2. Simulate the splice to get per-dialogue start/end timestamps.
  3. Group dialogues by BGM segment (from bgm_segments.json) and compute
     each segment's time interval.
  4. Extract that portion from the EXISTING speech-only file
     ``full_volume.mp3`` via ffmpeg. The input format is auto-detected so both
     legacy WAV-disguised files and genuine MP3 output are accepted.
  5. Mix the BGM clip into the extracted portion (loop/trim BGM as needed)
     via pydub, applying per-type volume and crossfade.
  6. Concatenate all mixed portions back into one MP3 with ffmpeg's concat
     demuxer, keeping memory use independent of the full programme length.

The output duration exactly matches the speech-only file.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

from app.core.parser import parse
from app.core.splicer import (
    FADE_DURATION,
    GAP_CHAPTER,
    GAP_DIALOGUE,
    GAP_PARAGRAPH,
    concat_wav_files_ffmpeg,
    probe_audio_duration,
)
from app.core.timeline import build_contiguous_intervals, gap_between_segments

# ─── Defaults ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BGM_DIR = PROJECT_ROOT / "output/bgm"
DEFAULT_MANIFEST_PATH = DEFAULT_BGM_DIR / "bgm_manifest.json"
DEFAULT_SEGMENTS_PATH = PROJECT_ROOT / "backend/data/bgm_segments.json"
DEFAULT_SPEECH_PATH = PROJECT_ROOT / "output/full_volume.mp3"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output/full_volume_bgm.mp3"

CROSSFADE_MS = 3000
BGM_MIX_CHUNK_MS = 5 * 60 * 1000
SPEECH_TIMELINE_TOLERANCE_SECONDS = 1.0
MIX_SAMPLE_RATE = 44100
MIX_CHANNELS = 2
MIX_SAMPLE_WIDTH = 2

BGM_VOLUME_MAP: Dict[str, float] = {
    "suspense": -11.0,  # reduced by 3dB from -8.0
    "daily": -13.0,     # -3 from -10
    "battle": -9.0,     # -3 from -6
    "sad": -12.0,       # -3 from -9
    "romantic": -13.0,  # -3 from -10
    "epic": -9.0,       # -3 from -6
    "comedy": -12.0,    # -3 from -9
    "horror": -10.0,    # -3 from -7
    "unknown": -11.0,   # -3 from -8
}


def load_config(config_path: str = "config/config.yaml") -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_dialogue_line_map(config: dict) -> Tuple[List[int], List[str]]:
    novel_path = config["novel"]["text_path"]
    labels_path = config["novel"]["labels_path"]
    with open(novel_path, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    dialogues, _ = parse(novel_text, labels)
    lines = [d.get("line", 0) for d in dialogues]
    chapters = [d.get("chapter", "") for d in dialogues]
    return lines, chapters


def _calc_dialogue_duration(wav: Path) -> int:
    """Return dialogue WAV duration in ms (fast, no full load)."""
    try:
        seg = AudioSegment.from_file(str(wav))
        return len(seg)
    except Exception as exc:
        raise RuntimeError(f"Cannot read dialogue segment duration: {wav}") from exc


def _compute_dialogue_timestamps(
    n: int, chapters: List[str], segments_dir: Path,
) -> Tuple[List[int], List[int]]:
    """Return (start_ms, end_ms) per dialogue index (same order as AudioSplicer)."""
    starts: List[int] = []
    ends: List[int] = []
    offset = 0
    for i in range(n):
        starts.append(offset)
        dur = _calc_dialogue_duration(segments_dir / f"{i:05d}.wav")
        ends.append(offset + dur)
        if i < n - 1:
            gap = gap_between_segments(
                {"chapter": chapters[i]},
                {"chapter": chapters[i + 1]},
                gap_dialogue=GAP_DIALOGUE,
                gap_paragraph=GAP_PARAGRAPH,
                gap_chapter=GAP_CHAPTER,
            )
            offset += dur + gap
        else:
            offset += dur
    return starts, ends


def _extract_audio_segment_ffmpeg(
    src: Path, start_ms: float, end_ms: float, dst: Path,
) -> None:
    """Extract a portion of audio via ffmpeg. Raises on failure."""
    start_s = start_ms / 1000.0
    dur_s = (end_ms - start_ms) / 1000.0
    if dur_s <= 0:
        raise ValueError(f"invalid audio extraction interval: {start_ms}..{end_ms}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{dur_s:.3f}",
        "-f", "wav",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        str(dst),
    ], capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
        raise RuntimeError(
            f"ffmpeg extract failed [{start_s:.1f}s-{start_s + dur_s:.1f}s]: "
            f"{result.stderr[:500]}"
        )


def _prepare_bgm(
    bgm_clips: List[AudioSegment],
    target_ms: int,
    *,
    offset_ms: int = 0,
) -> AudioSegment:
    """Build a BGM track by cycling through multiple clips for variety.

    Each clip in ``bgm_clips`` is ~30s. For a long segment (e.g. 600s),
    this cycles through the available clips so the same loop isn't
    repeated monotonously.
    """
    if target_ms < 0:
        raise ValueError(f"target BGM duration must be non-negative: {target_ms}")
    if offset_ms < 0:
        raise ValueError(f"BGM offset must be non-negative: {offset_ms}")

    bgm_clips = [clip for clip in bgm_clips if len(clip) > 0]
    if not bgm_clips:
        return AudioSegment.silent(duration=target_ms)
    if target_ms == 0:
        return bgm_clips[0][:0]

    if len(bgm_clips) == 1:
        cycle = bgm_clips[0]
    else:
        # Build one short cycle in a single linear copy. The caller processes
        # long programme intervals in bounded chunks, so the repetition below
        # never grows with the complete interval or novel duration.
        synced_clips = AudioSegment._sync(*bgm_clips)
        cycle = synced_clips[0]._spawn(b"".join(clip.raw_data for clip in synced_clips))

    # Work in frames rather than rounded millisecond lengths. MP3-decoded clips
    # can have a fractional-millisecond tail; frame arithmetic prevents a tiny
    # phase jump from accumulating at each processing-chunk boundary.
    cycle_frames = int(cycle.frame_count())
    offset_frames = int(cycle.frame_count(ms=offset_ms)) % cycle_frames
    target_frames = int(cycle.frame_count(ms=target_ms))
    required_frames = offset_frames + target_frames
    repeats = max(1, (required_frames + cycle_frames - 1) // cycle_frames)
    start_byte = offset_frames * cycle.frame_width
    end_byte = (offset_frames + target_frames) * cycle.frame_width
    return cycle._spawn((cycle.raw_data * repeats)[start_byte:end_byte])


def _iter_interval_chunks(
    start_ms: int,
    end_ms: int,
    *,
    chunk_ms: int = BGM_MIX_CHUNK_MS,
) -> List[Tuple[int, int]]:
    """Split an interval into adjacent, bounded chunks."""
    if end_ms <= start_ms:
        raise ValueError(f"invalid BGM interval: {start_ms}..{end_ms}")
    if chunk_ms <= 0:
        raise ValueError(f"BGM chunk duration must be positive: {chunk_ms}")

    return [
        (chunk_start, min(chunk_start + chunk_ms, end_ms))
        for chunk_start in range(start_ms, end_ms, chunk_ms)
    ]


def _validate_speech_timeline(
    speech_path: Path,
    expected_duration_seconds: float,
    *,
    tolerance_seconds: float = SPEECH_TIMELINE_TOLERANCE_SECONDS,
) -> float:
    """Ensure the speech file and its segment-derived timeline agree."""
    if tolerance_seconds < 0:
        raise ValueError("speech timeline tolerance must be non-negative")
    actual_duration = probe_audio_duration(speech_path)
    drift = abs(actual_duration - expected_duration_seconds)
    if drift > tolerance_seconds:
        raise RuntimeError(
            "Speech audio duration does not match the WAV segment timeline: "
            f"speech={actual_duration:.3f}s, timeline={expected_duration_seconds:.3f}s, "
            f"drift={drift:.3f}s (allowed {tolerance_seconds:.3f}s). "
            "Regenerate the spliced speech audio before mixing BGM."
        )
    return actual_duration


def _mix_audio_segment(
    speech_path: Path, bgm_clips: Optional[List[AudioSegment]],
    bgm_type: str, prev_bgm_type: Optional[str], output_path: Path,
    volume_db: float = -8.0,
    *,
    bgm_offset_ms: int = 0,
    fade_in_at_start: bool = True,
    fade_out_at_end: bool = True,
) -> None:
    """Mix BGM into a speech segment and export as WAV.

    Args:
        speech_path: Extracted speech portion (WAV).
        bgm_clips: List of BGM clips to cycle through (or None for no BGM).
        bgm_type: Current BGM type label.
        prev_bgm_type: Previous segment's BGM type (for crossfade detection).
        output_path: Where to save the mixed segment (WAV).
        volume_db: Global volume baseline from config; per-type adjustment is added on top.
        bgm_offset_ms: Position in the original BGM interval, preserving the
            loop when a long interval is processed in several chunks.
        fade_in_at_start: Allow the existing BGM fade-in at this chunk start.
        fade_out_at_end: Apply the existing BGM and speech fade-outs at this
            chunk end. Internal chunk boundaries disable both fades.
    """
    try:
        speech = AudioSegment.from_file(str(speech_path))
    except Exception as exc:
        raise RuntimeError(f"Cannot load extracted speech segment: {speech_path}") from exc

    if bgm_clips:
        dur = len(speech)
        bgm = _prepare_bgm(bgm_clips, dur, offset_ms=bgm_offset_ms)

        # Volume: config baseline + per-type adjustment
        type_adj = BGM_VOLUME_MAP.get(bgm_type, -8.0)  # already lowered
        bgm = bgm.apply_gain(volume_db + type_adj)

        # Fade-in at start (first segment or when type changes)
        if fade_in_at_start and (prev_bgm_type is None or bgm_type != prev_bgm_type):
            fade_in_ms = min(CROSSFADE_MS, dur // 4)
            if fade_in_ms > 0 and len(bgm) > fade_in_ms:
                bgm = bgm.fade_in(fade_in_ms)

        # Fade-out at end (always, for smooth transition between segments)
        if fade_out_at_end:
            fade_out_ms = min(CROSSFADE_MS, dur // 4)
            if fade_out_ms > 0 and len(bgm) > fade_out_ms:
                bgm = bgm.fade_out(fade_out_ms)

        speech = speech.overlay(bgm, position=0)

    # Speech fade-out (keep existing behavior)
    if fade_out_at_end:
        fade_len = min(FADE_DURATION, len(speech) // 2)
        if fade_len > 0:
            speech = speech.fade_out(fade_len)
    # FFmpeg's concat demuxer requires every input stream to have matching
    # parameters. Intervals without a BGM overlay would otherwise stay mono
    # while intervals with stereo BGM become stereo.
    speech = (
        speech.set_frame_rate(MIX_SAMPLE_RATE)
        .set_channels(MIX_CHANNELS)
        .set_sample_width(MIX_SAMPLE_WIDTH)
    )
    speech.export(str(output_path), format="wav")


def _concat_via_ffmpeg(
    wav_files: List[Path], output_mp3: Path, expected_duration: float,
) -> float:
    """Stream-concatenate WAVs into one MP3 and validate its timeline."""
    if output_mp3.suffix.lower() != ".mp3":
        raise ValueError(f"BGM output path must use the .mp3 extension: {output_mp3}")
    return concat_wav_files_ffmpeg(
        wav_files,
        output_mp3,
        output_format="mp3",
        sample_rate=MIX_SAMPLE_RATE,
        channels=MIX_CHANNELS,
        bitrate="192k",
        expected_duration_seconds=expected_duration,
        duration_tolerance_seconds=0.1,
    )


def mix_bgm(
    speech_path: Path = DEFAULT_SPEECH_PATH,
    bgm_dir: Path = DEFAULT_BGM_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    segments_path: Path = DEFAULT_SEGMENTS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    config_path: str = "config/config.yaml",
) -> float:
    """Mix BGM into the *already-spliced* speech file.

    Uses ``full_volume.mp3`` as source of truth for the audio timeline.
    """
    t0 = time.time()
    speech_path = Path(speech_path)
    bgm_dir = Path(bgm_dir)
    manifest_path = Path(manifest_path)
    segments_path = Path(segments_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp3":
        raise ValueError(f"BGM output path must use the .mp3 extension: {output_path}")

    config = load_config(config_path)
    segments_dir = Path(config["output"]["dir"]) / "segments"

    if not speech_path.exists():
        print(f"ERROR: speech file not found: {speech_path}")
        return 0.0
    if not segments_path.exists():
        print(f"ERROR: segments not found: {segments_path}")
        return 0.0

    bgm_segments: List[Dict] = json.loads(segments_path.read_text(encoding="utf-8"))
    dialogues_lines, chapters = _build_dialogue_line_map(config)
    n = len(dialogues_lines)

    # ── BGM manifest / cache ──
    bgm_manifest: Dict[str, Any] = {"segments": {}}
    if manifest_path.exists():
        bgm_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Determine clips_per_segment from manifest; fall back to scanning disk
    clips_per_segment = 1
    if "clips_per_segment" in bgm_manifest:
        clips_per_segment = bgm_manifest["clips_per_segment"]
    else:
        # Scan first segment's clip count from disk
        for i in range(20):
            if (bgm_dir / f"001_{i}.mp3").exists():
                clips_per_segment = max(clips_per_segment, i + 1)

    # Keep only lightweight paths here. Decoded PCM clips are loaded lazily for
    # one interval and released before the next, rather than caching the entire
    # novel's BGM set in memory.
    bgm_files: Dict[int, List[Path]] = {}
    for i, s in enumerate(bgm_segments):
        seg_idx = i  # 0-based index, matches interval building
        clips: List[Path] = []
        for ci in range(clips_per_segment):
            fp = bgm_dir / f"{seg_idx + 1:03d}_{ci}.mp3"
            if fp.exists():
                clips.append(fp)
        if clips:
            bgm_files[seg_idx] = clips
    total_clips = sum(len(c) for c in bgm_files.values())
    print(
        f"  BGM clips indexed: {total_clips} "
        f"(across {len(bgm_files)} segments, {clips_per_segment}/segment)"
    )

    # ── Compute dialogue timestamps ──
    print("  Computing dialogue timestamps...")
    starts, ends = _compute_dialogue_timestamps(n, chapters, segments_dir)
    total_speech_ms = ends[-1] if ends else 0
    print(f"  Speech duration: {total_speech_ms/1000:.0f}s = {total_speech_ms/3600000:.2f}h")
    expected_duration = total_speech_ms / 1000.0
    actual_speech_duration = _validate_speech_timeline(speech_path, expected_duration)
    print(
        "  Speech timeline verified: "
        f"file={actual_speech_duration:.3f}s, segments={expected_duration:.3f}s"
    )

    # ── Build BGM intervals (one per contiguous BGM segment) ──
    group_ids: List[int] = []
    for line in dialogues_lines:
        seg_idx: Optional[int] = None
        for si, seg in enumerate(bgm_segments):
            if seg["start_line"] <= line <= seg["end_line"]:
                seg_idx = si
                break
        if seg_idx is None:
            raise ValueError(f"BGM segment plan does not cover novel line {line}")
        group_ids.append(seg_idx)

    intervals = []
    for interval in build_contiguous_intervals(group_ids, starts, ends):
        seg_idx = int(interval["group_id"])
        intervals.append({
            "segment_index": seg_idx,
            "bgm_type": bgm_segments[seg_idx].get("bgm_type", "unknown"),
            "start_ms": interval["start_ms"],
            "end_ms": interval["end_ms"],
        })

    # ── Process each interval ──
    tmpdir = Path(tempfile.mkdtemp(prefix="bgm_mix_"))
    wav_files: List[Path] = []
    prev_bgm_type: Optional[str] = None
    report_interval = max(1, len(intervals) // 10)

    # Volume baseline from config
    bgm_config = config.get("bgm", {})
    volume_db = bgm_config.get("volume_db", -8.0)
    print(f"  BGM volume baseline: {volume_db} dB  (plus per-type adjustment)")

    try:
        print(f"  Processing {len(intervals)} BGM intervals...")
        for gi, iv in enumerate(intervals):
            seg_idx = iv["segment_index"]
            bgm_type = iv["bgm_type"]
            start_ms = iv["start_ms"]
            end_ms = iv["end_ms"]

            # Decode only the clips needed by this interval. A five-minute
            # processing bound keeps pydub's speech/BGM/overlay copies bounded,
            # while bgm_offset_ms keeps the loop sample-continuous across chunks.
            clip_paths = bgm_files.get(seg_idx, [])
            bgm_clips = [AudioSegment.from_file(str(path)) for path in clip_paths]
            interval_chunks = _iter_interval_chunks(start_ms, end_ms)
            for chunk_index, (chunk_start_ms, chunk_end_ms) in enumerate(interval_chunks):
                chunk_tag = f"{gi:04d}_{chunk_index:03d}"
                seg_wav = tmpdir / f"speech_{chunk_tag}.wav"
                _extract_audio_segment_ffmpeg(
                    speech_path,
                    chunk_start_ms,
                    chunk_end_ms,
                    seg_wav,
                )

                out_wav = tmpdir / f"mix_{chunk_tag}.wav"
                try:
                    _mix_audio_segment(
                        seg_wav,
                        bgm_clips or None,
                        bgm_type,
                        prev_bgm_type,
                        out_wav,
                        volume_db=volume_db,
                        bgm_offset_ms=chunk_start_ms - start_ms,
                        fade_in_at_start=chunk_index == 0,
                        fade_out_at_end=chunk_index == len(interval_chunks) - 1,
                    )
                finally:
                    seg_wav.unlink(missing_ok=True)
                wav_files.append(out_wav)

            prev_bgm_type = bgm_type

            if (gi + 1) % report_interval == 0 or gi == len(intervals) - 1:
                dur_s = (end_ms - start_ms) / 1000
                clip_count = len(clip_paths)
                print(
                    f"  [{gi+1:3d}/{len(intervals)}] {bgm_type:12s} | "
                    f"{start_ms/1000:>7.1f}s - {end_ms/1000:>7.1f}s "
                    f"({dur_s:7.1f}s, {clip_count} clips, {len(interval_chunks)} chunks)"
                )
            del bgm_clips

        # ── Concatenate ──
        print(f"\n  Concatenating {len(wav_files)} segments via ffmpeg...")
        duration = _concat_via_ffmpeg(wav_files, output_path, expected_duration)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.time() - t0
    file_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
    print(f"\n{'=' * 60}")
    print(f"Done! {duration:.1f}s ({file_mb:.0f} MB)")
    print(f"Output: {output_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")
    return duration


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 7-d: mix BGM into speech")
    p.add_argument("--speech", default=str(DEFAULT_SPEECH_PATH))
    p.add_argument("--bgm-dir", default=str(DEFAULT_BGM_DIR))
    p.add_argument("--segments", default=str(DEFAULT_SEGMENTS_PATH))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()
    mix_bgm(speech_path=Path(args.speech), bgm_dir=Path(args.bgm_dir),
            manifest_path=Path(args.manifest), segments_path=Path(args.segments),
            output_path=Path(args.output), config_path=args.config)
