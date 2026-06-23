"""
Stage 7-d: BGM mixing engine.

Strategy (robust against timing errors):
  1. Parse the novel structure to get the dialogue-to-line mapping.
  2. Simulate the splice to get per-dialogue start/end timestamps.
  3. Group dialogues by BGM segment (from bgm_segments.json) and compute
     each segment's time interval.
  4. Extract that portion from the EXISTING speech-only file
     ``full_volume.mp3`` (a WAV in disguise — produced by AudioSplicer,
     proven correct at ~6 h) via ffmpeg (fast seek on PCM).
  5. Mix the BGM clip into the extracted portion (loop/trim BGM as needed)
     via pydub, applying per-type volume and crossfade.
  6. Concatenate all mixed portions back into one MP3 via pydub.

The output duration exactly matches the speech-only file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

from app.core.parser import parse
from app.core.splicer import GAP_DIALOGUE, GAP_PARAGRAPH, GAP_CHAPTER, FADE_DURATION

# ─── Defaults ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BGM_DIR = PROJECT_ROOT / "output/bgm"
DEFAULT_MANIFEST_PATH = DEFAULT_BGM_DIR / "bgm_manifest.json"
DEFAULT_SEGMENTS_PATH = PROJECT_ROOT / "backend/data/bgm_segments.json"
DEFAULT_SPEECH_PATH = PROJECT_ROOT / "output/full_volume.mp3"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output/full_volume_bgm.mp3"

CROSSFADE_MS = 3000

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
    except Exception:
        return 1000


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
        # Gap after this dialogue (AudioSplicer applies fade *after* gap… but gap is between)
        fade = min(FADE_DURATION, dur // 2)
        if i < n - 1:
            prev_ch = chapters[i]
            curr_ch = chapters[i + 1]
            if curr_ch != prev_ch:
                gap = GAP_CHAPTER
            elif i > 0 and chapters[i] != chapters[i - 1]:
                gap = GAP_PARAGRAPH
            else:
                gap = GAP_DIALOGUE
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
        # Zero-length segment: write minimal silence
        AudioSegment.silent(duration=10).export(str(dst), format="wav")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "wav",  # force WAV input (file has .mp3 extension but is actually PCM)
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
        print(f"  ffmpeg extract failed [{start_s:.1f}s-{start_s+dur_s:.1f}s]: {result.stderr[:200]}")
        # Fallback: write silence of expected duration
        AudioSegment.silent(duration=int(dur_s * 1000)).export(str(dst), format="wav")


def _prepare_bgm(bgm_clips: List[AudioSegment], target_ms: int) -> AudioSegment:
    """Build a BGM track by cycling through multiple clips for variety.

    Each clip in ``bgm_clips`` is ~30s. For a long segment (e.g. 600s),
    this cycles through the available clips so the same loop isn't
    repeated monotonously.
    """
    if not bgm_clips:
        return AudioSegment.silent(duration=target_ms)
    if len(bgm_clips) == 1:
        clip = bgm_clips[0]
        repeats = max(1, (target_ms // len(clip)) + 1)
        return (clip * repeats)[:target_ms]

    # Multiple clips: cycle through them
    combined = AudioSegment.empty()
    total_unique = sum(len(c) for c in bgm_clips)
    repeats = (target_ms // total_unique) + 1
    for _ in range(repeats):
        for clip in bgm_clips:
            combined += clip
    return combined[:target_ms]


def _mix_audio_segment(
    speech_path: Path, bgm_clips: Optional[List[AudioSegment]],
    bgm_type: str, prev_bgm_type: Optional[str], output_path: Path,
    volume_db: float = -8.0,
) -> None:
    """Mix BGM into a speech segment and export as WAV.

    Args:
        speech_path: Extracted speech portion (WAV).
        bgm_clips: List of BGM clips to cycle through (or None for no BGM).
        bgm_type: Current BGM type label.
        prev_bgm_type: Previous segment's BGM type (for crossfade detection).
        output_path: Where to save the mixed segment (WAV).
        volume_db: Global volume baseline from config; per-type adjustment is added on top.
    """
    try:
        speech = AudioSegment.from_file(str(speech_path))
    except Exception:
        speech = AudioSegment.silent(duration=1000)

    if bgm_clips:
        dur = len(speech)
        bgm = _prepare_bgm(bgm_clips, dur)

        # Volume: config baseline + per-type adjustment
        type_adj = BGM_VOLUME_MAP.get(bgm_type, -8.0)  # already lowered
        bgm = bgm.apply_gain(volume_db + type_adj)

        # Fade-in at start (first segment or when type changes)
        if prev_bgm_type is None or bgm_type != prev_bgm_type:
            fade_in_ms = min(CROSSFADE_MS, dur // 4)
            if fade_in_ms > 0 and len(bgm) > fade_in_ms:
                bgm = bgm.fade_in(fade_in_ms)

        # Fade-out at end (always, for smooth transition between segments)
        fade_out_ms = min(CROSSFADE_MS, dur // 4)
        if fade_out_ms > 0 and len(bgm) > fade_out_ms:
            bgm = bgm.fade_out(fade_out_ms)

        speech = speech.overlay(bgm, position=0)

    # Speech fade-out (keep existing behavior)
    fade_len = min(FADE_DURATION, len(speech) // 2)
    if fade_len > 0:
        speech = speech.fade_out(fade_len)
    speech.export(str(output_path), format="wav")


def _concat_via_pydub(
    wav_files: List[Path], output_mp3: Path,
) -> float:
    """Concatenate WAVs using pydub (one file at a time, memory-safe)."""
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    combined = AudioSegment.empty()

    for wav in wav_files:
        seg = AudioSegment.from_file(str(wav))
        combined += seg

    combined.export(str(output_mp3), format="mp3", bitrate="192k", parameters=["-ar", "44100", "-ac", "2"])
    return len(combined) / 1000.0


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

    # Cache: seg_idx -> list of AudioSegment clips

    # Cache: seg_idx -> list of AudioSegment clips
    bgm_cache: Dict[int, List[AudioSegment]] = {}
    for i, s in enumerate(bgm_segments):
        seg_idx = i  # 0-based index, matches interval building
        clips: List[AudioSegment] = []
        for ci in range(clips_per_segment):
            fp = bgm_dir / f"{seg_idx + 1:03d}_{ci}.mp3"
            if fp.exists():
                clips.append(AudioSegment.from_file(str(fp)))
        if clips:
            bgm_cache[seg_idx] = clips
    total_clips = sum(len(c) for c in bgm_cache.values())
    print(f"  BGM clips loaded: {total_clips} (across {len(bgm_cache)} segments, {clips_per_segment}/segment)")

    # ── Compute dialogue timestamps ──
    print("  Computing dialogue timestamps...")
    starts, ends = _compute_dialogue_timestamps(n, chapters, segments_dir)
    total_speech_ms = ends[-1] if ends else 0
    print(f"  Speech duration: {total_speech_ms/1000:.0f}s = {total_speech_ms/3600000:.2f}h")

    # ── Build BGM intervals (one per BGM segment) ──
    intervals: List[Dict] = []
    current_seg_idx: Optional[int] = None
    current_start: int = 0
    last_di: int = -1

    for di, line in enumerate(dialogues_lines):
        seg_idx = None
        for si, seg in enumerate(bgm_segments):
            if seg["start_line"] <= line <= seg["end_line"]:
                seg_idx = si
                break

        if seg_idx != current_seg_idx:
            # Finalize previous interval
            if current_seg_idx is not None:
                intervals.append({
                    "segment_index": current_seg_idx,
                    "bgm_type": bgm_segments[current_seg_idx].get("bgm_type", "unknown"),
                    "start_ms": current_start,
                    "end_ms": ends[last_di] if last_di >= 0 else starts[di],
                })
            # Start new interval
            current_seg_idx = seg_idx
            current_start = starts[di]

        last_di = di

    # Final interval
    if current_seg_idx is not None:
        intervals.append({
            "segment_index": current_seg_idx,
            "bgm_type": bgm_segments[current_seg_idx].get("bgm_type", "unknown"),
            "start_ms": current_start,
            "end_ms": ends[last_di] if last_di >= 0 else 0,
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

    print(f"  Processing {len(intervals)} BGM intervals...")
    for gi, iv in enumerate(intervals):
        seg_idx = iv["segment_index"]
        bgm_type = iv["bgm_type"]
        start_ms = iv["start_ms"]
        end_ms = iv["end_ms"]

        # Extract speech portion via ffmpeg (fast, no memory overhead)
        seg_wav = tmpdir / f"speech_{gi:04d}.wav"
        _extract_audio_segment_ffmpeg(speech_path, start_ms, end_ms, seg_wav)

        # Load BGM clips (list, or None if no clips for this segment)
        bgm_clips: Optional[List[AudioSegment]] = bgm_cache.get(seg_idx) if seg_idx is not None else None

        # Mix with volume_db from config
        out_wav = tmpdir / f"mix_{gi:04d}.wav"
        _mix_audio_segment(seg_wav, bgm_clips, bgm_type, prev_bgm_type, out_wav, volume_db=volume_db)
        wav_files.append(out_wav)

        prev_bgm_type = bgm_type

        if (gi + 1) % report_interval == 0 or gi == len(intervals) - 1:
            dur_s = (end_ms - start_ms) / 1000
            clip_count = len(bgm_clips) if bgm_clips else 0
            print(f"  [{gi+1:3d}/{len(intervals)}] {bgm_type:12s} | {start_ms/1000:>7.1f}s - {end_ms/1000:>7.1f}s ({dur_s:7.1f}s, {clip_count} clips)")

    # ── Concatenate ──
    print(f"\n  Concatenating {len(wav_files)} segments via pydub...")
    duration = _concat_via_pydub(wav_files, output_path)
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
