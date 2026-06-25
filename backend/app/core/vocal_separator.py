"""Core vocal separation module powered by Demucs (htdemucs_6s).

Extracts the *vocals* stem from any audio file (MP3, WAV, FLAC, etc.),
leaving the instrumental stems (drums, bass, guitar, piano, other)
for future use. Everything runs locally — no upload, no account.

Additionally produces:
  - **metadata.json**: BPM, key, scale, LUFS, peak dBFS, dynamic range,
    tempo stability, and per-stem presence (0-100).
  - **waveform.png**: 2x3 grid overview of all 6 stem waveforms.
  - **peaks.json**: Per-stem [min, max] peak arrays (1500 pts each).

Usage (from CLI or another module):
    from app.core.vocal_separator import separate
    result = separate("my_clip.mp3")
    print(result["vocals"])  # Path to the extracted vocals WAV

Requires:
    - demucs>=4.0.1
    - soundfile>=0.12         (torchaudio 2.7+ backend fallback)
    - ffmpeg                  (audio transcoding)
    - librosa>=0.10           (BPM/key analysis; optional, skip if missing)
    - pyloudnorm>=0.1.1       (LUFS measurement; optional, skip if missing)
    - matplotlib              (waveform visualization; optional, skip if missing)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("vocal_separator")

# ── Monkey-patch: torchaudio 2.7+ removed its built-in WAV writer ──────
# Demucs 4.0.1 calls torchaudio.save() to write separated stems.
# On torch 2.7+ this fails because no audio backend is registered.
# We patch it to use soundfile instead.
def _patch_torchaudio_save():
    import torch
    import torchaudio

    _original_save = torchaudio.save

    def _save_with_soundfile(uri, src, sample_rate, **kwargs):
        """Drop-in replacement for torchaudio.save() using soundfile."""
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        # torchaudio: (channels, samples) → soundfile: (samples, channels)
        if src.ndim == 1:
            src = src[np.newaxis, :]
        src = src.T
        import soundfile as sf

        sf.write(str(uri), src, int(sample_rate), subtype="PCM_16")

    torchaudio.save = _save_with_soundfile


# ── Device detection ───────────────────────────────────────────────────
def _detect_device() -> str:
    """Pick the best available Torch device for Demucs."""
    forced = os.environ.get("VOCAL_DEMUCS_DEVICE", "").strip().lower()
    if forced in ("cuda", "cpu"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── FFmpeg helpers ─────────────────────────────────────────────────────
def _ffmpeg_cmd() -> str:
    return os.environ.get("VOCAL_FFMPEG", "ffmpeg")


def _transcode_to_wav(source: Path, dest: Path, timeout: int = 300) -> None:
    """Transcode any audio to 16-bit 44.1 kHz stereo WAV for Demucs."""
    cmd = [
        _ffmpeg_cmd(),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-sample_fmt",
        "s16",
        "-y",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed: {result.stderr.decode('utf-8', errors='replace').strip()}"
        )


def _decode_via_ffmpeg(source: Path, sr: int = 22050, max_duration: float = 180.0) -> tuple | None:
    """Decode source to mono float32 numpy array via ffmpeg (bypasses librosa's
    deprecated audioread fallback). Returns (samples, sr) or None on failure."""
    cmd = [
        _ffmpeg_cmd(),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",  # mono
        "-ar",
        str(sr),
        "-f",
        "f32le",  # raw 32-bit float little-endian
        "-t",
        str(max_duration),
        "-",  # write to stdout
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=max_duration + 30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("ffmpeg decode failed for %s: %s", source, e)
        return None
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        return None
    return y, sr


# ── Demucs runner ──────────────────────────────────────────────────────
def _run_demucs(
    source_wav: Path,
    output_dir: Path,
    device: str,
    model: str = "htdemucs_6s",
) -> Path:
    """Run Demucs separation as a subprocess. Returns the path to the
    Demucs output directory (<output_dir>/<model>/<stem>)."""
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model,
        "-d",
        device,
        "-o",
        str(output_dir),
        str(source_wav),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    logger.info("Running Demucs: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
        env=env,
    )
    if proc.stderr is None:
        raise RuntimeError("Demucs subprocess has no stderr pipe")

    tail: list[str] = []
    for line in proc.stderr:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 20:
            tail.pop(0)
        if "%" in line:
            logger.debug("Demucs: %s", line)

    proc.wait()
    if proc.returncode != 0:
        detail = "\n".join(tail[-10:]) if tail else f"exit code {proc.returncode}"
        raise RuntimeError(f"Demucs failed:\n{detail}")

    stems_root = output_dir / model / source_wav.stem
    if not stems_root.is_dir():
        raise RuntimeError(f"Demucs output not found at {stems_root}")
    return stems_root


# ── Audio analysis ─────────────────────────────────────────────────────
# Albrecht-Shanahan key profiles (pop/rock corpus)
_MAJOR_PROFILE = (5.47, 0.14, 2.55, 0.14, 3.15, 2.16, 0.37, 4.92, 0.21, 1.84, 0.18, 1.86)
_MINOR_PROFILE = (5.06, 0.14, 2.42, 2.42, 0.35, 1.96, 0.35, 4.16, 2.53, 0.28, 2.67, 0.62)
_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MINOR_TIE_BREAK_FRAC = 0.05


def _correlate(profile, chroma: list[float], shift: int) -> float:
    n = len(profile)
    rotated = [chroma[(i + shift) % n] for i in range(n)]
    mean_p = sum(profile) / n
    mean_c = sum(rotated) / n
    num = sum((profile[i] - mean_p) * (rotated[i] - mean_c) for i in range(n))
    denom_p = sum((profile[i] - mean_p) ** 2 for i in range(n)) ** 0.5
    denom_c = sum((rotated[i] - mean_c) ** 2 for i in range(n)) ** 0.5
    if denom_p == 0 or denom_c == 0:
        return 0.0
    return num / (denom_p * denom_c)


def _detect_key(chroma_mean: list[float]) -> tuple[str, str, int]:
    """Return (label, scale_name, confidence_pct)."""
    raw: list[tuple[float, str, int]] = []
    for shift in range(12):
        root = chroma_mean[shift]
        pmaj = _correlate(_MAJOR_PROFILE, chroma_mean, shift)
        pmin = _correlate(_MINOR_PROFILE, chroma_mean, shift)
        raw.append((pmaj * root, f"{_PITCHES[shift]} maj", shift))
        raw.append((pmin * root, f"{_PITCHES[shift]} min", shift))
    raw.sort(key=lambda x: x[0], reverse=True)

    best_maj = next(c for c in raw if c[1].endswith("maj"))
    best_min = next(c for c in raw if c[1].endswith("min"))
    gap = abs(best_maj[0] - best_min[0])
    threshold = max(abs(best_maj[0]), abs(best_min[0])) * _MINOR_TIE_BREAK_FRAC
    winner = best_maj if (best_maj[0] > best_min[0] and gap > threshold) else best_min

    runner_up = next(c for c in raw if c[1] != winner[1])
    confidence_score = winner[0] - runner_up[0]
    confidence_pct = max(0, min(100, round(confidence_score / 0.15 * 100)))

    label = winner[1]
    scale_name = "Major" if label.endswith("maj") else "Natural Minor"
    return label, scale_name, confidence_pct


def _measure_loudness(y, sr: int) -> tuple[float | None, float | None]:
    """Return (lufs, peak_db)."""
    if y is None or getattr(y, "size", 0) == 0:
        return None, None
    peak_lin = float(np.abs(y).max())
    peak_db = 20.0 * float(np.log10(peak_lin)) if peak_lin > 1e-9 else None
    lufs: float | None = None
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        lufs_raw = float(meter.integrated_loudness(y))
        if np.isfinite(lufs_raw):
            lufs = lufs_raw
    except ImportError:
        pass
    except Exception as e:
        logger.warning("LUFS measurement failed: %s", e)
    return lufs, peak_db


def _analyze_audio(source: Path) -> dict:
    """Analyze audio for BPM, key, loudness. Returns a dict of results
    (empty dict if librosa is not available)."""
    result: dict = {}

    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed -- skipping BPM/key analysis")
        return result

    loaded = _decode_via_ffmpeg(source, sr=22050, max_duration=180.0)
    if loaded is None:
        return result
    y, sr = loaded

    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
        try:
            tempo = float(tempo_arr[0])
        except (TypeError, IndexError):
            tempo = float(tempo_arr)
        result["bpm"] = int(round(tempo)) if tempo > 0 else None

        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_mean = chroma.mean(axis=1).tolist()
        if any(chroma_mean):
            key, scale, key_conf = _detect_key(chroma_mean)
            result["key"] = key
            result["scale"] = scale
            result["key_confidence"] = key_conf

        lufs, peak_db = _measure_loudness(y, sr)
        result["lufs"] = lufs
        result["peak_db"] = peak_db
        if lufs is not None and peak_db is not None:
            result["dynamic_range"] = round(peak_db - lufs, 1)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) > 2:
            intervals = np.diff(beat_times)
            mean_iv = float(intervals.mean())
            if mean_iv > 0:
                cv = float(intervals.std() / mean_iv)
                result["tempo_stability"] = max(0, min(100, round((1 - min(cv, 1)) * 100)))

    except Exception as e:
        logger.warning("Audio analysis failed: %s", e)

    return result


def _compute_stem_presence(stems_dir: Path, stem_names: list[str]) -> dict[str, int]:
    """Compute per-stem RMS energy, normalized to 0-100."""
    rms_values: dict[str, float] = {}
    for name in stem_names:
        wav = stems_dir / f"{name}.wav"
        if not wav.is_file():
            continue
        loaded = _decode_via_ffmpeg(wav, sr=22050, max_duration=180.0)
        if loaded is None:
            continue
        y, _ = loaded
        rms_values[name] = float(np.sqrt(np.mean(y**2)))
    if not rms_values:
        return {}
    max_rms = max(rms_values.values())
    if max_rms < 1e-9:
        return {name: 0 for name in rms_values}
    return {name: max(0, min(100, round(rms / max_rms * 100))) for name, rms in rms_values.items()}


# ── Waveform visualization ────────────────────────────────────────────
_WAVE_COLORS = {
    "vocals": "#e74c3c",
    "drums": "#f39c12",
    "bass": "#2ecc71",
    "guitar": "#3498db",
    "piano": "#9b59b6",
    "other": "#1abc9c",
}


def _waveform_peaks(path: Path, num_points: int = 2000) -> np.ndarray:
    try:
        import soundfile as sf

        data, _ = sf.read(path)
    except Exception:
        return np.zeros(num_points)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = np.abs(data)
    n = len(data)
    chunk = max(1, n // num_points)
    peaks = np.array([np.max(data[i : i + chunk]) for i in range(0, n, chunk)])
    return peaks[:num_points]


def _compute_peaks_json(stems_dir: Path, stem_names: list[str]) -> dict[str, list[list[float]]]:
    """Compute [min, max] peaks for each stem (1500 pts)."""
    import soundfile as sf

    peaks: dict[str, list[list[float]]] = {}
    _PEAK_POINTS = 1500
    for name in stem_names:
        wav = stems_dir / f"{name}.wav"
        if not wav.is_file():
            continue
        try:
            data, _ = sf.read(wav, dtype="float32", always_2d=True)
            ch = data[:, 0]
            n = len(ch)
            if n == 0:
                continue
            chunk = max(1, n // _PEAK_POINTS)
            result: list[list[float]] = []
            for i in range(0, n, chunk):
                block = ch[i : i + chunk]
                result.append([float(np.min(block)), float(np.max(block))])
            peaks[name] = result[:_PEAK_POINTS]
        except Exception:
            logger.warning("could not compute peaks for %s/%s", stems_dir.name, name)
    return peaks


def _generate_waveform_image(stems_dir: Path, stem_names: list[str]) -> bool:
    """Generate a 2x3 waveform overview image for the 6 stems.
    Returns True if successful, False if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping waveform image")
        return False

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, name in zip(axes_flat, stem_names):
        wav = stems_dir / f"{name}.wav"
        if not wav.is_file():
            ax.set_title(f"{name} (missing)")
            continue
        peaks = _waveform_peaks(wav)
        color = _WAVE_COLORS.get(name, "#333")
        ax.fill_between(np.linspace(0, 100, len(peaks)), peaks, alpha=0.7, color=color)
        ax.set_ylabel("amplitude")
        ax.set_title(name, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 100)
    for ax in axes_flat[len(stem_names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(stems_dir / "waveform.png", dpi=150)
    plt.close(fig)
    logger.info("    -> waveforms (2x3 grid) -> waveform.png")
    return True


def _generate_original_waveform(stems_dir: Path, source_wav: Path, output_name: str = "waveform_original.png") -> bool:
    """Generate a standalone waveform image for the original (source) audio.
    Returns True if successful, False if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping original waveform")
        return False
    if not source_wav.is_file():
        logger.warning("source_wav not found -- skipping original waveform")
        return False

    peaks = _waveform_peaks(source_wav)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(np.linspace(0, 100, len(peaks)), peaks, alpha=0.7, color="#95a5a6")
    ax.set_title("original", fontweight="bold")
    ax.set_ylabel("amplitude")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(stems_dir / output_name, dpi=150)
    plt.close(fig)
    logger.info("    -> original waveform -> %s", output_name)
    return True


# ── Public API ─────────────────────────────────────────────────────────
OUTPUT_DIR_DEFAULT = Path("output/vocal_separation")
DEMUCS_MODEL: str = "htdemucs_6s"
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")


def separate(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    device: str | None = None,
    keep_all_stems: bool = False,
) -> dict[str, Path]:
    """Separate an audio file into stems. Also writes metadata.json,
    waveform.png, and peaks.json to the stems/ subdirectory.

    Args:
        input_path: Path to any audio file (MP3, WAV, FLAC, etc.).
        output_dir: Output directory. Defaults to ``output/vocal_separation/<input_stem>/``.
        device: Torch device (``"cuda"`` or ``"cpu"``). Auto-detected if not set.
        keep_all_stems: If True, keep all 6 stems. If False (default), only
            keep ``vocals.wav`` and delete the rest.

    Returns:
        dict with keys for each kept stem name mapping to the WAV file Path.
    """
    # Patch torchaudio on first call
    _patch_torchaudio_save()

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    if output_dir is None:
        output_dir = OUTPUT_DIR_DEFAULT / source.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device or _detect_device()
    logger.info("Separating %s (device=%s)", source.name, device)

    # 1. Transcode input to 16-bit 44.1 kHz WAV
    source_wav = output_dir / "source.wav"
    logger.info("Transcoding to %s", source_wav.name)
    _transcode_to_wav(source, source_wav)

    # 2. Run Demucs
    stems_root = _run_demucs(source_wav, output_dir, device)

    # 3. Move / collect results
    stems_dir = output_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for name in STEM_NAMES:
        src = stems_root / f"{name}.wav"
        if src.is_file():
            dst = stems_dir / f"{name}.wav"
            if dst.exists():
                dst.unlink()
            src.rename(dst)
            result[name] = dst

    # 4. Original waveform — do this BEFORE deleting source_wav
    _generate_original_waveform(stems_dir, source_wav)

    # 5. Cleanup intermediate files
    _safe_unlink(source_wav)
    _safe_rmtree(stems_root)
    _safe_rmtree(output_dir / DEMUCS_MODEL)

    # 6. Audio analysis (best-effort, uses librosa; runs on original source)
    keep = list(result.keys())
    analysis = _analyze_audio(source)
    if not analysis:
        analysis = _analyze_audio(source_wav)
    if keep:
        analysis["stem_presence"] = _compute_stem_presence(stems_dir, keep)

    # 7. Waveform peaks (peaks.json)
    if keep:
        peaks = _compute_peaks_json(stems_dir, keep)
        if peaks:
            _safe_write_json(stems_dir / "peaks.json", peaks)

    # 8. 2x3 stem waveform image (waveform.png)
    _generate_waveform_image(stems_dir, list(STEM_NAMES))

    # 9. Metadata
    _write_metadata(output_dir, analysis, len(result))

    # 10. Cleanup non-vocal stems unless keep_all_stems
    if not keep_all_stems:
        for name, path in list(result.items()):
            if name != "vocals":
                _safe_unlink(path)
                result.pop(name, None)

    logger.info(
        "Done — extracted %d stem(s) in %.1f MB",
        len(result),
        sum(p.stat().st_size for p in result.values()) / 1024 / 1024,
    )
    return result


def _write_metadata(output_dir: Path, analysis: dict, stem_count: int) -> None:
    """Write metadata.json to the output directory (not stems/)."""
    meta = {
        "stem_count": stem_count,
        "stems_available": list(STEM_NAMES),
        "analysis": analysis,
    }
    path = output_dir / "metadata.json"
    _safe_write_json(path, meta)
    logger.info("    -> metadata.json  (%d fields)", len(meta["analysis"]))


# ── Internal helpers ───────────────────────────────────────────────────
def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_rmtree(path: Path) -> None:
    import shutil

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _safe_write_json(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("could not write %s: %s", path.name, e)


# ── CLI convenience ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    p = argparse.ArgumentParser(description="Extract vocals from an audio file using Demucs")
    p.add_argument("input", help="Input audio file (MP3, WAV, FLAC, etc.)")
    p.add_argument("--output", "-o", default=None, help="Output directory")
    p.add_argument("--device", "-d", default=None, help='Torch device: "cuda" or "cpu"')
    p.add_argument("--keep-all", action="store_true", help="Keep all 6 stems instead of just vocals")
    args = p.parse_args()

    result = separate(args.input, args.output, args.device, args.keep_all)
    print(f"\n{'=' * 50}")
    for name, path in result.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {name:20s}  {path}  ({size_mb:.1f} MB)")
    print(f"{'=' * 50}")