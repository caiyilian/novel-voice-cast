#!/usr/bin/env python
"""Vocal separation CLI — extract vocals from any audio file using Demucs.

Usage:
    python scripts/vocal_separate.py my_clip.mp3
    python scripts/vocal_separate.py my_clip.mp3 --device cpu
    python scripts/vocal_separate.py my_clip.mp3 --output output/vocals --keep-all

For full options:
    python scripts/vocal_separate.py --help
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.vocal_separator import separate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Extract vocals from an audio file using AI (Demucs htdemucs_6s)"
    )
    p.add_argument("input", help="Input audio file (MP3, WAV, FLAC, etc.)")
    p.add_argument("--output", "-o", default=None, help="Output directory")
    p.add_argument(
        "--device", "-d", default=None,
        help='Torch device: "cuda" or "cpu" (auto-detected if not set)',
    )
    p.add_argument(
        "--keep-all", action="store_true",
        help="Keep all 6 stems (vocals, drums, bass, guitar, piano, other) "
             "instead of only vocals",
    )
    args = p.parse_args()

    try:
        result = separate(
            input_path=args.input,
            output_dir=args.output,
            device=args.device,
            keep_all_stems=args.keep_all,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 50}")
    print("Separation complete!")
    for name, path in result.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {name:20s}  {path}  ({size_mb:.1f} MB)")
    print(f"{'=' * 50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
