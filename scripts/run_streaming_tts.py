"""Run the resumable VoxCPM consumer alongside performance direction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_full


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream ready performance lines into VoxCPM")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--range", dest="range_value", default="")
    parser.add_argument("--log", default="logs/streaming_tts.log")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_full.configure_logging(args.log)
    try:
        config = run_full.load_config(args.config)
        dialogues, characters, novel_text = run_full.step_parse(config)
        dialogues, characters = run_full.apply_dialogue_selection(
            dialogues,
            characters,
            args.limit,
            args.range_value,
        )
        gender_results = run_full.require_gender_results(characters, dialogues, novel_text)
        emotion_results = (
            run_full.require_emotion_results(dialogues, novel_text)
            if config.get("features", {}).get("emotion_label", True)
            else {}
        )
        run_full.run_streaming_tts(
            config,
            dialogues,
            novel_text,
            gender_results,
            emotion_results,
        )
    except KeyboardInterrupt:
        print("\nStreaming TTS interrupted; checkpoints were preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nSTREAMING TTS FAILED: {exc}", file=sys.stderr)
        return 1
    print("Streaming TTS complete; all dialogue WAV files are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
