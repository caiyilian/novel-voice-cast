"""
Run Stage 7-a BGM scene segmentation and cache the result as JSON.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.core.bgm_segmenter import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    SegmentationError,
    load_segments,
    save_segments,
    segment_novel,
    validate_segments,
)
from app.core.ollama_client import OllamaClient, OllamaConfig  # noqa: E402
from app.core.parser import parse  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_ollama_client(config: dict) -> OllamaClient:
    raw = config.get("ollama", {})
    ollama_config = OllamaConfig(
        base_url=raw.get("base_url", "http://localhost:11434"),
        model=raw.get("model", "qwen3:4b"),
        timeout=int(raw.get("timeout", 120)),
        retries=int(raw.get("retries", 2)),
        retry_delay=float(raw.get("retry_delay", 5.0)),
    )
    return OllamaClient(ollama_config)


def parse_novel(config: dict) -> tuple[list, list, str]:
    novel_path = config["novel"]["text_path"]
    labels_path = config["novel"]["labels_path"]

    with open(novel_path, "r", encoding="utf-8") as f:
        novel_text = f.read()
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    dialogues, characters = parse(novel_text, labels)
    return dialogues, characters, novel_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7-a: segment a novel into BGM scenes")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON cache path")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--min-segments", type=int, default=5, help="Minimum accepted scene segments")
    parser.add_argument("--max-segments", type=int, default=20, help="Maximum accepted scene segments")
    parser.add_argument("--max-tool-steps", type=int, default=80, help="Maximum LLM tool-calling turns")
    args = parser.parse_args()

    start_time = time.time()
    output_path = Path(args.output)

    config = load_config(args.config)
    dialogues, characters, novel_text = parse_novel(config)
    total_lines = len(novel_text.splitlines())

    print("=" * 60)
    print("Stage 7-a: BGM scene segmentation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Novel lines: {total_lines}")
    print(f"Parsed dialogue/narration entries: {len(dialogues)}")
    print(f"Characters: {len(characters)}")

    if output_path.exists() and not args.force:
        segments = load_segments(output_path)
        problems = validate_segments(segments, total_lines, args.min_segments, args.max_segments)
        if problems:
            print(f"Existing cache is invalid: {'; '.join(problems)}")
            print("Use --force to regenerate.")
            return 1
        print(f"Loaded existing cache: {output_path}")
        print(f"Segments: {len(segments)}")
        return 0

    client = build_ollama_client(config)
    print(f"Ollama: {client.config.base_url} / {client.config.model}")

    try:
        segments = segment_novel(
            novel_text,
            dialogues=dialogues,
            client=client,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
            max_tool_steps=args.max_tool_steps,
        )
    except SegmentationError as exc:
        print(f"Segmentation failed: {exc}")
        return 1

    save_segments(output_path, segments)
    elapsed = time.time() - start_time

    print(f"Saved: {output_path}")
    print(f"Segments: {len(segments)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(json.dumps(segments, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
