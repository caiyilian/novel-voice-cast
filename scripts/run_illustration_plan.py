#!/usr/bin/env python
"""Phase 1 — Run the illustration planning agent over a novel.

The agent reads the novel through tool calls, identifies visually impactful
moments, and outputs a detailed illustration plan (locations, descriptions,
character involvement, image prompts).

Speaker labels from ``labels.txt`` are embedded into the text as ``【角色】``
so the LLM sees who is speaking without extra tool calls.

Usage:
    python scripts/run_illustration_plan.py --novel novels/novel.txt --labels novels/labels.txt
    python scripts/run_illustration_plan.py --novel novels/novel.txt --labels novels/labels.txt --output output/my_plan.json
    python scripts/run_illustration_plan.py --novel novels/novel.txt --labels novels/labels.txt --resume
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.illustration_planner import plan_illustrations
from app.core.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _install_log_file(path: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)
    print(f"Log file: {log_path}")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Phase 1 — Illustration Planning Agent")
    p.add_argument("--novel", "-n", required=True, help="Path to novel text file")
    p.add_argument("--labels", "-l", help="Path to speaker labels file (one label per line, matching novel.txt lines)")
    p.add_argument("--output", "-o", default="output/illustration_plan.json", help="Output illustration plan path")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint instead of starting fresh")
    p.add_argument("--fresh", action="store_true", help="Delete the checkpoint for this output before running")
    p.add_argument("--debug", action="store_true", help="Write raw LLM responses next to the output plan")
    p.add_argument("--log-file", help="Also write console output to this log file")
    p.add_argument("--character-card", default="docs/角色卡.md", help="Path to character card markdown/table")
    p.add_argument("--visual-memory-output", default="output/character_visual_memory.json", help="Output path for extracted character visual memory")
    p.add_argument("--no-visual-memory", action="store_true", help="Skip the visual memory scan")
    args = p.parse_args()

    if args.log_file:
        _install_log_file(args.log_file)

    novel_path = Path(args.novel)
    if not novel_path.is_file():
        print(f"Error: Novel file not found: {novel_path}", file=sys.stderr)
        return 1

    if args.fresh:
        ckpt_path = Path(str(args.output).replace(".json", ".checkpoint.json"))
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"Deleted checkpoint: {ckpt_path}")

    text = novel_path.read_text(encoding="utf-8")

    labels: list[str] = []
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.is_file():
            labels = labels_path.read_text(encoding="utf-8").splitlines()
            print(f"Labels: {len(labels)} lines loaded from {labels_path.name}")
        else:
            print(f"Warning: Labels file not found: {labels_path}", file=sys.stderr)

    client = LLMClient()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"Novel: {novel_path.name} ({len(text)} chars, {len(text.splitlines())} lines)")
    print(f"LLM: {client.log_summary()}")
    if args.resume:
        print("Mode: RESUME from checkpoint")
    print()

    t0 = time.time()
    try:
        plan = plan_illustrations(
            text=text,
            labels=labels,
            client=client,
            output_path=args.output,
            resume=args.resume,
            debug=args.debug,
            character_card_path=args.character_card,
            visual_memory_path=args.visual_memory_output,
            enable_visual_memory=not args.no_visual_memory,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
