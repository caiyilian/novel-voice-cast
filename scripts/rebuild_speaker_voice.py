"""Rebuild one speaker's cloned voice while reusing every unaffected artifact.

The utility creates an isolated output variant.  It reuses compatible TTS WAVs
for every other speaker (hard links when possible), regenerates only the target
speaker with VoxCPM, then rebuilds the speech splice, BGM mix, subtitled videos,
and optional under-limit video parts.  Source outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for import_path in (ROOT, BACKEND):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.core.bgm_mixer import mix_bgm  # noqa: E402
from scripts import run_full as pipeline  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/config.yaml"
DEFAULT_REFERENCE = ROOT / "backend/data/presets/design_female_gentle.wav"
DEFAULT_OUTPUT_DIR = ROOT / "output/holo_design_female_gentle"
REBUILD_STAGES = ("tts", "splice", "bgm-mix", "video", "split")


class RebuildError(RuntimeError):
    """The isolated speaker rebuild cannot proceed safely."""


def _copy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({key: value for key, value in config.items() if key != "_config_path"})


def build_variant_config(
    source_config: dict[str, Any],
    *,
    speaker: str,
    reference_audio: Path,
    target_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Point mutable outputs at an isolated directory and immutable caches at source."""

    variant = _copy_config(source_config)
    target_dir = target_dir.resolve()
    reference_audio = reference_audio.resolve()
    variant.setdefault("characters", {})[speaker] = str(reference_audio)
    variant.setdefault("output", {})["dir"] = str(target_dir)
    variant["output"]["filename"] = "full_volume"
    variant["output"]["format"] = "mp3"

    streaming = variant.setdefault("streaming_tts", {})
    streaming["enabled"] = False
    streaming["checkpoint_path"] = str(target_dir / "streaming_tts.checkpoint.json")
    streaming["log_path"] = str(target_dir / "streaming_tts.log")

    bgm = variant.setdefault("bgm", {})
    bgm["segments_path"] = str(pipeline.bgm_segments_path(source_config))
    bgm["force_segmentation"] = False
    bgm["force_label"] = False
    bgm["force_generate"] = False

    source_illustrations = pipeline.illustration_variant_specs(source_config)
    illustrations = variant.setdefault("illustrations", {})
    illustrations["plan_path"] = str(pipeline.illustration_plan_path(source_config))
    illustrations["prompt_audit_checkpoint_path"] = str(
        pipeline.visual_prompt_checkpoint_path(source_config)
    )
    illustrations["character_cards_path"] = str(
        pipeline.character_cards_path(source_config)
    )
    portrait = source_illustrations[0]
    illustrations["output_dir"] = str(portrait["directory"])
    illustrations["checkpoint_path"] = str(portrait["checkpoint"])
    illustrations["force_plan"] = False
    illustrations["force_generate"] = False
    illustrations["force_prompt_audit"] = False
    if len(source_illustrations) > 1:
        source_landscape = source_illustrations[1]
        landscape = illustrations.setdefault("landscape", {})
        landscape["enabled"] = True
        landscape["output_dir"] = str(source_landscape["directory"])
        landscape["checkpoint_path"] = str(source_landscape["checkpoint"])

    video = variant.setdefault("video", {})
    video["force"] = False
    video["output_path"] = str(
        target_dir / "illustration_video_portrait_7x9_subtitled.mp4"
    )
    video["subtitle_path"] = str(
        target_dir / "illustration_video_portrait_7x9_subtitles.srt"
    )
    if len(source_illustrations) > 1:
        landscape_video = video.setdefault("landscape", {})
        landscape_video["enabled"] = True
        landscape_video["output_path"] = str(
            target_dir / "illustration_video_landscape_16x9_subtitled.mp4"
        )
        landscape_video["subtitle_path"] = str(
            target_dir / "illustration_video_landscape_16x9_subtitles.srt"
        )

    variant["_config_path"] = str(config_path.resolve())
    return variant


def write_variant_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _copy_config(config)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_analysis_state(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dialogues, characters, novel_text = pipeline.step_parse(config)
    genders = pipeline.require_gender_results(characters, dialogues, novel_text)
    emotions = (
        pipeline.require_emotion_results(dialogues, novel_text)
        if config.get("features", {}).get("emotion_label", True)
        else {}
    )
    performances = (
        pipeline.require_performance_results(
            config,
            dialogues,
            novel_text,
            genders,
            emotions,
        )
        if config.get("features", {}).get("performance_direction", False)
        else {}
    )
    return dialogues, genders, emotions, performances


def build_tts_tasks(
    config: dict[str, Any],
    dialogues: Sequence[dict[str, Any]],
    genders: Mapping[str, Any],
    emotions: Mapping[str, Any],
    performances: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, dialogue in enumerate(dialogues):
        raw_speaker = dialogue.get("speaker", "")
        speaker = pipeline.effective_speaker(raw_speaker)
        gender = genders.get(speaker, genders.get(raw_speaker, {})).get("gender", "male")
        if gender not in {"male", "female"}:
            gender = "male"
        tasks.append(
            pipeline.make_tts_task(
                config,
                index,
                dialogue,
                gender,
                emotions.get(str(index), {}),
                performances.get(str(index), {}),
            )
        )
    return tasks


def speaker_indices(
    dialogues: Sequence[Mapping[str, Any]], speaker: str
) -> set[int]:
    normalized = pipeline.effective_speaker(speaker)
    return {
        index
        for index, dialogue in enumerate(dialogues)
        if pipeline.effective_speaker(dialogue.get("speaker", "")) == normalized
    }


def _link_or_copy(source: Path, target: Path) -> str:
    """Replace one controlled target file using a disk-saving hard link when possible."""

    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() and not target.is_symlink():
            raise RebuildError(f"refusing to replace non-file target: {target}")
        target.unlink()
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def seed_variant_tts_cache(
    *,
    source_manifest: Mapping[str, Any],
    source_tasks: Sequence[dict[str, Any]],
    variant_tasks: Sequence[dict[str, Any]],
    target_indices: set[int],
    variant_manifest_path: Path,
    variant_source_hash: str,
) -> dict[str, int]:
    """Seed a valid per-item manifest without trusting stale variant WAV files."""

    if len(source_tasks) != len(variant_tasks):
        raise RebuildError("source and variant task counts differ")
    source_entries = source_manifest.get("segments", {})
    if not isinstance(source_entries, Mapping):
        raise RebuildError("source TTS manifest has no segment entries")

    existing = pipeline.read_json(variant_manifest_path, {})
    existing_entries = (
        existing.get("segments", {})
        if isinstance(existing, dict)
        and existing.get("version") == pipeline.TTS_PIPELINE_VERSION
        and existing.get("source_hash") == variant_source_hash
        and existing.get("total_segments") == len(variant_tasks)
        else {}
    )
    if not isinstance(existing_entries, Mapping):
        existing_entries = {}

    new_entries: dict[str, Any] = {}
    counts = {"already_valid": 0, "linked": 0, "copied": 0, "target_pending": 0}
    for index, (source_task, variant_task) in enumerate(zip(source_tasks, variant_tasks)):
        key = str(index)
        existing_entry = existing_entries.get(key, {})
        if pipeline.reusable_tts_entry(existing_entry, variant_task):
            new_entries[key] = pipeline.completed_tts_entry(variant_task)
            counts["already_valid"] += 1
            continue
        if index in target_indices:
            counts["target_pending"] += 1
            continue

        source_entry = source_entries.get(key, {})
        if not pipeline.reusable_tts_entry(source_entry, source_task):
            raise RebuildError(f"source TTS segment {index} is missing, stale, or invalid")
        method = _link_or_copy(
            Path(source_task["output_path"]),
            Path(variant_task["output_path"]),
        )
        new_entries[key] = pipeline.completed_tts_entry(variant_task)
        counts[method] += 1

    payload = {
        "version": pipeline.TTS_PIPELINE_VERSION,
        "source_hash": variant_source_hash,
        "total_segments": len(variant_tasks),
        "segments": new_entries,
    }
    if existing != payload:
        pipeline.write_json(variant_manifest_path, payload)
    return counts


def _fresh_media(output: Path, dependencies: Sequence[Path], reference: Path) -> bool:
    output_duration = pipeline.media_duration_seconds(output)
    reference_duration = pipeline.media_duration_seconds(reference)
    return (
        pipeline.nonempty_file(output)
        and pipeline.dependencies_are_older(output, dependencies)
        and output_duration > 0
        and reference_duration > 0
        and abs(output_duration - reference_duration) <= 1.0
    )


def rebuild_splice(config: dict[str, Any], segments: list[dict[str, Any]]) -> Path:
    output = pipeline.speech_output_path(config)
    manifest = pipeline.output_dir(config) / "segments/segments_manifest.json"
    if _fresh_media(output, [manifest], output):
        print(f"  using cached speech splice: {output}")
        return output
    pipeline.step_splice(config, segments)
    return output


def rebuild_bgm_mix(
    source_config: dict[str, Any],
    variant_config: dict[str, Any],
) -> Path:
    source_bgm_dir = pipeline.output_dir(source_config) / "bgm"
    source_bgm_manifest = source_bgm_dir / "bgm_manifest.json"
    bgm_segments = pipeline.bgm_segments_path(source_config)
    speech = pipeline.speech_output_path(variant_config)
    output = pipeline.mixed_audio_path(variant_config)
    dependencies = [speech, source_bgm_manifest, bgm_segments]
    missing = [path for path in dependencies if not pipeline.nonempty_file(path)]
    if missing:
        raise RebuildError(f"BGM reuse inputs are missing: {missing}")
    if _fresh_media(output, dependencies, speech):
        print(f"  using cached BGM mix: {output}")
        return output
    duration = mix_bgm(
        speech_path=speech,
        bgm_dir=source_bgm_dir,
        manifest_path=source_bgm_manifest,
        segments_path=bgm_segments,
        output_path=output,
        config_path=variant_config["_config_path"],
    )
    if duration <= 0 or not _fresh_media(output, dependencies, speech):
        raise RebuildError(f"BGM mixer did not create a valid fresh output: {output}")
    return output


def split_cache_valid(
    manifest_path: Path,
    videos: Mapping[str, Path],
    max_minutes: float,
) -> bool:
    manifest = pipeline.read_json(manifest_path, {})
    if not isinstance(manifest, dict) or not manifest.get("outputs"):
        return False
    if manifest.get("strict_max_duration_ms") != int(round(max_minutes * 60_000)):
        return False
    if not pipeline.dependencies_are_older(manifest_path, videos.values()):
        return False
    raw_inputs = manifest.get("inputs", {})
    raw_outputs = manifest.get("outputs", {})
    for name, video in videos.items():
        source = raw_inputs.get(name, {})
        records = raw_outputs.get(name, [])
        if (
            not video.is_file()
            or source.get("path") != str(video.resolve())
            or source.get("size_bytes") != video.stat().st_size
            or not isinstance(records, list)
            or not records
            or not all(Path(record.get("path", "")).is_file() for record in records)
        ):
            return False
    return True


def rebuild_split_parts(
    config: dict[str, Any],
    *,
    max_minutes: float,
    safety_seconds: float,
) -> Path:
    variants = pipeline.video_variant_specs(config)
    videos = {variant["name"]: Path(variant["output"]).resolve() for variant in variants}
    limit_label = f"{max_minutes:g}".replace(".", "p")
    output = pipeline.output_dir(config) / f"video_parts_under_{limit_label}min"
    manifest = output / "split_manifest.json"
    if split_cache_valid(manifest, videos, max_minutes):
        print(f"  using cached video parts: {output}")
        return output
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/split_video_parts.py"),
    ]
    for name, path in videos.items():
        command.extend(["--input", f"{name}={path}"])
    command.extend(
        [
            "--output-dir",
            str(output),
            "--novel",
            str(pipeline.resolve_path(config["novel"]["text_path"])),
            "--labels",
            str(pipeline.resolve_path(config["novel"]["labels_path"])),
            "--segments-dir",
            str(pipeline.output_dir(config) / "segments"),
            "--illustration-plan",
            str(pipeline.illustration_plan_path(config)),
            "--bgm-segments",
            str(pipeline.bgm_segments_path(config)),
            "--ffmpeg",
            str(config.get("video", {}).get("ffmpeg", "ffmpeg")),
            "--ffprobe",
            str(config.get("video", {}).get("ffprobe", "ffprobe")),
            "--max-minutes",
            str(max_minutes),
            "--safety-seconds",
            str(safety_seconds),
            "--force",
        ]
    )
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise RebuildError(f"video splitting failed ({completed.returncode})")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--speaker", default="赫萝")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--to-stage", choices=REBUILD_STAGES, default="split")
    parser.add_argument("--max-part-minutes", type=float, default=60.0)
    parser.add_argument("--part-safety-seconds", type=float, default=30.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source caches and report the exact regeneration scope without writing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.speaker.strip():
        raise SystemExit("--speaker must not be empty")
    if args.max_part_minutes <= 0:
        raise SystemExit("--max-part-minutes must be positive")
    if args.part_safety_seconds < 0:
        raise SystemExit("--part-safety-seconds must be non-negative")

    source_config = pipeline.load_config(str(args.config))
    source_output = pipeline.output_dir(source_config)
    target_dir = pipeline.resolve_path(args.output_dir)
    if target_dir == source_output:
        raise RebuildError("--output-dir must differ from the source output directory")
    reference_audio = pipeline.resolve_path(args.reference_audio)
    if not pipeline.nonempty_file(reference_audio):
        raise RebuildError(f"reference audio is missing or empty: {reference_audio}")

    config_path = target_dir / "rebuild_config.yaml"
    variant_config = build_variant_config(
        source_config,
        speaker=args.speaker.strip(),
        reference_audio=reference_audio,
        target_dir=target_dir,
        config_path=config_path,
    )
    dialogues, genders, emotions, performances = load_analysis_state(source_config)
    targets = speaker_indices(dialogues, args.speaker)
    if not targets:
        raise RebuildError(f"speaker {args.speaker!r} has no parsed dialogue segments")

    source_problems = pipeline.validate_tts_manifest(
        source_config,
        dialogues,
        genders,
        emotions,
        performances,
    )
    if source_problems:
        raise RebuildError(f"source TTS cache is not reusable: {source_problems[:5]}")
    source_tasks = build_tts_tasks(
        source_config, dialogues, genders, emotions, performances
    )
    variant_tasks = build_tts_tasks(
        variant_config, dialogues, genders, emotions, performances
    )
    changed_indices = {
        index
        for index, (source_task, variant_task) in enumerate(
            zip(source_tasks, variant_tasks)
        )
        if source_task["fingerprint"] != variant_task["fingerprint"]
    }
    unexpected = changed_indices - targets
    unchanged_targets = targets - changed_indices
    if unexpected:
        raise RebuildError(
            "variant configuration would also invalidate non-target TTS segments: "
            f"{sorted(unexpected)[:10]}"
        )
    if unchanged_targets:
        raise RebuildError(
            "the requested reference audio does not change every target TTS fingerprint; "
            f"unchanged target indices: {sorted(unchanged_targets)[:10]}"
        )

    print("=" * 72)
    print("Single-speaker isolated voice rebuild")
    print(f"Speaker: {args.speaker}")
    print(f"Reference: {reference_audio}")
    print(f"Source output: {source_output}")
    print(f"Variant output: {target_dir}")
    print(
        f"Dialogues: total={len(dialogues)}, regenerate={len(changed_indices)}, "
        f"reuse={len(dialogues) - len(changed_indices)}"
    )
    print(f"Through stage: {args.to_stage}")
    print("=" * 72)
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    write_variant_config(config_path, variant_config)
    manifest_path = target_dir / "rebuild_speaker_voice_manifest.json"
    run_manifest: dict[str, Any] = {
        "version": 1,
        "status": "running",
        "speaker": args.speaker,
        "reference_audio": str(reference_audio),
        "source_output": str(source_output),
        "variant_output": str(target_dir),
        "total_dialogues": len(dialogues),
        "target_dialogues": len(targets),
        "target_indices": sorted(targets),
        "completed_stages": [],
    }
    pipeline.write_json(manifest_path, run_manifest)

    try:
        variant_source_hash = pipeline.tts_source_hash(
            variant_config,
            dialogues,
            genders,
            emotions,
            performances,
        )
        source_manifest = pipeline.read_json(
            pipeline.output_dir(source_config) / "segments/segments_manifest.json",
            {},
        )
        seed_counts = seed_variant_tts_cache(
            source_manifest=source_manifest,
            source_tasks=source_tasks,
            variant_tasks=variant_tasks,
            target_indices=targets,
            variant_manifest_path=target_dir / "segments/segments_manifest.json",
            variant_source_hash=variant_source_hash,
        )
        print(f"TTS cache seeded: {seed_counts}")
        segments = pipeline.step_tts(
            variant_config,
            dialogues,
            genders,
            emotions,
            performances,
        )
        run_manifest["tts_cache"] = seed_counts
        run_manifest["completed_stages"].append("tts")
        pipeline.write_json(manifest_path, run_manifest)
        if args.to_stage == "tts":
            raise StopIteration

        speech = rebuild_splice(variant_config, segments)
        run_manifest["speech"] = str(speech)
        run_manifest["completed_stages"].append("splice")
        pipeline.write_json(manifest_path, run_manifest)
        if args.to_stage == "splice":
            raise StopIteration

        mixed = rebuild_bgm_mix(source_config, variant_config)
        run_manifest["mixed_audio"] = str(mixed)
        run_manifest["completed_stages"].append("bgm-mix")
        pipeline.write_json(manifest_path, run_manifest)
        if args.to_stage == "bgm-mix":
            raise StopIteration

        pipeline.step_video(variant_config)
        videos = {
            item["name"]: str(Path(item["output"]).resolve())
            for item in pipeline.video_variant_specs(variant_config)
        }
        run_manifest["videos"] = videos
        run_manifest["completed_stages"].append("video")
        pipeline.write_json(manifest_path, run_manifest)
        if args.to_stage == "video":
            raise StopIteration

        parts = rebuild_split_parts(
            variant_config,
            max_minutes=args.max_part_minutes,
            safety_seconds=args.part_safety_seconds,
        )
        run_manifest["video_parts"] = str(parts)
        run_manifest["completed_stages"].append("split")
    except StopIteration:
        pass
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["error"] = str(exc)
        pipeline.write_json(manifest_path, run_manifest)
        raise

    run_manifest["status"] = "complete"
    pipeline.write_json(manifest_path, run_manifest)
    print("=" * 72)
    print(f"Rebuild complete: {target_dir}")
    print(f"Manifest: {manifest_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
