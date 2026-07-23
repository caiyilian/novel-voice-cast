"""Run a small, production-shaped quality validation for performance direction.

This CLI intentionally uses the current parser, three-agent profile pipeline,
three-agent line-direction pipeline, and strict production validators. Its
artifacts are isolated from the full pipeline so a quality run cannot satisfy
or disturb normal production checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.emotion_labeler import (  # noqa: E402
    EMOTIONS,
    EMOTION_PIPELINE_VERSION,
    TONES,
    emotion_source_hash,
)
from app.core.llm_client import LLMClient, SENSENOVA_FLASH_LITE_MODEL  # noqa: E402
from app.core.parser import parse  # noqa: E402
from app.core.performance_director import (  # noqa: E402
    PERFORMANCE_DIRECTION_PIPELINE_VERSION,
    PERFORMANCE_PROMPT_SIGNATURE,
    PERFORMANCE_PROFILE_PIPELINE_VERSION,
    PROFILE_PROMPT_SIGNATURE,
    build_performance_profiles,
    direct_all_performances,
    performance_direction_source_hash,
    performance_profile_source_hash,
    validate_performance_payload,
    validate_profile_payload,
)


DEFAULT_INDICES = (4, 248, 294, 316, 564, 797, 1804, 2590)
STAGES = ("primary", "independent_review", "final_adjudication")
logger = logging.getLogger("issue99_performance_validation")


class ValidationError(RuntimeError):
    """The validation run inputs or outputs are not production-compatible."""


@dataclass(frozen=True)
class ValidationArtifacts:
    profile_checkpoint: Path
    profile_output: Path
    direction_checkpoint: Path
    direction_output: Path
    telemetry: Path
    log: Path


def default_artifacts(root: Path = ROOT) -> ValidationArtifacts:
    data_dir = root / "backend" / "data" / "issue99_validation"
    return ValidationArtifacts(
        profile_checkpoint=data_dir / "performance_profiles.checkpoint.json",
        profile_output=data_dir / "performance_profiles.json",
        direction_checkpoint=data_dir / "performance_directions.checkpoint.json",
        direction_output=data_dir / "performance_directions.json",
        telemetry=root / "logs" / "issue99_performance_validation_llm_calls.jsonl",
        log=root / "logs" / "issue99_performance_validation.log",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small real-data, three-agent quality validation of the current "
            "performance director. This command makes real LLM calls."
        )
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=[str(value) for value in DEFAULT_INDICES],
        metavar="INDEX",
        help=(
            "Dialogue indices, separated by spaces and/or commas "
            f"(default: {','.join(str(value) for value in DEFAULT_INDICES)})"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Project config path, relative to the project root by default",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible validation outputs and resume compatible checkpoints",
    )
    return parser


def parse_indices(values: Sequence[str | int]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                index = int(part)
            except ValueError as exc:
                raise ValidationError(f"invalid dialogue index: {part!r}") from exc
            if index < 0:
                raise ValidationError(f"dialogue indices must be non-negative: {index}")
            parsed.append(index)
    if not parsed:
        raise ValidationError("at least one dialogue index is required")
    return sorted(set(parsed))


def resolve_project_path(value: str | os.PathLike[str], root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )


def load_inputs(
    config_value: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, Path, Path]:
    config_path = resolve_project_path(config_value)
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValidationError(f"cannot read config: {config_path}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("novel"), dict):
        raise ValidationError(f"config has no novel mapping: {config_path}")

    novel_config = config["novel"]
    if not novel_config.get("text_path") or not novel_config.get("labels_path"):
        raise ValidationError("config novel.text_path and novel.labels_path are required")
    novel_path = resolve_project_path(novel_config["text_path"])
    labels_path = resolve_project_path(novel_config["labels_path"])
    try:
        novel_text = novel_path.read_text(encoding="utf-8")
        labels = [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise ValidationError(
            f"cannot read current novel/labels: {novel_path}, {labels_path}"
        ) from exc
    dialogues, _ = parse(novel_text, labels)
    if not dialogues:
        raise ValidationError("the current parser produced no dialogues")
    return config, dialogues, novel_text, novel_path, labels_path


def load_character_cards(config: dict[str, Any]) -> str:
    performance = config.get("performance", {})
    illustrations = config.get("illustrations", {})
    configured = performance.get(
        "character_cards_path",
        illustrations.get("character_cards_path", "docs/\u89d2\u8272\u5361.md"),
    )
    path = resolve_project_path(configured)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _payload_value(payload: dict[str, Any], key: str) -> Any:
    meta = payload.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return payload.get(key)


def _usable_emotion(value: Any, dialogue_index: int, dialogue: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        if "dialogue_index" in value and int(value["dialogue_index"]) != dialogue_index:
            return False
        if "source_line" in value and int(value["source_line"]) != int(dialogue.get("line", 0)):
            return False
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        return False
    return (
        value.get("emotion") in EMOTIONS
        and value.get("tone") in TONES
        and 0.0 <= confidence <= 1.0
    )


def merge_emotion_advisories(
    novel_text: str,
    dialogues: list[dict[str, Any]],
    target_indices: Sequence[int],
    *,
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge final then checkpoint emotion records; compatible checkpoint wins."""
    expected_hash = emotion_source_hash(novel_text, dialogues)
    paths = (
        ("final", root / "backend" / "data" / "emotion_results.json"),
        ("checkpoint", root / "backend" / "data" / "emotion_results.checkpoint.json"),
        ("targeted", root / "backend" / "data" / "issue99_validation" / "emotion_v3_targeted.json"),
    )
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for source_name, path in paths:
        payload = read_json(path)
        payload_hash = _payload_value(payload, "source_hash")
        pipeline_version = _payload_value(payload, "pipeline_version")
        if payload_hash and payload_hash != expected_hash:
            logger.warning("Ignoring source-incompatible emotion %s: %s", source_name, path)
            continue
        if pipeline_version is not None and pipeline_version != EMOTION_PIPELINE_VERSION:
            logger.warning("Ignoring version-incompatible emotion %s: %s", source_name, path)
            continue
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        for dialogue_index in target_indices:
            value = results.get(str(dialogue_index))
            if _usable_emotion(value, dialogue_index, dialogues[dialogue_index]):
                merged[str(dialogue_index)] = value
                sources[str(dialogue_index)] = source_name
    return merged, sources


def performance_settings(config: dict[str, Any]) -> dict[str, int]:
    raw = config.get("performance", {})
    settings = {
        "context_radius": int(raw.get("context_radius", 100)),
        "min_control_chars": int(raw.get("min_control_chars", 18)),
        "max_control_chars": int(raw.get("max_control_chars", 140)),
        "max_agent_rounds": int(raw.get("max_agent_rounds", 8)),
        "item_retries": int(raw.get("item_retries", 3)),
    }
    if settings["context_radius"] < 1:
        raise ValidationError("performance.context_radius must be positive")
    if not 1 <= settings["min_control_chars"] <= settings["max_control_chars"]:
        raise ValidationError("performance control character limits are invalid")
    if settings["max_agent_rounds"] < 1 or settings["item_retries"] < 1:
        raise ValidationError("performance retry limits must be positive")
    return settings


def _usage_from_checkpoint(path: Path, client: Any) -> dict[str, Any]:
    value = read_json(path).get("llm_usage")
    if isinstance(value, dict):
        return value
    summary = getattr(client, "usage_summary", None)
    return summary() if callable(summary) else {}


def build_profile_payload(
    profiles: dict[str, dict[str, Any]],
    speakers: Sequence[str],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    character_cards: str,
    artifacts: ValidationArtifacts,
    client: Any,
) -> dict[str, Any]:
    return {
        "meta": {
            "model": SENSENOVA_FLASH_LITE_MODEL,
            "pipeline_version": PERFORMANCE_PROFILE_PIPELINE_VERSION,
            "prompt_signature": PROFILE_PROMPT_SIGNATURE,
            "source_hash": performance_profile_source_hash(
                novel_text, speakers, dialogues, character_cards
            ),
            "target_speakers": list(speakers),
        },
        "profiles": profiles,
        "llm_usage": _usage_from_checkpoint(artifacts.profile_checkpoint, client),
    }


def build_direction_payload(
    results: dict[str, dict[str, Any]],
    target_indices: Sequence[int],
    dialogues: list[dict[str, Any]],
    novel_text: str,
    profiles: dict[str, dict[str, Any]],
    emotions: dict[str, Any],
    character_cards: str,
    settings: dict[str, int],
    profile_source_hash: str,
    artifacts: ValidationArtifacts,
    client: Any,
) -> dict[str, Any]:
    return {
        "meta": {
            "model": SENSENOVA_FLASH_LITE_MODEL,
            "pipeline_version": PERFORMANCE_DIRECTION_PIPELINE_VERSION,
            "prompt_signature": PERFORMANCE_PROMPT_SIGNATURE,
            "source_hash": performance_direction_source_hash(
                novel_text,
                target_indices,
                dialogues,
                profiles,
                emotions,
                character_cards,
                settings["context_radius"],
                settings["min_control_chars"],
                settings["max_control_chars"],
            ),
            "target_count": len(target_indices),
            "profile_source_hash": profile_source_hash,
        },
        "results": results,
        "llm_usage": _usage_from_checkpoint(artifacts.direction_checkpoint, client),
    }


def _require_valid(problems: Sequence[str], label: str) -> None:
    if problems:
        detail = "; ".join(str(problem) for problem in problems[:8])
        raise ValidationError(f"{label} failed strict validation: {detail}")


def _format_usage(stage: str, usage: Any) -> str:
    usage = usage if isinstance(usage, dict) else {}
    utilization = float(usage.get("max_context_utilization", 0.0) or 0.0)
    return (
        f"    {stage:<20} calls={int(usage.get('calls', 0) or 0)} "
        f"rounds={int(usage.get('logical_rounds', 0) or 0)} "
        f"tokens={int(usage.get('prompt_tokens', 0) or 0)}/"
        f"{int(usage.get('completion_tokens', 0) or 0)}/"
        f"{int(usage.get('total_tokens', 0) or 0)} "
        f"context_peak={int(usage.get('peak_context_tokens', 0) or 0)} "
        f"reserved_peak={int(usage.get('peak_reserved_context_tokens', 0) or 0)} "
        f"utilization={utilization:.2%}"
    )


def print_quality_summary(
    profiles: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    target_indices: Sequence[int],
    emotion_sources: dict[str, str],
    artifacts: ValidationArtifacts,
) -> None:
    print("\nProfile agent token/context summary")
    for speaker, profile in profiles.items():
        print(f"  speaker={speaker}")
        usage = profile.get("agent_usage", {})
        for stage in STAGES:
            print(_format_usage(stage, usage.get(stage)))

    print("\nPer-line controls and agent token/context summary")
    for dialogue_index in target_indices:
        value = results[str(dialogue_index)]
        advisory = emotion_sources.get(str(dialogue_index), "none")
        print(
            f"  [{dialogue_index}] line={value.get('source_line')} "
            f"speaker={value.get('speaker')} emotion_advisory={advisory}"
        )
        print(f"    text: {value.get('text', '')}")
        print(f"    control: {value.get('performance_control', '')}")
        usage = value.get("agent_usage", {})
        for stage in STAGES:
            print(_format_usage(stage, usage.get(stage)))

    print("\nStrict validation: OK")
    print(f"Profile output: {artifacts.profile_output}")
    print(f"Direction output: {artifacts.direction_output}")
    print(f"Telemetry: {artifacts.telemetry}")
    print(f"Log: {artifacts.log}")


def run_validation(args: argparse.Namespace, artifacts: ValidationArtifacts) -> None:
    configure_logging(artifacts.log)
    target_indices = parse_indices(args.indices)
    config, dialogues, novel_text, novel_path, labels_path = load_inputs(args.config)
    out_of_range = [value for value in target_indices if value >= len(dialogues)]
    if out_of_range:
        raise ValidationError(
            f"dialogue indices exceed current parsed count {len(dialogues)}: {out_of_range}"
        )

    speakers = list(
        dict.fromkeys(str(dialogues[index].get("speaker", "")).strip() for index in target_indices)
    )
    if any(not speaker for speaker in speakers):
        raise ValidationError("every target dialogue must have a speaker label")
    settings = performance_settings(config)
    character_cards = load_character_cards(config)
    emotions, emotion_sources = merge_emotion_advisories(
        novel_text, dialogues, target_indices
    )
    source_counts = {
        source: sum(value == source for value in emotion_sources.values())
        for source in ("final", "checkpoint", "targeted")
    }
    print(f"Novel: {novel_path}")
    print(f"Labels: {labels_path}")
    print(f"Targets: {len(target_indices)} dialogues, {len(speakers)} speakers")
    print(
        "Emotion advisory: "
        f"final={source_counts['final']} checkpoint={source_counts['checkpoint']} "
        f"targeted={source_counts['targeted']} "
        f"none={len(target_indices) - len(emotion_sources)}"
    )

    client = LLMClient.for_flash_lite(
        "issue99_performance_validation",
        telemetry_path=artifacts.telemetry,
    )

    profile_payload = read_json(artifacts.profile_output) if args.resume else {}
    profile_problems = validate_profile_payload(
        profile_payload,
        speakers,
        dialogues,
        novel_text,
        character_cards_text=character_cards,
    )
    if profile_payload and not profile_problems:
        profiles = profile_payload["profiles"]
        print(f"Resumed strict profile output: {artifacts.profile_output}")
    else:
        profiles = build_performance_profiles(
            speakers,
            novel_text,
            dialogues,
            character_cards_text=character_cards,
            client=client,
            checkpoint_path=artifacts.profile_checkpoint,
            resume=bool(args.resume),
            max_agent_rounds=settings["max_agent_rounds"],
        )
        profile_payload = build_profile_payload(
            profiles,
            speakers,
            dialogues,
            novel_text,
            character_cards,
            artifacts,
            client,
        )
        profile_problems = validate_profile_payload(
            profile_payload,
            speakers,
            dialogues,
            novel_text,
            character_cards_text=character_cards,
        )
        _require_valid(profile_problems, "profile payload")
        atomic_write_json(artifacts.profile_output, profile_payload)

    _require_valid(
        validate_profile_payload(
            profile_payload,
            speakers,
            dialogues,
            novel_text,
            character_cards_text=character_cards,
        ),
        "profile payload",
    )

    direction_payload = read_json(artifacts.direction_output) if args.resume else {}
    direction_problems = validate_performance_payload(
        direction_payload,
        target_indices,
        dialogues,
        novel_text,
        profiles,
        emotions,
        character_cards_text=character_cards,
        context_radius=settings["context_radius"],
        min_control_chars=settings["min_control_chars"],
        max_control_chars=settings["max_control_chars"],
    )
    if direction_payload and not direction_problems:
        results = direction_payload["results"]
        print(f"Resumed strict direction output: {artifacts.direction_output}")
    else:
        results = direct_all_performances(
            target_indices,
            dialogues,
            novel_text,
            profiles,
            emotions,
            character_cards_text=character_cards,
            client=client,
            checkpoint_path=artifacts.direction_checkpoint,
            resume=bool(args.resume),
            context_radius=settings["context_radius"],
            min_control_chars=settings["min_control_chars"],
            max_control_chars=settings["max_control_chars"],
            max_agent_rounds=settings["max_agent_rounds"],
            item_retries=settings["item_retries"],
        )
        direction_payload = build_direction_payload(
            results,
            target_indices,
            dialogues,
            novel_text,
            profiles,
            emotions,
            character_cards,
            settings,
            profile_payload["meta"]["source_hash"],
            artifacts,
            client,
        )
        direction_problems = validate_performance_payload(
            direction_payload,
            target_indices,
            dialogues,
            novel_text,
            profiles,
            emotions,
            character_cards_text=character_cards,
            context_radius=settings["context_radius"],
            min_control_chars=settings["min_control_chars"],
            max_control_chars=settings["max_control_chars"],
        )
        _require_valid(direction_problems, "direction payload")
        atomic_write_json(artifacts.direction_output, direction_payload)

    persisted = read_json(artifacts.direction_output)
    if not persisted and direction_payload:
        persisted = direction_payload
    _require_valid(
        validate_performance_payload(
            persisted,
            target_indices,
            dialogues,
            novel_text,
            profiles,
            emotions,
            character_cards_text=character_cards,
            context_radius=settings["context_radius"],
            min_control_chars=settings["min_control_chars"],
            max_control_chars=settings["max_control_chars"],
        ),
        "persisted direction payload",
    )
    print_quality_summary(
        profiles,
        persisted["results"],
        target_indices,
        emotion_sources,
        artifacts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_validation(args, default_artifacts())
    except KeyboardInterrupt:
        print("Validation interrupted; checkpoints were preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Performance validation failed")
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
