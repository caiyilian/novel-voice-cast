import json
import importlib
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts import run_full
from app.core.bgm_generator import (
    ACE_STEP_MODEL,
    BGM_GENERATION_VERSION,
    build_bgm_seed,
    build_manifest,
    build_segment_bgm_prompt,
)


def write_test_wav(path: Path, frames: int = 4800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\0\0" * frames)


def test_run_checked_subprocess_streams_output_and_reports_failure(capsys):
    with pytest.raises(run_full.PipelineError, match="final failure detail"):
        run_full.run_checked_subprocess(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('visible progress'); print('final failure detail'); sys.exit(7)",
            ],
            timeout=10,
        )

    assert "visible progress" in capsys.readouterr().out


def test_run_checked_subprocess_returns_explicitly_allowed_exit_code():
    returncode = run_full.run_checked_subprocess(
        [sys.executable, "-c", "import sys; sys.exit(75)"],
        timeout=10,
        allowed_returncodes={75},
    )

    assert returncode == 75


def test_execute_stage_records_keyboard_interrupt(tmp_path):
    recorder = run_full.PipelineRecorder(tmp_path / "manifest.json", ["tts"])

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_full.execute_stage(recorder, "tts", interrupt)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "interrupted"
    assert manifest["stages"]["tts"]["status"] == "interrupted"


def test_run_checked_subprocess_enforces_timeout_without_child_output():
    started_at = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_full.run_checked_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
        )

    assert time.monotonic() - started_at < 5


def test_bgm_generator_cli_import_does_not_change_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from scripts import run_bgm_generate

    importlib.reload(run_bgm_generate)

    assert Path.cwd() == tmp_path


def test_bgm_generator_defaults_to_resident_turbo_and_couples_dit_offload(monkeypatch):
    from scripts import run_bgm_generate

    monkeypatch.delenv("ACESTEP_CPU_OFFLOAD", raising=False)
    monkeypatch.delenv("ACESTEP_OFFLOAD_DIT_TO_CPU", raising=False)
    assert run_bgm_generate._resolve_offload_policy() == (False, False)

    monkeypatch.setenv("ACESTEP_CPU_OFFLOAD", "true")
    monkeypatch.setenv("ACESTEP_OFFLOAD_DIT_TO_CPU", "false")
    assert run_bgm_generate._resolve_offload_policy() == (True, False)

    monkeypatch.setenv("ACESTEP_CPU_OFFLOAD", "false")
    monkeypatch.setenv("ACESTEP_OFFLOAD_DIT_TO_CPU", "true")
    assert run_bgm_generate._resolve_offload_policy() == (False, False)


def test_bgm_noise_guard_separates_tone_from_broadband_pulses():
    import numpy as np
    from scripts import run_bgm_generate

    sample_rate = 8_000
    seconds = 3
    timeline = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    musical = (
        0.45 * np.sin(2 * np.pi * 220 * timeline)
        + 0.25 * np.sin(2 * np.pi * 330 * timeline)
    )
    rng = np.random.default_rng(99)
    noisy_pulses = rng.normal(0, 0.25, musical.shape).astype(np.float32)
    pulse = np.zeros_like(noisy_pulses)
    pulse[:: sample_rate // 4] = 1.0
    noisy_pulses += np.convolve(pulse, np.ones(240, dtype=np.float32), mode="same")

    music_flatness, music_harmonicity = run_bgm_generate._music_quality_metrics(
        musical, sample_rate
    )
    noise_flatness, noise_harmonicity = run_bgm_generate._music_quality_metrics(
        noisy_pulses, sample_rate
    )

    assert music_flatness < run_bgm_generate.MAX_NOISE_SPECTRAL_FLATNESS
    assert noise_flatness > music_flatness
    assert music_harmonicity > noise_harmonicity


def test_bgm_generator_synchronizes_around_dit_offload():
    from scripts import run_bgm_generate

    events = []

    class Handler:
        offload_to_cpu = True
        offload_dit_to_cpu = True

        @contextmanager
        def _load_model_context(self, model_name):
            events.append(f"load:{model_name}")
            try:
                yield
            finally:
                events.append(f"offload:{model_name}")

    handler = Handler()
    assert run_bgm_generate._install_model_offload_sync_guard(
        handler,
        synchronize=lambda: events.append("synchronize"),
    )

    with handler._load_model_context("model"):
        events.append("work")

    assert events == [
        "load:model",
        "work",
        "synchronize",
        "offload:model",
        "synchronize",
    ]


def test_resumable_bgm_subprocess_restarts_after_checkpoint_progress(monkeypatch, tmp_path):
    returncodes = iter([
        run_full.BGM_PROCESS_RESTART_EXIT_CODE,
        run_full.WINDOWS_ACCESS_VIOLATION_EXIT_CODE,
        0,
    ])
    progress = iter([440, 460, 461])
    calls = []

    monkeypatch.setattr(
        run_full,
        "run_checked_subprocess",
        lambda command, *args, **kwargs: calls.append(
            (list(command), kwargs["allowed_returncodes"])
        ) or next(returncodes),
    )
    monkeypatch.setattr(
        run_full,
        "_bgm_checkpoint_clip_count",
        lambda _path: next(progress),
    )

    run_full.run_resumable_bgm_subprocess(
        ["ace-step", "--force"],
        timeout=10,
        manifest_path=tmp_path / "manifest.json",
    )

    assert len(calls) == 3
    assert "--force" in calls[0][0]
    assert all("--force" not in command for command, _allowed in calls[1:])
    assert all(
        run_full.BGM_PROCESS_RESTART_EXIT_CODE in allowed
        for _command, allowed in calls
    )


def test_resumable_bgm_subprocess_stops_after_repeated_native_crash_without_progress(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        run_full,
        "run_checked_subprocess",
        lambda *args, **kwargs: run_full.WINDOWS_ACCESS_VIOLATION_EXIT_CODE,
    )
    monkeypatch.setattr(run_full, "_bgm_checkpoint_clip_count", lambda _path: 440)

    with pytest.raises(run_full.PipelineError, match=r"without checkpoint progress \(3 attempts\)"):
        run_full.run_resumable_bgm_subprocess(
            ["ace-step"],
            timeout=10,
            manifest_path=tmp_path / "manifest.json",
            native_no_progress_limit=3,
        )


def test_bgm_manifest_uses_requested_output_directory(tmp_path):
    output_dir = tmp_path / "custom-bgm"

    manifest = build_manifest(
        [{"segment_index": 7, "bgm_type": "daily"}],
        30,
        30,
        1.5,
        clips_per_segment=2,
        output_dir=output_dir,
    )

    assert manifest["segments"]["7"]["clips"] == [
        {
            "clip_index": 0,
            "file": "007_0.mp3",
            "path": str(output_dir / "007_0.mp3"),
            "seed": build_bgm_seed(7, 0),
        },
        {
            "clip_index": 1,
            "file": "007_1.mp3",
            "path": str(output_dir / "007_1.mp3"),
            "seed": build_bgm_seed(7, 1),
        },
    ]


def test_bgm_prompts_and_seeds_are_scene_specific():
    first = {
        "segment_index": 1,
        "bgm_type": "daily",
        "bgm_evidence": "Primary: quiet harvest reflection | Review: ignored duplicate",
    }
    second = {
        "segment_index": 2,
        "bgm_type": "daily",
        "bgm_evidence": "Primary: cautious negotiation with mounting uncertainty",
    }

    assert build_segment_bgm_prompt(first) != build_segment_bgm_prompt(second)
    assert "ignored duplicate" not in build_segment_bgm_prompt(first)
    assert len({build_bgm_seed(index, clip) for index in range(1, 4) for clip in range(3)}) == 9


def test_bgm_generation_checkpoints_each_successful_clip(tmp_path, monkeypatch):
    from scripts import run_bgm_generate

    segments_path = tmp_path / "segments.json"
    output_dir = tmp_path / "bgm"
    segments_path.write_text(
        json.dumps(
            [
                {
                    "segment_index": 1,
                    "bgm_type": "daily",
                    "bgm_evidence": "Primary: quiet harvest reflection",
                }
            ]
        ),
        encoding="utf-8",
    )
    attempts = []

    def fake_generate_clip(*, output_path, **_kwargs):
        attempts.append(output_path.name)
        if len(attempts) == 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"mp3")
            return True
        return False

    monkeypatch.setattr(run_bgm_generate.os, "chdir", lambda _path: None)
    monkeypatch.setattr(run_bgm_generate, "_init_ace_step", lambda: (object(), None, None, None))
    monkeypatch.setattr(run_bgm_generate, "generate_clip", fake_generate_clip)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bgm_generate.py",
            "--segments", str(segments_path),
            "--output-dir", str(output_dir),
            "--clips-per-segment", "2",
        ],
    )

    assert run_bgm_generate.main() == 1
    manifest = json.loads((output_dir / "bgm_manifest.json").read_text(encoding="utf-8"))
    assert [clip["clip_index"] for clip in manifest["segments"]["1"]["clips"]] == [0]
    assert not list(output_dir.glob("*.tmp"))


def make_config(tmp_path: Path) -> dict:
    return {
        "_config_path": str(tmp_path / "config.yaml"),
        "novel": {
            "text_path": str(tmp_path / "novel.txt"),
            "labels_path": str(tmp_path / "labels.txt"),
        },
        "output": {
            "dir": str(tmp_path / "output"),
            "filename": "full_volume",
            "format": "mp3",
        },
        "characters": {"Main": str(tmp_path / "main.wav")},
        "default_audio": {},
        "voxcpm": {"model_path": str(tmp_path / "VoxCPM2")},
        "edge_tts": {
            "male_voice": "male-test",
            "female_voice": "female-test",
        },
        "bgm": {"enabled": True, "clips_per_segment": 2},
    }


def write_bgm_cache(
    segments: list[dict], manifest_path: Path, bgm_dir: Path, clips_per_segment: int
) -> None:
    manifest_segments = {}
    for position, segment in enumerate(segments, 1):
        index = segment.get("segment_index", position)
        clips = []
        for clip_index in range(clips_per_segment):
            filename = f"{index:03d}_{clip_index}.mp3"
            (bgm_dir / filename).write_bytes(b"mp3")
            clips.append({
                "clip_index": clip_index,
                "file": filename,
                "seed": build_bgm_seed(index, clip_index),
            })
        manifest_segments[str(index)] = {
            "bgm_type": segment["bgm_type"],
            "prompt": build_segment_bgm_prompt(segment),
            "clips": clips,
        }
    manifest_path.write_text(
        json.dumps(
            {
                "generation_version": BGM_GENERATION_VERSION,
                "model": ACE_STEP_MODEL,
                "segments": manifest_segments,
                "total_segments": len(segments),
                "clips_per_segment": clips_per_segment,
                "duration_per_segment": 30,
                "inference_steps": 8,
            }
        ),
        encoding="utf-8",
    )


def test_load_config_resolves_repo_paths_independent_of_cwd(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("output:\n  dir: output\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = run_full.load_config(str(config_path))

    assert Path(config["output"]["dir"]) == run_full.ROOT / "output"
    assert run_full.BACKEND_DIR == run_full.ROOT / "backend"


def test_illustration_and_video_variant_specs_keep_outputs_independent(tmp_path):
    config = make_config(tmp_path)
    config["illustrations"] = {
        "output_dir": str(tmp_path / "portrait"),
        "checkpoint_path": str(tmp_path / "portrait.json"),
        "size": "896x1152",
        "landscape": {
            "enabled": True,
            "size": "1280x720",
            "output_dir": str(tmp_path / "landscape"),
            "checkpoint_path": str(tmp_path / "landscape.json"),
        },
    }
    config["video"] = {
        "output_path": str(tmp_path / "portrait.mp4"),
        "subtitle_path": str(tmp_path / "portrait.srt"),
        "landscape": {
            "enabled": True,
            "output_path": str(tmp_path / "landscape.mp4"),
            "subtitle_path": str(tmp_path / "landscape.srt"),
        },
    }

    image_variants = run_full.illustration_variant_specs(config)
    video_variants = run_full.video_variant_specs(config)

    assert [item["name"] for item in image_variants] == ["portrait", "landscape"]
    assert [item["settings"]["size"] for item in image_variants] == ["896x1152", "1280x720"]
    assert image_variants[0]["directory"] != image_variants[1]["directory"]
    assert image_variants[0]["checkpoint"] != image_variants[1]["checkpoint"]
    assert [item["output"] for item in video_variants] == [
        tmp_path / "portrait.mp4",
        tmp_path / "landscape.mp4",
    ]


def test_h3_variant_specs_use_independent_native_chain_caches(tmp_path):
    config = make_config(tmp_path)
    config["illustrations"] = {
        "output_dir": str(tmp_path / "portrait-images"),
        "checkpoint_path": str(tmp_path / "portrait-images.json"),
        "size": "896x1152",
        "landscape": {
            "enabled": True,
            "size": "1280x720",
            "output_dir": str(tmp_path / "landscape-images"),
            "checkpoint_path": str(tmp_path / "landscape-images.json"),
        },
    }
    config["video"] = {
        "output_path": str(tmp_path / "portrait.mp4"),
        "h3": {
            "enabled": True,
            "mode": "native-chain",
            "output_dir": str(tmp_path / "h3"),
            "minimum_duration": 5,
            "maximum_duration": 10,
            "portrait": {"width": 672, "height": 864},
            "landscape": {"width": 960, "height": 544},
        },
        "landscape": {
            "enabled": True,
            "output_path": str(tmp_path / "landscape.mp4"),
        },
    }

    variants = run_full.video_variant_specs(config)
    portrait = run_full.h3_variant_spec(config, variants[0])
    landscape = run_full.h3_variant_spec(config, variants[1])

    assert portrait["mode"] == "native-chain"
    assert (portrait["width"], portrait["height"]) == (672, 864)
    assert (landscape["width"], landscape["height"]) == (960, 544)
    assert portrait["checkpoint"] != landscape["checkpoint"]
    assert portrait["clips_dir"].parent.name == "portrait"
    assert landscape["clips_dir"].parent.name == "landscape"


def test_h3_variant_specs_reject_invalid_model_constraints(tmp_path):
    config = make_config(tmp_path)
    config["illustrations"] = {"size": "896x1152"}
    config["video"] = {
        "h3": {
            "enabled": True,
            "mode": "native-chain",
            "portrait": {"width": 671, "height": 864},
        }
    }

    with pytest.raises(run_full.PipelineError, match="multiples of 16"):
        run_full.h3_variant_spec(config, run_full.video_variant_specs(config)[0])


def test_native_h3_video_dry_run_does_not_require_legacy_illustrations(
    tmp_path, monkeypatch, capsys
):
    config = make_config(tmp_path)
    config["illustrations"] = {"size": "896x1152"}
    config["video"] = {
        "h3": {"enabled": True, "mode": "native-chain"},
    }
    monkeypatch.setattr(run_full, "validate_visual_prompt_audit", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        run_full,
        "stage_cache_status",
        lambda stage, _config: (False, "legacy illustrations are absent")
        if stage == "illustrations"
        else (False, "H3 output is not complete"),
    )

    assert run_full.dry_run_report(config, ("video",))
    output = capsys.readouterr().out
    assert "prerequisite visual-prompt-audit: READY" in output
    assert "prerequisite illustrations" not in output


def test_voice_assignment_uses_voxcpm_only_for_configured_characters(tmp_path):
    config = make_config(tmp_path)

    main = run_full.get_voice_assignment("Main", "male", config)
    other_male = run_full.get_voice_assignment("Other", "male", config)
    other_female = run_full.get_voice_assignment("Other", "female", config)

    assert main == {
        "engine": "voxcpm",
        "reference_audio": str((tmp_path / "main.wav").resolve()),
    }
    assert other_male == {"engine": "edge-tts", "voice_id": "male-test"}
    assert other_female == {"engine": "edge-tts", "voice_id": "female-test"}


def test_force_all_voice_assignment_uses_explicit_and_gender_default_clones(tmp_path):
    config = make_config(tmp_path)
    male_default = tmp_path / "other-male.wav"
    female_default = tmp_path / "other-female.wav"
    config["default_audio"] = {
        "male": str(male_default),
        "female": str(female_default),
    }
    config["voxcpm"]["force_all_characters"] = True
    dialogues = [
        {"speaker": "Main", "text": "main"},
        {"speaker": "Other Man", "text": "male"},
        {"speaker": "Other Woman", "text": "female"},
    ]
    genders = {
        "Main": {"gender": "male"},
        "Other Man": {"gender": "male"},
        "Other Woman": {"gender": "female"},
    }

    assert run_full.get_voice_assignment("Main", "male", config) == {
        "engine": "voxcpm",
        "reference_audio": str((tmp_path / "main.wav").resolve()),
    }
    assert run_full.get_voice_assignment("Other Man", "male", config) == {
        "engine": "voxcpm",
        "reference_audio": str(male_default.resolve()),
    }
    assert run_full.get_voice_assignment("Other Woman", "female", config) == {
        "engine": "voxcpm",
        "reference_audio": str(female_default.resolve()),
    }
    assert run_full.performance_target_indices(config, dialogues, genders) == [0]
    assert run_full.supplemental_performance_target_indices(config, dialogues, genders) == [1, 2]
    assert run_full.all_performance_target_indices(config, dialogues, genders) == [0, 1, 2]


def test_blank_speaker_is_narrator_only_in_supplemental_performance_and_tts(tmp_path):
    config = make_config(tmp_path)
    narrator_reference = tmp_path / "narrator.wav"
    config["characters"][run_full.NARRATOR_SPEAKER] = str(narrator_reference)
    config["default_audio"] = {
        "male": str(tmp_path / "other-male.wav"),
        "female": str(tmp_path / "other-female.wav"),
    }
    config["voxcpm"]["force_all_characters"] = True
    dialogues = [
        {"speaker": "Main", "text": "main", "chapter": "one"},
        {"speaker": "", "text": "quoted non-character fragment", "chapter": "one"},
        {"speaker": "Extra", "text": "extra", "chapter": "one"},
        {"speaker": run_full.NARRATOR_SPEAKER, "text": "narration", "chapter": "one"},
    ]
    genders = {
        "Main": {"gender": "male"},
        "Extra": {"gender": "female"},
        run_full.NARRATOR_SPEAKER: {"gender": "male"},
    }

    primary, supplemental = run_full._performance_groups(config, dialogues, genders)

    assert primary["targets"] == [0, 3]
    assert primary["dialogues"] is dialogues
    assert supplemental["targets"] == [1, 2]
    assert supplemental["speakers"] == [run_full.NARRATOR_SPEAKER, "Extra"]
    assert dialogues[1]["speaker"] == ""
    assert supplemental["dialogues"] is not dialogues
    assert supplemental["dialogues"][1]["speaker"] == run_full.NARRATOR_SPEAKER
    assert run_full.get_voice_assignment("", "male", config) == {
        "engine": "voxcpm",
        "reference_audio": str(narrator_reference.resolve()),
    }

    task = run_full.make_tts_task(config, 1, dialogues[1], "male", {}, {})
    assert task["entry"]["speaker"] == run_full.NARRATOR_SPEAKER
    assert task["reference_audio"] == str(narrator_reference.resolve())
    assert run_full.segments_for_dialogues(config, dialogues)[1]["speaker"] == run_full.NARRATOR_SPEAKER


def test_streaming_tts_consumes_only_ready_performance_and_uses_its_own_checkpoint(
    tmp_path,
    monkeypatch,
):
    config = make_config(tmp_path)
    main_reference = tmp_path / "main.wav"
    female_default = tmp_path / "other-female.wav"
    main_reference.write_bytes(b"main reference")
    female_default.write_bytes(b"female reference")
    config.update(
        default_audio={"male": str(tmp_path / "other-male.wav"), "female": str(female_default)},
        features={"performance_direction": True},
        streaming_tts={
            "checkpoint_path": str(tmp_path / "streaming.json"),
            "poll_interval_seconds": 0.01,
            "minimum_batch_size": 1,
            "maximum_batch_size": 10,
            "maximum_batch_wait_seconds": 0.01,
        },
    )
    config["voxcpm"]["force_all_characters"] = True
    dialogues = [
        {"speaker": "Main", "text": "wait for direction", "chapter": "one"},
        {"speaker": "Extra", "text": "emotion fallback", "chapter": "one"},
    ]
    genders = {"Main": {"gender": "male"}, "Extra": {"gender": "female"}}
    snapshots = [
        ({"0": {"performance_control": "low and deliberate"}}, False),
        (
            {
                "0": {"performance_control": "low and deliberate"},
                "1": {"performance_control": "warm and unhurried"},
            },
            True,
        ),
    ]
    batches = []

    def fake_snapshot(*_args):
        return snapshots.pop(0) if snapshots else (
            {
                "0": {"performance_control": "low and deliberate"},
                "1": {"performance_control": "warm and unhurried"},
            },
            True,
        )

    def fake_voxcpm(tasks, _config):
        batches.append([task["index"] for task in tasks])
        for task in tasks:
            write_test_wav(Path(task["output_path"]))
        return [{"index": task["index"], "status": "ok"} for task in tasks]

    monkeypatch.setattr(run_full, "load_partial_performance_results", fake_snapshot)
    monkeypatch.setattr(run_full, "run_voxcpm_tasks", fake_voxcpm)

    segments = run_full.run_streaming_tts(config, dialogues, "novel", genders, {})

    assert batches == [[0], [1]]
    assert all(Path(segment["audio_path"]).is_file() for segment in segments)
    checkpoint = json.loads((tmp_path / "streaming.json").read_text(encoding="utf-8"))
    assert checkpoint["active_batch"] == []
    assert set(checkpoint["segments"]) == {"0", "1"}
    assert checkpoint["segments"]["0"]["style_control"] == run_full.compact_performance_control(
        "low and deliberate", speaker="Main", emotion=""
    )
    assert checkpoint["segments"]["1"]["engine"] == "voxcpm"
    assert checkpoint["segments"]["1"]["style_control"] == run_full.compact_performance_control(
        "warm and unhurried", speaker="Extra", emotion=""
    )


def test_bgm_cache_requires_exact_manifest_and_files(tmp_path):
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    manifest_path = bgm_dir / "bgm_manifest.json"
    segments = [
        {"segment_index": 1, "bgm_type": "calm"},
        {"segment_index": 4, "bgm_type": "tense"},
    ]
    write_bgm_cache(segments, manifest_path, bgm_dir, clips_per_segment=2)

    assert run_full.validate_bgm_cache(segments, manifest_path, bgm_dir, 2) == []

    changed = [segments[0], {"segment_index": 5, "bgm_type": "tense"}]
    problems = run_full.validate_bgm_cache(changed, manifest_path, bgm_dir, 2)
    assert any("segment IDs" in problem for problem in problems)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments"]["1"]["clips"][0]["seed"] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = run_full.validate_bgm_cache(segments, manifest_path, bgm_dir, 2)
    assert "segment 1 clip 0 seed does not match" in problems

    (bgm_dir / "001_1.mp3").unlink()
    problems = run_full.validate_bgm_cache(segments, manifest_path, bgm_dir, 2)
    assert "missing BGM clip: 001_1.mp3" in problems


def test_bgm_cache_does_not_accept_fixed_46_file_threshold(tmp_path):
    bgm_dir = tmp_path / "bgm"
    bgm_dir.mkdir()
    for index in range(46):
        (bgm_dir / f"legacy_{index}.mp3").write_bytes(b"old")

    problems = run_full.validate_bgm_cache(
        [{"segment_index": 1, "bgm_type": "calm"}],
        bgm_dir / "bgm_manifest.json",
        bgm_dir,
        3,
    )

    assert problems
    assert any("manifest" in problem for problem in problems)


def test_illustration_cache_binds_audited_plan_novel_cards_and_endpoint(tmp_path):
    from app.core.visual_prompt_auditor import (
        VISUAL_PROMPT_PIPELINE_VERSION,
        visual_prompt_source_hash,
    )
    from scripts.generate_illustrations import (
        CHECKPOINT_VERSION,
        apply_audited_prompts,
        generation_prompt_hash,
        generation_source_hash,
    )

    plan = [{
        "title": "scene",
        "description": "wheat field",
        "reason": "opening beat",
        "characters": [],
        "composition": "wide shot",
        "prompt": "wind-bent wheat",
        "start_line": 1,
        "end_line": 1,
    }]
    audit_result = {"illustration_index": 0, "audited_prompt": "literal wind-bent wheat only"}
    plan_path = tmp_path / "plan.json"
    novel_path = tmp_path / "novel.txt"
    cards_path = tmp_path / "cards.md"
    directory = tmp_path / "images"
    checkpoint_path = tmp_path / "images.checkpoint.json"
    audit_path = tmp_path / "audit.checkpoint.json"
    directory.mkdir()
    image_path = directory / "0001_scene.png"
    image_path.write_bytes(b"png")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    novel_path.write_text("Wheat bends in wind.", encoding="utf-8")
    cards_path.write_text("No characters.", encoding="utf-8")
    source_hash = visual_prompt_source_hash(
        novel_path.read_text(encoding="utf-8"),
        plan,
        cards_path.read_text(encoding="utf-8"),
    )
    audit_path.write_text(
        json.dumps({
            "pipeline_version": VISUAL_PROMPT_PIPELINE_VERSION,
            "source_hash": source_hash,
            "model": run_full.SENSENOVA_FLASH_LITE_MODEL,
            "total_items": 1,
            "completed_indices": [0],
            "results": [audit_result],
            "errors": {},
        }),
        encoding="utf-8",
    )
    audited_plan = apply_audited_prompts(plan, [audit_result])
    endpoint = "https://agnes.test/v1/images/generations"
    checkpoint_path.write_text(
        json.dumps({
            "version": CHECKPOINT_VERSION,
            "provider": "agnes",
            "model": "agnes-image-2.1-flash",
            "endpoint": endpoint,
            "size": "896x1152",
            "source_hash": generation_source_hash(plan),
            "audit_source_hash": source_hash,
            "images": [{
                "index": 0,
                "status": "success",
                "output_file": str(image_path),
                "prompt_hash": generation_prompt_hash(audited_plan[0]["prompt"]),
            }],
        }),
        encoding="utf-8",
    )

    args = (
        plan_path,
        directory,
        checkpoint_path,
        audit_path,
    )
    kwargs = {
        "expected_model": "agnes-image-2.1-flash",
        "expected_size": "896x1152",
        "expected_endpoint": endpoint,
        "novel_path": novel_path,
        "character_cards": cards_path,
    }
    assert run_full.validate_illustrations(*args, **kwargs) == []

    novel_path.write_text("The source changed.", encoding="utf-8")
    problems = run_full.validate_illustrations(*args, **kwargs)
    assert "visual prompt audit checkpoint is incomplete or incompatible" in problems

    novel_path.write_text("Wheat bends in wind.", encoding="utf-8")
    problems = run_full.validate_illustrations(*args, **dict(kwargs, expected_endpoint="https://other.test"))
    assert "illustration checkpoint endpoint does not match configuration" in problems


def test_step_tts_routes_engines_and_resumes_by_fingerprint(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    (tmp_path / "main.wav").write_bytes(b"reference")
    dialogues = [
        {"speaker": "Main", "text": "cloned line", "chapter": "one"},
        {"speaker": "Extra", "text": "preset line", "chapter": "one"},
    ]
    genders = {"Main": {"gender": "male"}, "Extra": {"gender": "female"}}
    calls = {"voxcpm": 0, "edge": 0}

    def fake_voxcpm(tasks, _config):
        calls["voxcpm"] += len(tasks)
        for task in tasks:
            write_test_wav(Path(task["output_path"]))
        return [{"index": task["index"], "status": "ok"} for task in tasks]

    def fake_edge(_text, voice_id, path):
        assert voice_id == "female-test"
        calls["edge"] += 1
        write_test_wav(path)

    monkeypatch.setattr(run_full, "run_voxcpm_tasks", fake_voxcpm)
    monkeypatch.setattr(run_full, "synthesize_edge_tts_to_wav", fake_edge)

    segments = run_full.step_tts(config, dialogues, genders, {})
    assert calls == {"voxcpm": 1, "edge": 1}
    assert all(Path(item["audio_path"]).is_file() for item in segments)
    assert run_full.validate_tts_manifest(config) == []

    run_full.step_tts(config, dialogues, genders, {})
    assert calls == {"voxcpm": 1, "edge": 1}

    assert run_full.validate_tts_manifest(config, dialogues, genders, {}) == []
    (tmp_path / "main.wav").write_bytes(b"changed reference audio")
    problems = run_full.validate_tts_manifest(config, dialogues, genders, {})
    assert "segment manifest does not belong to current TTS inputs" in problems


def test_performance_style_enters_voxcpm_fingerprint_and_task(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    reference_path = tmp_path / "main.wav"
    reference_path.write_bytes(b"reference")
    dialogue = {"speaker": "Main", "text": "hold the line", "chapter": "one"}
    genders = {"Main": {"gender": "male"}}
    emotion = {"0": {"emotion": "calm", "tone": "soft"}}
    captured_tasks = []

    def fake_voxcpm(tasks, _config):
        assert len(tasks) == 1
        captured_tasks.append(dict(tasks[0]))
        write_test_wav(Path(tasks[0]["output_path"]))
        return [{"index": tasks[0]["index"], "status": "ok"}]

    monkeypatch.setattr(run_full, "run_voxcpm_tasks", fake_voxcpm)

    first_performance = {"0": {"performance_control": "克制、轻声，语速舒缓"}}
    run_full.step_tts(config, [dialogue], genders, emotion, first_performance)

    assignment = run_full.get_voice_assignment("Main", "male", config)
    first_task = captured_tasks[0]
    first_parent = run_full.make_tts_task(
        config, 0, dialogue, "male", emotion["0"], first_performance["0"]
    )
    assert first_task["style_control"] == "克制，轻声，语速舒缓"
    assert first_task["fingerprint"] == first_parent["chunks"][0]["fingerprint"]

    second_performance = {"0": {"performance_control": "严肃，吐字清晰，语速稍快"}}
    run_full.step_tts(config, [dialogue], genders, emotion, second_performance)

    assert len(captured_tasks) == 2
    assert captured_tasks[1]["style_control"] == "严肃，语速稍快"
    assert captured_tasks[1]["fingerprint"] != first_task["fingerprint"]

    edge_assignment = {"engine": "edge-tts", "voice_id": "female-test"}
    assert run_full.tts_fingerprint(
        dialogue, edge_assignment, emotion["0"], first_performance["0"], config
    ) == run_full.tts_fingerprint(
        dialogue, edge_assignment, emotion["0"], second_performance["0"], config
    )


def test_voxcpm_child_script_compiles_and_reuses_reference_cache(tmp_path):
    config = make_config(tmp_path)
    reference_path = tmp_path / "main.wav"
    tasks = [
        {
            "index": index,
            "text": "line with 'quotes'\nand a newline",
            "output_path": str(tmp_path / "output" / "segments" / f"{index:05d}.wav"),
            "fingerprint": f"fingerprint-{index}",
            "reference_audio": str(reference_path),
            "style_control": "measured and clear",
        }
        for index in range(2)
    ]

    script = run_full.create_voxcpm_script(tasks, config)

    compile(script, "_batch_voxcpm.py", "exec")
    assert "prompt_caches = {}" in script
    assert "if reference not in prompt_caches:" in script
    assert "build_prompt_cache(reference_wav_path=reference)" in script
    assert "prompt_cache=prompt_caches[reference]" in script
    assert "max_len=4096" in script
    assert "badcase remained after retries" in script
    assert "audio is anomalously fast; retrying a fresh VoxCPM take" in script
    assert "correct_fast_audio" not in script
    assert "atempo=" not in script
    assert ".tempo.wav" not in script
    assert "os.replace(temporary_wav, task[\"output_path\"])" in script
    assert '"wav_sha256": file_sha256(path)' in script
    assert "if False:" in script  # normalize defaults off, matching VoxCPM's public API


def test_recover_voxcpm_tasks_requires_matching_successful_nonempty_outputs(tmp_path):
    config = make_config(tmp_path)
    checkpoint_path = tmp_path / "output" / "voxcpm_results.json"
    checkpoint_path.parent.mkdir(parents=True)
    tasks = []
    for index in range(6):
        output_path = tmp_path / "output" / "segments" / f"{index:05d}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if index < 4:
            with wave.open(str(output_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(48000)
                handle.writeframes(b"\0\0" * 4800)
        elif index == 4:
            output_path.write_bytes(b"")
        tasks.append(
            {
                "index": index,
                "fingerprint": f"fingerprint-{index}",
                "output_path": str(output_path),
            }
        )

    results = {
        str(task["index"]): {
            "index": task["index"],
            "status": "ok",
            "fingerprint": task["fingerprint"],
            "output_path": task["output_path"],
        }
        for task in tasks
    }
    for task in tasks[:4]:
        results[str(task["index"])].update(
            run_full.inspect_generated_wav(Path(task["output_path"])) or {}
        )
    results["1"]["status"] = "error"
    results["2"]["fingerprint"] = "stale-fingerprint"
    results["3"]["output_path"] = str(tmp_path / "other.wav")
    checkpoint_path.write_text(
        json.dumps(
            {
                "version": run_full.VOXCPM_BATCH_CHECKPOINT_VERSION,
                "source_hash": run_full.voxcpm_batch_source_hash(tasks, config),
                "results": results,
            }
        ),
        encoding="utf-8",
    )

    recovered, pending = run_full.recover_voxcpm_tasks(tasks, config)

    assert [item["index"] for item in recovered] == [0]
    assert [item["index"] for item in pending] == [1, 2, 3, 4, 5]

    Path(tasks[0]["output_path"]).write_bytes(b"corrupted")
    recovered, pending = run_full.recover_voxcpm_tasks(tasks, config)
    assert recovered == []
    assert [item["index"] for item in pending] == [0, 1, 2, 3, 4, 5]

    stale_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    stale_checkpoint["source_hash"] = "stale-source-hash"
    checkpoint_path.write_text(json.dumps(stale_checkpoint), encoding="utf-8")

    recovered, pending = run_full.recover_voxcpm_tasks(tasks, config)
    assert recovered == []
    assert pending == tasks


def test_main_can_resume_directly_from_tts(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    dialogues = [{"speaker": "Main", "text": "line", "chapter": "one"}]
    characters = ["Main"]
    captured = {}

    monkeypatch.setattr(run_full, "load_config", lambda _path: config)
    monkeypatch.setattr(run_full, "step_parse", lambda _config: (dialogues, characters, "novel"))
    monkeypatch.setattr(run_full, "require_gender_results", lambda *_args: {"Main": {"gender": "male"}})
    monkeypatch.setattr(run_full, "require_emotion_results", lambda *_args: {"0": {"emotion": "calm"}})

    def fake_tts(_config, parsed_dialogues, genders, emotions, _performances):
        captured.update(
            dialogues=parsed_dialogues,
            genders=genders,
            emotions=emotions,
        )
        return []

    monkeypatch.setattr(run_full, "step_tts", fake_tts)

    result = run_full.main(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--from-stage", "tts",
            "--to-stage", "tts",
            "--log", str(tmp_path / "run.log"),
        ]
    )

    assert result == 0
    assert captured == {
        "dialogues": dialogues,
        "genders": {"Main": {"gender": "male"}},
        "emotions": {"0": {"emotion": "calm"}},
    }


def test_main_from_tts_requires_and_passes_enabled_performance_cache(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config["features"] = {"emotion_label": True, "performance_direction": True}
    dialogues = [{"speaker": "Main", "text": "line", "chapter": "one"}]
    characters = ["Main"]
    performances = {"0": {"performance_control": "low and deliberate"}}
    events = []

    monkeypatch.setattr(run_full, "load_config", lambda _path: config)
    monkeypatch.setattr(run_full, "step_parse", lambda _config: (dialogues, characters, "novel"))

    def fake_gender(*_args):
        events.append("gender")
        return {"Main": {"gender": "male"}}

    def fake_emotion(*_args):
        events.append("emotion")
        return {"0": {"emotion": "calm"}}

    def fake_performance(*_args):
        events.append("performance")
        return performances

    def fake_tts(_config, _dialogues, _genders, _emotions, performance_results):
        events.append("tts")
        assert performance_results is performances
        return []

    monkeypatch.setattr(run_full, "require_gender_results", fake_gender)
    monkeypatch.setattr(run_full, "require_emotion_results", fake_emotion)
    monkeypatch.setattr(run_full, "require_performance_results", fake_performance)
    monkeypatch.setattr(run_full, "step_tts", fake_tts)

    result = run_full.main(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--from-stage", "tts",
            "--to-stage", "tts",
            "--log", str(tmp_path / "run.log"),
        ]
    )

    assert result == 0
    assert events == ["gender", "emotion", "performance", "tts"]


def test_step_tts_stops_on_edge_failure_and_keeps_checkpoint(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    dialogues = [{"speaker": "Extra", "text": "line", "chapter": "one"}]

    def fail_edge(*_args):
        raise RuntimeError("network down")

    monkeypatch.setattr(run_full, "synthesize_edge_tts_to_wav", fail_edge)

    with pytest.raises(run_full.PipelineError, match="edge-tts failed"):
        run_full.step_tts(config, dialogues, {"Extra": {"gender": "male"}}, {})

    manifest = json.loads(
        (tmp_path / "output/segments/segments_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["segments"] == {}


def test_execute_stage_records_failure(tmp_path):
    recorder = run_full.PipelineRecorder(tmp_path / "run.json", ["video"])

    def fail():
        raise RuntimeError("ffmpeg failed")

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        run_full.execute_stage(recorder, "video", fail)

    payload = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert payload["run_status"] == "failed"
    assert payload["stages"]["video"]["status"] == "failed"
    assert "ffmpeg failed" in payload["stages"]["video"]["error"]


def test_stage_slice_rejects_reverse_range():
    assert run_full.stage_slice("emotion", "tts") == (
        "emotion",
        "performance",
        "tts",
    )
    assert run_full.stage_slice("bgm-mix", "video") == (
        "bgm-mix",
        "illustration-plan",
        "illustrations",
        "video",
    )
    with pytest.raises(ValueError):
        run_full.stage_slice("video", "parse")


def test_dry_run_tts_checks_performance_as_direct_prerequisite(monkeypatch):
    checked = []

    def fake_stage_cache_status(stage, _config):
        checked.append(stage)
        return True, f"{stage} ready"

    monkeypatch.setattr(run_full, "stage_cache_status", fake_stage_cache_status)

    assert run_full.dry_run_report({}, ("tts",)) is True
    assert checked == ["performance", "tts"]
