import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import rebuild_speaker_voice as rebuild  # noqa: E402
from scripts import run_full as pipeline  # noqa: E402


def write_wav(path: Path, frames: int = 2400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * frames)


def task(index: int, path: Path, fingerprint: str) -> dict:
    return {
        "index": index,
        "output_path": str(path),
        "fingerprint": fingerprint,
        "entry": {
            "index": index,
            "speaker": "测试角色",
            "engine": "voxcpm",
            "fingerprint": fingerprint,
            "audio_path": str(path),
        },
    }


def test_speaker_indices_only_selects_requested_character():
    dialogues = [
        {"speaker": "赫萝"},
        {"speaker": "罗伦斯"},
        {"speaker": "赫萝"},
        {"speaker": ""},
    ]
    assert rebuild.speaker_indices(dialogues, "赫萝") == {0, 2}
    assert rebuild.speaker_indices(dialogues, "旁白") == {3}


def test_build_variant_config_does_not_modify_source_config(tmp_path):
    source = pipeline.load_config(str(ROOT / "config/config.yaml"))
    old_reference = source["characters"]["赫萝"]
    target = tmp_path / "variant"
    config_path = target / "rebuild_config.yaml"
    reference = ROOT / "backend/data/presets/design_female_gentle.wav"

    variant = rebuild.build_variant_config(
        source,
        speaker="赫萝",
        reference_audio=reference,
        target_dir=target,
        config_path=config_path,
    )

    assert source["characters"]["赫萝"] == old_reference
    assert variant["characters"]["赫萝"] == str(reference.resolve())
    assert Path(variant["output"]["dir"]) == target.resolve()
    assert Path(variant["illustrations"]["output_dir"]) == pipeline.illustration_output_dir(source)
    assert Path(variant["video"]["output_path"]).parent == target.resolve()
    assert variant["streaming_tts"]["enabled"] is False


def test_seed_variant_cache_reuses_other_speaker_but_not_target(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_tasks = [
        task(0, source_dir / "segments/00000.wav", "same-0"),
        task(1, source_dir / "segments/00001.wav", "old-target-1"),
    ]
    variant_tasks = [
        task(0, target_dir / "segments/00000.wav", "same-0"),
        task(1, target_dir / "segments/00001.wav", "new-target-1"),
    ]
    for item in source_tasks:
        write_wav(Path(item["output_path"]))
    source_manifest = {
        "segments": {
            str(item["index"]): pipeline.completed_tts_entry(item)
            for item in source_tasks
        }
    }
    manifest_path = target_dir / "segments/segments_manifest.json"

    counts = rebuild.seed_variant_tts_cache(
        source_manifest=source_manifest,
        source_tasks=source_tasks,
        variant_tasks=variant_tasks,
        target_indices={1},
        variant_manifest_path=manifest_path,
        variant_source_hash="new-source",
    )

    assert counts["linked"] + counts["copied"] == 1
    assert counts["target_pending"] == 1
    assert Path(variant_tasks[0]["output_path"]).is_file()
    assert not Path(variant_tasks[1]["output_path"]).exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload["segments"]) == {"0"}
    assert payload["source_hash"] == "new-source"


def test_split_cache_requires_fresh_manifest_and_all_outputs(tmp_path):
    video = tmp_path / "video.mp4"
    part = tmp_path / "part_01.mp4"
    video.write_bytes(b"video")
    part.write_bytes(b"part")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "strict_max_duration_ms": 3_600_000,
                "inputs": {
                    "portrait": {
                        "path": str(video.resolve()),
                        "size_bytes": video.stat().st_size,
                    }
                },
                "outputs": {"portrait": [{"path": str(part.resolve())}]},
            }
        ),
        encoding="utf-8",
    )
    # Ensure the manifest is at least as new as its source dependency.
    manifest.touch()

    assert rebuild.split_cache_valid(manifest, {"portrait": video}, 60.0)
    part.unlink()
    assert not rebuild.split_cache_valid(manifest, {"portrait": video}, 60.0)
