import json
from pathlib import Path

from scripts.progress_monitor import ProgressCollector, _size_info


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_size_info_reduces_configured_portrait_ratio():
    assert _size_info("896x1152") == {
        "raw": "896x1152",
        "width": 896,
        "height": 1152,
        "aspect_ratio": "7:9",
        "orientation": "portrait",
    }


def test_collector_combines_audit_generation_video_and_live_process(tmp_path):
    config = tmp_path / "config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
illustrations:
  output_dir: output/images
  checkpoint_path: output/images.checkpoint.json
  prompt_audit_checkpoint_path: backend/data/audit.checkpoint.json
  size: 896x1152
  steps: 25
  cfg: 7.0
  provider: local-http
  endpoint: http://127.0.0.1:8000/generate
  landscape:
    enabled: true
    size: 1280x720
    output_dir: output/images-landscape
    checkpoint_path: output/images-landscape.checkpoint.json
video:
  output_path: output/final.mp4
  subtitle_path: output/final.srt
  landscape:
    enabled: true
    output_path: output/final-landscape.mp4
    subtitle_path: output/final-landscape.srt
""".strip(),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "output/run_full_manifest.json",
        {
            "run_status": "running",
            "selected_stages": ["illustrations", "video"],
            "stages": {"illustrations": {"status": "failed", "error": "previous run"}},
        },
    )
    _write_json(
        tmp_path / "backend/data/audit.checkpoint.json",
        {
            "model": "sensenova-6.7-flash-lite",
            "total_items": 4,
            "completed_indices": [0, 1],
            "results": [{"illustration_index": 0}, {"illustration_index": 1}],
            "errors": {},
        },
    )
    image = tmp_path / "output/images/0001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    _write_json(
        tmp_path / "output/images.checkpoint.json",
        {
            "images": [
                {"index": 0, "title": "one", "status": "success", "attempts": 1, "output_file": str(image)},
                {"index": 1, "title": "two", "status": "running", "attempts": 2},
                {"index": 2, "title": "three", "status": "pending", "attempts": 0},
                {"index": 3, "title": "four", "status": "failed", "attempts": 5},
            ]
        },
    )

    collector = ProgressCollector(
        tmp_path,
        Path("config/config.yaml"),
        process_probe=lambda: [{"pid": 123, "command": "run_full.py"}],
    )
    status = collector.collect()

    assert status["pipeline_running"] is True
    assert status["phase"]["code"] == "prompt-audit"
    assert status["audit"]["completed"] == 2
    assert status["audit"]["next_index"] == 3
    assert status["audit"]["percent"] == 50.0
    assert status["image_generation"]["success"] == 1
    assert status["image_generation"]["running"] == 1
    assert status["image_generation"]["failed"] == 1
    assert status["image_generation"]["output_count"] == 1
    assert status["image"]["aspect_ratio"] == "7:9"
    assert [item["aspect_ratio"] for item in status["image_sizes"]] == ["7:9", "16:9"]
    assert [item["name"] for item in status["image_variants"]] == ["portrait", "landscape"]
    assert len(status["videos"]) == 2
    assert next(item for item in status["stages"] if item["name"] == "illustrations")["status"] == "running"
