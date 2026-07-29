import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from scripts import run_full
from scripts.desktop_events import DesktopEventEmitter


ROOT = Path(__file__).resolve().parents[1]


def event_payloads(text: str, prefix: str) -> list[dict]:
    marker = f"{prefix} "
    return [
        json.loads(line[len(marker) :])
        for line in text.splitlines()
        if line.startswith(marker)
    ]


def test_desktop_event_emitter_writes_utf8_single_line_json_and_monotonic_progress():
    stream = io.StringIO()
    emitter = DesktopEventEmitter(True, stream)
    emitter.configure(True, stream=stream, command="python run_full.py --novel 中文 小说.txt")

    emitter.stage("emotion", index=3, total=13, status="running", operation="正在标注情绪")
    emitter.progress("emotion", current=8, total=10, operation="第 8 项")
    emitter.progress("emotion", current=3, total=10, operation="重试第 3 项")
    emitter.log("info", "包含换行\n但仍是单条 JSON")

    output = stream.getvalue()
    assert len(output.splitlines()) == 4
    stages = event_payloads(output, "[STAGE]")
    progress = event_payloads(output, "[PROGRESS]")
    logs = event_payloads(output, "[LOG]")
    assert stages[0]["version"] == 1
    assert stages[0]["stage"] == "emotion"
    assert progress[0]["percent"] == 80.0
    assert progress[1]["percent"] == 80.0
    assert logs[0]["message"] == "包含换行\n但仍是单条 JSON"
    assert logs[0]["stage"] == "emotion"


def test_disabled_desktop_event_emitter_preserves_normal_cli_output():
    stream = io.StringIO()
    emitter = DesktopEventEmitter(False, stream)

    emitter.stage("parse", index=1, total=13, status="running")
    emitter.progress("parse", current=1, total=1, operation="完成")
    emitter.log("info", "不会输出")

    assert stream.getvalue() == ""


def test_execute_stage_emits_start_and_complete_events(tmp_path, monkeypatch):
    stream = io.StringIO()
    emitter = DesktopEventEmitter(True, stream)
    emitter.configure(True, stream=stream, command="test command")
    monkeypatch.setattr(run_full, "DESKTOP_EVENTS", emitter)
    recorder = run_full.PipelineRecorder(tmp_path / "manifest.json", ["parse"])

    result = run_full.execute_stage(
        recorder,
        "parse",
        lambda: "ok",
        [tmp_path / "novel.txt"],
    )

    assert result == "ok"
    stages = event_payloads(stream.getvalue(), "[STAGE]")
    progress = event_payloads(stream.getvalue(), "[PROGRESS]")
    assert [item["status"] for item in stages] == ["running", "complete"]
    assert [item["percent"] for item in progress] == [0.0, 100.0]
    assert stages[-1]["artifacts"] == [str(tmp_path / "novel.txt")]


def test_execute_stage_emits_failed_and_interrupted_states(tmp_path, monkeypatch):
    stream = io.StringIO()
    emitter = DesktopEventEmitter(True, stream)
    emitter.configure(True, stream=stream)
    monkeypatch.setattr(run_full, "DESKTOP_EVENTS", emitter)
    recorder = run_full.PipelineRecorder(tmp_path / "manifest.json", ["gender", "emotion"])

    def fail():
        raise RuntimeError("模型失败")

    def interrupt():
        raise KeyboardInterrupt

    try:
        run_full.execute_stage(recorder, "gender", fail)
    except RuntimeError:
        pass
    try:
        run_full.execute_stage(recorder, "emotion", interrupt)
    except KeyboardInterrupt:
        pass

    stages = event_payloads(stream.getvalue(), "[STAGE]")
    assert [item["status"] for item in stages] == [
        "running",
        "failed",
        "running",
        "interrupted",
    ]
    assert stages[1]["error"] == "模型失败"
    assert stages[3]["error"] == "interrupted by user"


def test_input_overrides_are_in_memory_only_and_resolve_chinese_paths(tmp_path):
    config_path = tmp_path / "配置.yaml"
    original = {
        "novel": {"text_path": "old.txt", "labels_path": "old-labels.txt"},
        "output": {"dir": str(tmp_path / "out")},
    }
    config_path.write_text(yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")
    novel = tmp_path / "中文 小说.txt"
    labels = tmp_path / "角色 标注.txt"
    novel.write_text("第一章\n旁白。", encoding="utf-8")
    labels.write_text("", encoding="utf-8")

    loaded = run_full.load_config(str(config_path))
    run_full.apply_input_overrides(loaded, novel_path=str(novel), labels_path=str(labels))

    assert loaded["novel"] == {
        "text_path": str(novel.resolve()),
        "labels_path": str(labels.resolve()),
    }
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == original


def test_stop_file_watcher_interrupts_once_and_stops_cleanly(tmp_path):
    requested = threading.Event()
    calls = []
    stop_file = tmp_path / "desktop.stop"
    watcher = run_full.StopFileWatcher(
        stop_file,
        interrupt=lambda: (calls.append("interrupt"), requested.set()),
        poll_seconds=0.02,
    )

    watcher.start()
    stop_file.write_text("stop", encoding="utf-8")
    assert requested.wait(timeout=2)
    watcher.stop()

    assert calls == ["interrupt"]


def test_real_parse_only_cli_emits_all_desktop_event_types(tmp_path):
    fixture = ROOT / "desktop" / "fixtures"
    config_path = tmp_path / "桌面 测试.yaml"
    config = yaml.safe_load((fixture / "config.yaml").read_text(encoding="utf-8"))
    config["output"]["dir"] = str(tmp_path / "output")
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_full.py"),
        "--config",
        str(config_path),
        "--novel",
        str(fixture / "novel.txt"),
        "--labels",
        str(fixture / "labels.txt"),
        "--from-stage",
        "parse",
        "--to-stage",
        "parse",
        "--desktop-events",
        "--log",
        str(tmp_path / "parse.log"),
    ]
    environment = {**os.environ, "PYTHONUTF8": "1"}

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    stages = event_payloads(completed.stdout, "[STAGE]")
    progress = event_payloads(completed.stdout, "[PROGRESS]")
    logs = event_payloads(completed.stdout, "[LOG]")
    assert [item["status"] for item in stages] == ["running", "complete"]
    assert progress[-1]["percent"] == 100.0
    assert any("解析完成" in item["message"] for item in logs)
    assert all(item["version"] == 1 for item in [*stages, *progress, *logs])
    manifest = json.loads(
        (tmp_path / "output" / "run_full_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_status"] == "complete"
