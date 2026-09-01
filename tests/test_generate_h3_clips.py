from pathlib import Path

import pytest

from scripts.generate_h3_clips import (
    H3Client,
    H3GenerationError,
    H3JobResumeRequired,
    atomic_write_json,
    choose_duration,
    download_completed_job,
    h3_frame_count,
    wait_for_job,
)


class _Response:
    status_code = 200

    def json(self):
        return {"job_id": "job-123"}


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        files = kwargs.get("files") or {}
        self.calls.append(
            {
                "url": url,
                "json": kwargs.get("json"),
                "data": kwargs.get("data"),
                "files": sorted(files),
            }
        )
        return _Response()


def test_h3_duration_uses_deployed_17k_plus_5_frame_rule():
    assert h3_frame_count(5) == 124
    assert h3_frame_count(10) == 243
    assert choose_duration(10.0, minimum_seconds=5, maximum_seconds=10) == (
        9,
        h3_frame_count(9) / 24,
    )
    assert choose_duration(5.0, minimum_seconds=5, maximum_seconds=10) == (None, None)


def test_h3_client_uses_json_for_t2v_and_multipart_for_i2v(tmp_path: Path):
    client = H3Client("http://h3.example/")
    session = _Session()
    client.session = session

    assert client.submit_request(prompt="scene", width=864, height=480, duration=5) == "job-123"

    frame = tmp_path / "continuation.png"
    frame.write_bytes(b"png")
    assert client.submit_request(
        prompt="continue",
        width=864,
        height=480,
        duration=5,
        first_frame=frame,
    ) == "job-123"

    assert session.calls[0]["json"] == {
        "prompt": "scene",
        "width": 864,
        "height": 480,
        "duration": 5,
    }
    assert session.calls[0]["data"] is None
    assert session.calls[1]["json"] is None
    assert session.calls[1]["data"]["width"] == "864"
    assert session.calls[1]["files"] == ["first_frame"]


def test_job_timeout_requests_local_resume_without_declaring_remote_failure():
    class Client:
        def status(self, _job_id):
            raise AssertionError("expired timeout should stop before another request")

    with pytest.raises(H3JobResumeRequired, match="did not finish"):
        wait_for_job(Client(), "still-valid", poll_seconds=0, job_timeout=-1)


def test_job_timeout_excludes_transient_server_outage(monkeypatch):
    class Client:
        calls = 0

        def status(self, _job_id):
            self.calls += 1
            if self.calls == 1:
                raise H3GenerationError("temporary connection failure")
            return {"status": "completed", "progress": 100}

    clock = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr("scripts.generate_h3_clips.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("scripts.generate_h3_clips.time.sleep", lambda _seconds: None)

    result = wait_for_job(Client(), "survives-outage", poll_seconds=0, job_timeout=1.5)

    assert result["status"] == "completed"


def test_completed_job_download_network_error_preserves_remote_job(tmp_path: Path):
    class Client:
        def download(self, _job_id, _output):
            raise OSError("temporary disk problem")

    with pytest.raises(H3JobResumeRequired, match="completed but local download failed"):
        download_completed_job(Client(), "completed-job", tmp_path / "clip.mp4")


def test_atomic_checkpoint_retries_transient_windows_destination_lock(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "checkpoint.json"
    real_replace = __import__("os").replace
    attempts = []

    def flaky_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError(5, "destination is briefly locked")
        real_replace(source, destination)

    monkeypatch.setattr("scripts.generate_h3_clips.os.replace", flaky_replace)
    monkeypatch.setattr("scripts.generate_h3_clips.time.sleep", lambda _seconds: None)

    atomic_write_json(target, {"completed": 170})

    assert len(attempts) == 3
    assert target.read_text(encoding="utf-8").find('"completed": 170') >= 0
    assert not list(tmp_path.glob(".*.tmp"))
