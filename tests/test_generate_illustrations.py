import base64
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_illustrations as gi  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, *, payload=None, text="", headers=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self.content = content

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses, *, downloads=None, before_post=None):
        self.responses = list(responses)
        self.downloads = list(downloads or [])
        self.before_post = before_post
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if self.before_post:
            self.before_post()
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.downloads.pop(0)


def image_payload(content=b"png-bytes"):
    encoded = base64.b64encode(content).decode("ascii")
    return {"data": [{"b64_json": encoded}]}


def test_load_api_key_prefers_environment_and_supports_key_file(tmp_path):
    key_file = tmp_path / "agneskey.txt"
    key_file.write_text("file-secret\n", encoding="utf-8")

    assert gi.load_api_key(key_file, environ={"AGNES_API_KEY": " env-secret "}) == "env-secret"
    assert gi.load_api_key(key_file, environ={}) == "file-secret"


def test_generation_prompt_marks_comparison_targets_as_absent():
    prompt = "Wheat makes wave patterns that look like packs of wolves running through the field."

    prepared = gi.build_generation_prompt(prompt)

    assert "comparison targets" in prepared
    assert "packs of wolves running through the field" in prepared
    assert "must be absent" in prepared


def test_apply_audited_prompts_preserves_original_and_decision_chain():
    plans = [{"title": "field", "prompt": "literal wolves in wheat"}]
    audits = [{
        "audited_prompt": "wind-bent wheat forming layered golden waves",
        "decision_path": "final_adjudication",
    }]

    result = gi.apply_audited_prompts(plans, audits)

    assert result[0]["original_prompt"] == "literal wolves in wheat"
    assert result[0]["prompt"] == "wind-bent wheat forming layered golden waves"
    assert result[0]["prompt_audit"] == audits[0]
    assert plans[0]["prompt"] == "literal wolves in wheat"


def test_retry_is_exponential_respects_retry_after_and_redacts_key(caplog):
    secret = "agnes-super-secret"
    session = FakeSession(
        [
            FakeResponse(
                429,
                text=f"rate limited for Bearer {secret}",
                headers={"Retry-After": "7"},
            ),
            FakeResponse(503, text="temporarily unavailable"),
            FakeResponse(payload=image_payload(b"generated")),
        ]
    )
    sleeps = []
    attempts = []
    caplog.set_level(logging.INFO, logger="generate_illustrations")
    client = gi.AgnesImageClient(
        api_key=secret,
        endpoint="https://example.test/v1/images/generations",
        proxy=None,
        session=session,
        sleep_fn=sleeps.append,
        random_fn=lambda low, high: 1.0,
        interval_min=1.0,
        interval_max=2.0,
        backoff_base=2.0,
    )

    result = client.generate("draw this", on_attempt=attempts.append)

    assert result.content == b"generated"
    assert result.source == "base64"
    assert attempts == [1, 2, 3]
    assert sleeps == [1.0, 7.0, 1.0, 4.0, 1.0]
    assert session.posts[0][1]["json"] == {
        "model": gi.DEFAULT_MODEL,
        "prompt": gi.build_generation_prompt("draw this"),
        "size": gi.DEFAULT_SIZE,
        "extra_body": {"response_format": "b64_json"},
    }
    assert "metaphor" in session.posts[0][1]["json"]["prompt"]
    assert secret not in caplog.text


def test_checkpoint_tracks_each_state_and_resume_handles_url(tmp_path):
    output_dir = tmp_path / "images"
    checkpoint_path = tmp_path / "checkpoint.json"
    plan = [
        {"title": "first", "prompt": "first prompt"},
        {"title": "second", "prompt": "second prompt"},
    ]
    checkpoint_snapshots = []

    def capture_checkpoint():
        checkpoint_snapshots.append(
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
        )

    first_session = FakeSession(
        [
            FakeResponse(payload=image_payload(b"first-image")),
            FakeResponse(400, text="invalid prompt"),
        ],
        before_post=capture_checkpoint,
    )
    first_client = gi.AgnesImageClient(
        api_key="secret",
        proxy=None,
        session=first_session,
        sleep_fn=lambda _seconds: None,
        interval_min=0,
        interval_max=0,
    )

    first_checkpoint = gi.run_generation(
        plan,
        client=first_client,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )

    assert [item["status"] for item in checkpoint_snapshots[0]["images"]] == [
        "running",
        "pending",
    ]
    assert [item["status"] for item in checkpoint_snapshots[1]["images"]] == [
        "success",
        "running",
    ]
    assert [item["status"] for item in first_checkpoint["images"]] == [
        "success",
        "failed",
    ]
    assert [item["attempts"] for item in first_checkpoint["images"]] == [1, 1]
    for record in first_checkpoint["images"]:
        assert record["started_at"]
        assert record["ended_at"]
        assert record["duration_seconds"] is not None
    assert Path(first_checkpoint["images"][0]["output_file"]).read_bytes() == b"first-image"
    assert first_checkpoint["images"][1]["error_summary"] == "HTTP 400: invalid prompt"
    assert not list(tmp_path.rglob("*.tmp"))

    second_session = FakeSession(
        [FakeResponse(payload={"data": [{"url": "https://images.test/second.png"}]})],
        downloads=[FakeResponse(content=b"second-image")],
    )
    second_client = gi.AgnesImageClient(
        api_key="secret",
        proxy=None,
        session=second_session,
        sleep_fn=lambda _seconds: None,
        interval_min=0,
        interval_max=0,
    )

    resumed = gi.run_generation(
        plan,
        client=second_client,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert len(second_session.posts) == 1
    assert len(second_session.gets) == 1
    assert [item["status"] for item in resumed["images"]] == ["success", "success"]
    assert [item["attempts"] for item in resumed["images"]] == [1, 2]
    assert Path(resumed["images"][1]["output_file"]).read_bytes() == b"second-image"
    assert not list(tmp_path.rglob("*.tmp"))


def test_legacy_checkpoint_cannot_mark_pulid_images_as_agnes_success(tmp_path):
    list_plan = tmp_path / "list-plan.json"
    dict_plan = tmp_path / "dict-plan.json"
    item = {"title": "legacy", "prompt": "prompt"}
    list_plan.write_text(json.dumps([item]), encoding="utf-8")
    dict_plan.write_text(json.dumps({"illustrations": [item]}), encoding="utf-8")
    assert gi.load_plan(list_plan) == gi.load_plan(dict_plan) == [item]

    output_dir = tmp_path / "images"
    output_path = output_dir / "0001_legacy.png"
    output_dir.mkdir()
    output_path.write_bytes(b"existing")
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text("[0]", encoding="utf-8")
    session = FakeSession([FakeResponse(payload=image_payload(b"agnes-regenerated"))])
    client = gi.AgnesImageClient(
        api_key="secret",
        proxy=None,
        session=session,
        sleep_fn=lambda _seconds: None,
        interval_min=0,
        interval_max=0,
    )

    checkpoint = gi.run_generation(
        [item],
        client=client,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert len(session.posts) == 1
    assert checkpoint["version"] == gi.CHECKPOINT_VERSION
    assert checkpoint["provider"] == "agnes"
    assert checkpoint["model"] == gi.DEFAULT_MODEL
    assert checkpoint["images"][0]["status"] == "success"
    assert checkpoint["images"][0]["output_file"] == str(output_path)
    assert output_path.read_bytes() == b"agnes-regenerated"
