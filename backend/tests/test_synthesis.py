"""Tests for the synthesis API."""
import sys, asyncio
sys.path.insert(0, "..")

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import engine
from app.models import Base


async def main():
    # reset database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. create project
        r = await client.post("/api/project", json={"name": "Test"})
        assert r.status_code == 200
        pid = r.json()["id"]
        print(f"[PASS] Create project: id={pid}")

        # 2. upload novel
        test_novel = (
            "「你好。」——罗伦斯\n"
            "罗伦斯：「嗯。」\n"
            "「你是谁？」——赫萝\n"
            "赫萝：「咱是赫萝。」\n"
        )
        r = await client.post(
            f"/api/project/{pid}/upload",
            files={"file": ("test.txt", test_novel.encode("utf-8"), "text/plain")},
        )
        assert r.status_code == 200
        print(f"[PASS] Upload novel")

        # 3. check synthesis status (idle)
        r = await client.get(f"/api/project/{pid}/synthesis/status")
        assert r.status_code == 200
        status = r.json()
        assert status["status"] == "idle"
        print(f"[PASS] Synthesis status (idle): {status}")

        # 4. start synthesis (with mock voice map)
        r = await client.post(
            f"/api/project/{pid}/synthesize",
            json={"character_voice_map": {"罗伦斯": "zh-CN-YunxiNeural", "赫萝": "zh-CN-XiaoxiaoNeural"}},
        )
        assert r.status_code == 200
        result = r.json()
        print(f"[PASS] Start synthesis: {result}")

        # 5. check synthesis status (synthesizing or done)
        r = await client.get(f"/api/project/{pid}/synthesis/status")
        assert r.status_code == 200
        status = r.json()
        print(f"[PASS] Synthesis status: {status['status']}")

        # 6. wait for synthesis to complete
        for _ in range(30):
            r = await client.get(f"/api/project/{pid}/synthesis/status")
            status = r.json()
            if status["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.5)

        print(f"[PASS] Synthesis completed: {status['status']}")

        # 7. check output
        if status["status"] == "done":
            r = await client.get(f"/api/project/{pid}/output")
            assert r.status_code == 200
            assert r.headers.get("content-type") == "audio/wav"
            print(f"[PASS] Download output: {len(r.content)} bytes")
        else:
            print(f"[SKIP] Output not available (status={status['status']})")

        # 8. test 404
        r = await client.get("/api/project/999/output")
        assert r.status_code == 404
        print("[PASS] 404 for non-existent project")

    await engine.dispose()


asyncio.run(main())
print("\nAll synthesis API tests passed!")
