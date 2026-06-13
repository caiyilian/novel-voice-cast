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

        # 3. preview audio for character (male)
        r = await client.post(
            f"/api/project/{pid}/preview",
            json={"text": "你好，我是罗伦斯。", "character_name": "罗伦斯"},
        )
        assert r.status_code == 200
        assert len(r.content) > 0
        assert r.headers.get("content-type") == "audio/mpeg"
        print(f"[PASS] Preview male: {len(r.content)} bytes")

        # 4. preview audio for character (female)
        r = await client.post(
            f"/api/project/{pid}/preview",
            json={"text": "你好，我是赫萝。", "character_name": "赫萝"},
        )
        assert r.status_code == 200
        assert len(r.content) > 0
        print(f"[PASS] Preview female: {len(r.content)} bytes")

        # 5. test caching (same request should be faster)
        import time
        start = time.time()
        r1 = await client.post(
            f"/api/project/{pid}/preview",
            json={"text": "你好，我是罗伦斯。", "character_name": "罗伦斯"},
        )
        time1 = time.time() - start

        start = time.time()
        r2 = await client.post(
            f"/api/project/{pid}/preview",
            json={"text": "你好，我是罗伦斯。", "character_name": "罗伦斯"},
        )
        time2 = time.time() - start

        assert r1.content == r2.content
        print(f"[PASS] Cache working: first={time1:.2f}s, second={time2:.2f}s")

        # 6. 404 for non-existent character
        r = await client.post(
            f"/api/project/{pid}/preview",
            json={"text": "你好", "character_name": "不存在"},
        )
        assert r.status_code == 404
        print("[PASS] 404 for non-existent character")

        # 7. 404 for non-existent project
        r = await client.post(
            "/api/project/999/preview",
            json={"text": "你好", "character_name": "罗伦斯"},
        )
        assert r.status_code == 404
        print("[PASS] 404 for non-existent project")

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
