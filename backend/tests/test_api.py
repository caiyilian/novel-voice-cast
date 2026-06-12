import sys, asyncio, json
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
        # 1. health check
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        print("[PASS] GET /health")

        # 2. create project
        r = await client.post("/api/project", json={"name": "Wolf and Spice Vol.1"})
        assert r.status_code == 200
        proj = r.json()
        pid = proj["id"]
        print(f"[PASS] POST /api/project -> id={pid}")

        # 3. upload novel (use test data)
        test_novel = (
            "\u300c\u4f60\u7ec8\u4e8e\u6765\u4e86\u300d\u2014\u2014\u7f57\u4f26\u65af\n"
            "\u7f57\u4f26\u65af\uff1a\u300c\u55ef\uff0c\u8def\u4e0a\u6709\u70b9\u4e8b\u3002\u300d\n"
            "\u300c\u90a3\u5c31\u597d\u300d\u2014\u2014\u7f57\u4f26\u65af\n"
            "\u7b2c7\u7ae0 \u521d\u89c1\n"
            "\u300c\u4f60\u597d\u300d\u2014\u2014\u8d6b\u841d\n"
            "\u8d6b\u841d\uff1a\u300c\u55ef\uff0c\u521a\u521a\u597d\u300d\n"
        )
        r = await client.post(
            f"/api/project/{pid}/upload",
            files={"file": ("test.txt", test_novel.encode("utf-8"), "text/plain")},
        )
        assert r.status_code == 200, f"upload failed: {r.status_code} {r.text}"
        result = r.json()
        print(f"[PASS] POST /api/project/{pid}/upload -> {result}")

        # 4. get project overview
        r = await client.get(f"/api/project/{pid}")
        assert r.status_code == 200
        overview = r.json()
        print(f"[PASS] GET /api/project/{pid} -> {overview}")

        # 5. verify counts
        assert overview["character_count"] == 2, f"expected 2 chars, got {overview['character_count']}"
        assert overview["dialogue_count"] == 5, f"expected 5 dialogues, got {overview['dialogue_count']}"
        assert overview["chapter_count"] == 1, f"expected 1 chapter, got {overview['chapter_count']}"
        assert overview["status"] == "parsed", f"expected status=parsed, got {overview['status']}"
        print("[PASS] All assertions passed")

        # 6. test 404
        r = await client.get("/api/project/999")
        assert r.status_code == 404
        print("[PASS] GET /api/project/999 -> 404")

    await engine.dispose()

asyncio.run(main())
print("\nAll tests passed!")
