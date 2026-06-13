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
        # 1. health check
        r = await client.get("/health")
        assert r.status_code == 200
        print("[PASS] GET /health")

        # 2. create project
        r = await client.post("/api/project", json={"name": "Test Project"})
        assert r.status_code == 200
        pid = r.json()["id"]
        print(f"[PASS] POST /api/project -> id={pid}")

        # 3. upload novel with chapters
        test_novel = (
            "开头文字\n"
            "第1章 觉醒\n"
            "「你好。」——罗伦斯\n"
            "罗伦斯：「嗯。」\n"
            "第2章 风暴\n"
            "「发生了什么？」——赫萝\n"
            "「不知道。」——罗伦斯\n"
        )
        r = await client.post(
            f"/api/project/{pid}/upload",
            files={"file": ("test.txt", test_novel.encode("utf-8"), "text/plain")},
        )
        assert r.status_code == 200, f"upload failed: {r.status_code} {r.text}"
        result = r.json()
        print(f"[PASS] POST /api/project/{pid}/upload -> {result}")
        assert result["characters"] == 2
        assert result["dialogues"] == 4
        assert result["chapters"] >= 1
        assert result["chapter_method"] in ("llm", "regex")
        print(f"  chapter_method: {result['chapter_method']}")

        # 4. get project overview
        r = await client.get(f"/api/project/{pid}")
        assert r.status_code == 200
        overview = r.json()
        print(f"[PASS] GET /api/project/{pid} -> chapters={overview['chapter_count']}")

        # 5. get chapters list
        r = await client.get(f"/api/project/{pid}/chapters")
        assert r.status_code == 200
        chapters = r.json()
        print(f"[PASS] GET /api/project/{pid}/chapters -> {chapters['total_chapters']} chapters")
        for ch in chapters["chapters"]:
            print(f"  {ch['title']}: {ch['dialogue_count']} dialogues")

        # 6. test 404
        r = await client.get("/api/project/999/chapters")
        assert r.status_code == 404
        print("[PASS] GET /api/project/999/chapters -> 404")

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
