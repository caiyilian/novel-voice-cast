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

        # 2. upload novel with characters
        test_novel = (
            "第1章 开始\n"
            "「你好。」——罗伦斯\n"
            "罗伦斯：「嗯。」\n"
            "「你是谁？」——赫萝\n"
            "赫萝：「咱是赫萝。」\n"
            "「我叫罗伦斯。」——罗伦斯\n"
        )
        r = await client.post(
            f"/api/project/{pid}/upload",
            files={"file": ("test.txt", test_novel.encode("utf-8"), "text/plain")},
        )
        assert r.status_code == 200
        print(f"[PASS] Upload novel: {r.json()}")

        # 3. list characters
        r = await client.get(f"/api/project/{pid}/characters")
        assert r.status_code == 200
        chars = r.json()
        print(f"[PASS] List characters: {chars['total_characters']} characters")
        for c in chars["characters"]:
            print(f"  {c['name']}: gender={c['gender']}, dialogues={c['dialogue_count']}")

        # 4. get character detail
        r = await client.get(f"/api/project/{pid}/characters/罗伦斯")
        assert r.status_code == 200
        char = r.json()
        print(f"[PASS] Get character: {char['name']}, gender={char['gender']}")

        # 5. update character gender
        r = await client.patch(
            f"/api/project/{pid}/characters/罗伦斯",
            json={"gender": "male"},
        )
        assert r.status_code == 200
        assert r.json()["gender"] == "male"
        print(f"[PASS] Update gender: {r.json()['name']} -> {r.json()['gender']}")

        # 6. list character dialogues
        r = await client.get(f"/api/project/{pid}/characters/罗伦斯/dialogues")
        assert r.status_code == 200
        dls = r.json()
        print(f"[PASS] List dialogues: {dls['total_dialogues']} dialogues")
        for d in dls["dialogues"]:
            print(f"  [{d['order']}] {d['text'][:30]}")

        # 7. list dialogues with limit
        r = await client.get(f"/api/project/{pid}/characters/罗伦斯/dialogues?limit=2")
        assert r.status_code == 200
        assert r.json()["total_dialogues"] == 2
        print(f"[PASS] List dialogues with limit: {r.json()['total_dialogues']} dialogues")

        # 8. 404 for non-existent character
        r = await client.get(f"/api/project/{pid}/characters/不存在")
        assert r.status_code == 404
        print("[PASS] 404 for non-existent character")

        # 9. 404 for non-existent project
        r = await client.get("/api/project/999/characters")
        assert r.status_code == 404
        print("[PASS] 404 for non-existent project")

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
