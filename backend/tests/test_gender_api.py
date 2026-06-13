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

        # 2. upload novel (use proper name format)
        test_novel = """第一章 初见

「你好，我叫罗伦斯。」——罗伦斯
他是二十五岁的旅行商人，独自行商已有七个年头。

「汝是谁？」——赫萝
她有着美丽的脸孔，头上长着像小狗的耳朵。
「咱的名字是赫萝。」——赫萝

罗伦斯拔出短剑，指着赫萝。
「你到底是谁？」
赫萝瞇起带点红色的琥珀色眼睛。
「胆敢拿剑指着咱，真是不懂礼貌。」
"""
        r = await client.post(
            f"/api/project/{pid}/upload",
            files={"file": ("test.txt", test_novel.encode("utf-8"), "text/plain")},
        )
        assert r.status_code == 200
        print(f"[PASS] Upload novel: {r.json()}")

        # 3. check initial genders
        r = await client.get(f"/api/project/{pid}/characters")
        chars = r.json()["characters"]
        print(f"[PASS] Initial characters:")
        for c in chars:
            print(f"  {c['name']}: gender={c['gender']}")
        assert all(c["gender"] == "unknown" for c in chars)

        # 4. identify genders
        r = await client.post(
            f"/api/project/{pid}/characters/identify-genders",
            files={"file": ("test.txt", test_novel.encode("utf-8"), "text/plain")},
        )
        assert r.status_code == 200
        result = r.json()
        print(f"[PASS] Identify genders: {result['identified']}/{result['total_characters']} identified")
        for gr in result["results"]:
            print(f"  {gr['character_name']}: {gr['gender']} (confidence={gr['confidence']})")

        # 5. verify database updated
        r = await client.get(f"/api/project/{pid}/characters")
        chars = r.json()["characters"]
        print(f"[PASS] Updated characters:")
        for c in chars:
            print(f"  {c['name']}: gender={c['gender']}")
        
        # check that at least some genders are updated
        updated = sum(1 for c in chars if c["gender"] != "unknown")
        assert updated > 0, "No genders were updated"
        print(f"[PASS] {updated} characters gender updated in database")

        # 6. manual override still works
        r = await client.patch(
            f"/api/project/{pid}/characters/罗伦斯",
            json={"gender": "male"},
        )
        assert r.status_code == 200
        assert r.json()["gender"] == "male"
        print("[PASS] Manual override still works")

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
