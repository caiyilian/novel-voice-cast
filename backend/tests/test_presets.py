import sys, asyncio, wave, struct
sys.path.insert(0, "..")

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import engine
from app.models import Base
from app.config import PRESET_DIR


def create_test_wav(filepath: str, duration_ms: int = 100):
    """Create a simple test WAV file."""
    sample_rate = 24000
    num_samples = int(sample_rate * duration_ms / 1000)
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))


async def main():
    # reset database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # ensure presets exist
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    create_test_wav(str(PRESET_DIR / "default_male.wav"), 500)
    create_test_wav(str(PRESET_DIR / "default_female.wav"), 500)
    print("[SETUP] Created preset files")

    # create test audio
    create_test_wav("test_audio.wav", 500)

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

        # 3. list presets
        r = await client.get(f"/api/project/{pid}/presets")
        assert r.status_code == 200
        presets = r.json()
        print(f"[PASS] List presets: {presets['total']} presets")
        for p in presets["presets"]:
            print(f"  {p['name']}: {p['file_size']} bytes")

        # 4. download preset
        r = await client.get(f"/api/project/{pid}/presets/default_male/download")
        assert r.status_code == 200
        assert len(r.content) > 0
        print(f"[PASS] Download preset: {len(r.content)} bytes")

        # 5. 404 for non-existent preset
        r = await client.get(f"/api/project/{pid}/presets/nonexistent/download")
        assert r.status_code == 404
        print("[PASS] 404 for non-existent preset")

        # 6. get audio status (no assignments)
        r = await client.get(f"/api/project/{pid}/audio-status")
        assert r.status_code == 200
        status = r.json()
        print(f"[PASS] Audio status: {status['assigned_count']}/{status['total_characters']} assigned")
        for c in status["characters"]:
            print(f"  {c['character_name']}: {c['audio_source']}")

        # 7. upload audio and assign
        with open("test_audio.wav", "rb") as f:
            audio_content = f.read()
        r = await client.post(
            f"/api/project/{pid}/audio/upload",
            params={"character_name": "罗伦斯"},
            files={"file": ("test.wav", audio_content, "audio/wav")},
        )
        assert r.status_code == 200
        print(f"[PASS] Upload audio for 罗伦斯")

        # 8. check status again
        r = await client.get(f"/api/project/{pid}/audio-status")
        assert r.status_code == 200
        status = r.json()
        print(f"[PASS] Audio status after upload: {status['assigned_count']}/{status['total_characters']} assigned")
        for c in status["characters"]:
            print(f"  {c['character_name']}: {c['audio_source']} (id={c['audio_id']})")

    # cleanup
    import os
    os.remove("test_audio.wav")
    os.remove(str(PRESET_DIR / "default_male.wav"))
    os.remove(str(PRESET_DIR / "default_female.wav"))

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
