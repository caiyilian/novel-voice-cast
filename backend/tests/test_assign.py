import sys, asyncio, wave, struct
sys.path.insert(0, "..")

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.database import engine
from app.models import Base


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

    # create test audio files
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

        # 3. upload audio for 罗伦斯
        with open("test_audio.wav", "rb") as f:
            audio_content = f.read()
        r = await client.post(
            f"/api/project/{pid}/audio/upload",
            params={"character_name": "罗伦斯"},
            files={"file": ("test.wav", audio_content, "audio/wav")},
        )
        assert r.status_code == 200
        audio_id = r.json()["id"]
        print(f"[PASS] Upload audio: id={audio_id}")

        # 4. assign audio to 赫萝 (reassign)
        r = await client.post(
            f"/api/project/{pid}/characters/赫萝/assign",
            json={"source": "upload", "audio_id": audio_id},
        )
        assert r.status_code == 200
        print(f"[PASS] Assign audio: {r.json()}")

        # 5. assign preset
        r = await client.post(
            f"/api/project/{pid}/characters/罗伦斯/assign",
            json={"source": "preset_male"},
        )
        assert r.status_code == 200
        print(f"[PASS] Assign preset: {r.json()}")

        # 6. unassign audio
        r = await client.delete(f"/api/project/{pid}/characters/赫萝/assign")
        assert r.status_code == 200
        print(f"[PASS] Unassign audio: {r.json()}")

        # 7. 404 for non-existent character
        r = await client.post(
            f"/api/project/{pid}/characters/不存在/assign",
            json={"source": "upload", "audio_id": audio_id},
        )
        assert r.status_code == 404
        print("[PASS] 404 for non-existent character")

        # 8. invalid source
        r = await client.post(
            f"/api/project/{pid}/characters/罗伦斯/assign",
            json={"source": "invalid"},
        )
        assert r.status_code == 400
        print("[PASS] Invalid source rejected")

        # 9. missing audio_id for upload
        r = await client.post(
            f"/api/project/{pid}/characters/罗伦斯/assign",
            json={"source": "upload"},
        )
        assert r.status_code == 400
        print("[PASS] Missing audio_id rejected")

    # cleanup
    import os
    os.remove("test_audio.wav")

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
