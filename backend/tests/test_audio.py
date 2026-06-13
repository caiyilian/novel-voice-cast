import sys, asyncio, os, wave, struct
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

    # create test audio file
    test_audio = "test_audio.wav"
    create_test_wav(test_audio, 500)

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
        print(f"[PASS] Upload novel: {r.json()}")

        # 3. upload audio
        with open(test_audio, "rb") as f:
            audio_content = f.read()
        r = await client.post(
            f"/api/project/{pid}/audio/upload",
            params={"character_name": "罗伦斯"},
            files={"file": ("test.wav", audio_content, "audio/wav")},
        )
        assert r.status_code == 200
        audio_id = r.json()["id"]
        print(f"[PASS] Upload audio: id={audio_id}, path={r.json()['file_path']}")

        # 4. get audio info
        r = await client.get(f"/api/audio/{audio_id}")
        assert r.status_code == 200
        print(f"[PASS] Get audio info: {r.json()}")

        # 5. download audio
        r = await client.get(f"/api/audio/{audio_id}/download")
        assert r.status_code == 200
        assert len(r.content) > 0
        print(f"[PASS] Download audio: {len(r.content)} bytes")

        # 6. upload another audio
        with open(test_audio, "rb") as f:
            audio_content = f.read()
        r = await client.post(
            f"/api/project/{pid}/audio/upload",
            params={"character_name": "赫萝"},
            files={"file": ("test.wav", audio_content, "audio/wav")},
        )
        assert r.status_code == 200
        audio_id2 = r.json()["id"]
        print(f"[PASS] Upload another audio: id={audio_id2}")

        # 7. delete audio
        r = await client.delete(f"/api/project/{pid}/audio/{audio_id}")
        assert r.status_code == 200
        print(f"[PASS] Delete audio: {r.json()}")

        # 8. verify deleted
        r = await client.get(f"/api/audio/{audio_id}")
        assert r.status_code == 404
        print("[PASS] Verify deleted: 404")

        # 9. 404 for non-existent audio
        r = await client.get("/api/audio/999")
        assert r.status_code == 404
        print("[PASS] 404 for non-existent audio")

        # 10. invalid format
        with open(test_audio, "rb") as f:
            audio_content = f.read()
        r = await client.post(
            f"/api/project/{pid}/audio/upload",
            params={"character_name": "罗伦斯"},
            files={"file": ("test.txt", audio_content, "text/plain")},
        )
        assert r.status_code == 400
        print(f"[PASS] Invalid format rejected: {r.json()}")

    # cleanup
    os.remove(test_audio)

    await engine.dispose()


asyncio.run(main())
print("\nAll tests passed!")
