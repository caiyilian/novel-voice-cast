import sys, asyncio, wave, io
sys.path.insert(0, "..")

from app.tts.preset import EdgeTTSProvider, PRESET_VOICES
from app.tts.manager import TTSProviderManager


async def test_provider_availability():
    """Test if edge-tts is available."""
    provider = EdgeTTSProvider()
    available = await provider.check_available()
    print(f"[{'PASS' if available else 'SKIP'}] Provider available: {available}")
    return available


async def test_list_voices():
    """Test listing available voices."""
    provider = EdgeTTSProvider()
    voices = await provider.get_voices()
    assert len(voices) > 0
    print(f"[PASS] List voices: {len(voices)} voices")
    for v in voices[:3]:
        print(f"  {v.voice_id}: {v.name} ({v.gender})")


async def test_synthesize():
    """Test synthesizing speech."""
    provider = EdgeTTSProvider()
    audio = await provider.synthesize("你好，我是罗伦斯。", "zh-CN-YunxiNeural")
    assert audio is not None
    assert len(audio) > 0
    print(f"[PASS] Synthesize: {len(audio)} bytes")

    # check it starts with MP3 header (ff fb or f3 f0)
    header = audio[:4]
    is_mp3 = header[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2')
    is_id3 = header[:3] == b'ID3'
    print(f"  Format: {'MP3' if is_mp3 or is_id3 else 'Unknown'} ({len(audio)} bytes)")


async def test_synthesize_with_emotion():
    """Test synthesizing with emotion parameters."""
    provider = EdgeTTSProvider()

    # test with emotion
    audio = await provider.synthesize(
        "你好，我是罗伦斯。",
        "zh-CN-YunxiNeural",
        emotion="happy",
        tone="gentle",
    )
    assert audio is not None
    assert len(audio) > 0
    print(f"[PASS] Synthesize with emotion: {len(audio)} bytes")


async def test_clone_voice_raises():
    """Test that clone_voice raises NotImplementedError."""
    provider = EdgeTTSProvider()
    try:
        await provider.clone_voice("test.wav", "test")
        print("[FAIL] Should have raised NotImplementedError")
    except NotImplementedError:
        print("[PASS] Clone voice raises NotImplementedError")


async def test_manager_integration():
    """Test integration with TTSProviderManager."""
    manager = TTSProviderManager()
    provider = EdgeTTSProvider()
    manager.register(provider)

    # map a voice
    manager.map_voice("zh-CN-YunxiNeural", "edge-tts")

    # synthesize through manager
    audio = await manager.synthesize("你好", "zh-CN-YunxiNeural")
    assert audio is not None
    assert len(audio) > 0
    print(f"[PASS] Manager integration: {len(audio)} bytes")

    # check available
    status = await manager.check_available_providers()
    assert status.get("edge-tts") is True
    print(f"[PASS] Provider status: {status}")


async def main():
    print("=== Edge-TTS Provider Tests ===\n")

    available = await test_provider_availability()
    if not available:
        print("\nedge-tts not available, skipping remaining tests")
        return

    await test_list_voices()
    await test_synthesize()
    await test_synthesize_with_emotion()
    await test_clone_voice_raises()
    await test_manager_integration()

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
