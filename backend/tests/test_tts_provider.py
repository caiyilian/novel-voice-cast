import sys
sys.path.insert(0, "..")

import asyncio
from app.tts.base import TTSProvider, VoiceProfile, VoiceInfo
from app.tts.manager import TTSProviderManager


class MockProvider(TTSProvider):
    """Mock TTS provider for testing."""

    def __init__(self, name: str, voices: list = None):
        self._name = name
        self._voices = voices or [
            VoiceInfo(voice_id=f"{name}_male", name=f"{name} Male", provider=name, gender="male"),
            VoiceInfo(voice_id=f"{name}_female", name=f"{name} Female", provider=name, gender="female"),
        ]

    @property
    def name(self) -> str:
        return self._name

    async def synthesize(self, text: str, voice_id: str, emotion=None, tone=None, **params) -> bytes:
        return f"audio from {self._name}: {text}".encode()

    async def clone_voice(self, audio_path: str, voice_name: str) -> VoiceProfile:
        return VoiceProfile(
            voice_id=f"{self._name}_cloned_{voice_name}",
            name=voice_name,
            audio_path=audio_path,
        )

    async def get_voices(self) -> list:
        return self._voices


def test_provider_registration():
    """Test provider registration and retrieval."""
    manager = TTSProviderManager()
    provider = MockProvider("test")

    manager.register(provider)
    assert manager.get_provider("test") is provider
    assert manager.get_provider("nonexistent") is None
    print("[PASS] Provider registration")


def test_voice_mapping():
    """Test voice to provider mapping."""
    manager = TTSProviderManager()
    provider = MockProvider("test")
    manager.register(provider)

    manager.map_voice("test_male", "test")
    assert manager.get_provider_for_voice("test_male") is provider
    assert manager.get_provider_for_voice("unknown") is None
    print("[PASS] Voice mapping")


def test_unregister():
    """Test provider unregistration."""
    manager = TTSProviderManager()
    provider = MockProvider("test")
    manager.register(provider)
    manager.map_voice("test_male", "test")

    manager.unregister("test")
    assert manager.get_provider("test") is None
    assert manager.get_provider_for_voice("test_male") is None
    print("[PASS] Provider unregistration")


async def test_synthesize():
    """Test synthesis through manager."""
    manager = TTSProviderManager()
    provider = MockProvider("test")
    manager.register(provider)
    manager.map_voice("test_male", "test")

    audio = await manager.synthesize("Hello", "test_male")
    assert audio is not None
    assert len(audio) > 0
    print("[PASS] Synthesize")


async def test_clone_voice():
    """Test voice cloning through manager."""
    manager = TTSProviderManager()
    provider = MockProvider("test")
    manager.register(provider)

    profile = await manager.clone_voice("/path/to/audio.wav", "my_voice")
    assert profile.voice_id.startswith("test_cloned_")
    assert profile.name == "my_voice"
    print("[PASS] Clone voice")


async def test_get_all_voices():
    """Test getting voices from all providers."""
    manager = TTSProviderManager()
    provider1 = MockProvider("provider1")
    provider2 = MockProvider("provider2")
    manager.register(provider1)
    manager.register(provider2)

    voices = await manager.get_all_voices()
    assert len(voices) == 4  # 2 from each provider
    print("[PASS] Get all voices")


async def test_check_available():
    """Test provider availability check."""
    manager = TTSProviderManager()
    provider = MockProvider("test")
    manager.register(provider)

    status = await manager.check_available_providers()
    assert status["test"] is True
    print("[PASS] Check available")


def run_async_tests():
    """Run all async tests."""
    asyncio.run(test_synthesize())
    asyncio.run(test_clone_voice())
    asyncio.run(test_get_all_voices())
    asyncio.run(test_check_available())


if __name__ == "__main__":
    test_provider_registration()
    test_voice_mapping()
    test_unregister()
    run_async_tests()
    print("\nAll TTS provider tests passed!")
