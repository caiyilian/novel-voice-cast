import sys, asyncio, os, wave, struct
sys.path.insert(0, "..")

from app.tts.voice_clone import VoxCPMProvider
from app.tts.base import VoiceProfile, VoiceInfo
from app.tts.manager import TTSProviderManager


def create_test_wav(filepath: str, duration_ms: int = 500):
    """Create a simple test WAV file."""
    sample_rate = 24000
    num_samples = int(sample_rate * duration_ms / 1000)
    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))


def test_provider_init():
    """Test provider initialization."""
    provider = VoxCPMProvider(
        model_path="./nonexistent",
        voice_dir="./test_voices",
    )
    assert provider.name == "voxcpm"
    print("[PASS] Provider initialization")

    # cleanup
    import shutil
    if os.path.exists("./test_voices"):
        shutil.rmtree("./test_voices")


def test_clone_voice():
    """Test voice cloning (without model)."""
    provider = VoxCPMProvider(
        model_path="./nonexistent",
        voice_dir="./test_voices",
    )

    # create test audio
    create_test_wav("./test_audio.wav", 500)

    # clone voice (should work without model)
    profile = provider._clone_voice_sync("./test_audio.wav", "Test Voice")
    assert profile.voice_id.startswith("voxcpm_")
    assert profile.name == "Test Voice"
    assert os.path.exists(profile.audio_path)
    print(f"[PASS] Clone voice: {profile.voice_id}")

    # check mapping saved
    assert profile.voice_id in provider._mapping
    print("[PASS] Voice mapping saved")

    # cleanup
    os.remove("./test_audio.wav")
    import shutil
    if os.path.exists("./test_voices"):
        shutil.rmtree("./test_voices")


def test_load_save_mapping():
    """Test voice mapping persistence."""
    provider = VoxCPMProvider(
        model_path="./nonexistent",
        voice_dir="./test_voices2",
    )

    # create test audio
    create_test_wav("./test_audio.wav", 500)

    # clone voice
    profile = provider._clone_voice_sync("./test_audio.wav", "Persistent Voice")

    # create new provider instance (should load from file)
    provider2 = VoxCPMProvider(
        model_path="./nonexistent",
        voice_dir="./test_voices2",
    )
    assert profile.voice_id in provider2._mapping
    assert provider2._mapping[profile.voice_id].name == "Persistent Voice"
    print("[PASS] Mapping persistence")

    # cleanup
    os.remove("./test_audio.wav")
    import shutil
    if os.path.exists("./test_voices2"):
        shutil.rmtree("./test_voices2")


def test_manager_integration():
    """Test integration with TTSProviderManager."""
    manager = TTSProviderManager()
    provider = VoxCPMProvider(
        model_path="./nonexistent",
        voice_dir="./test_voices3",
    )
    manager.register(provider)

    # check provider registered
    assert manager.get_provider("voxcpm") is provider
    print("[PASS] Manager integration")

    # cleanup
    import shutil
    if os.path.exists("./test_voices3"):
        shutil.rmtree("./test_voices3")


def test_check_available():
    """Test availability check."""
    provider = VoxCPMProvider(
        model_path="./nonexistent",
        voice_dir="./test_voices4",
    )
    assert provider.check_available_sync() is False
    print("[PASS] Check available (returns False for nonexistent)")

    # cleanup
    import shutil
    if os.path.exists("./test_voices4"):
        shutil.rmtree("./test_voices4")


if __name__ == "__main__":
    test_provider_init()
    test_clone_voice()
    test_load_save_mapping()
    test_manager_integration()
    test_check_available()
    print("\nAll VoxCPM provider tests passed!")
