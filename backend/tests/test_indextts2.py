"""
Test IndexTTS2 provider.
Run with: E:\projects\音色克隆\index-tts2\index-tts-main\.venv\python.exe tests\test_indextts2.py
"""
import sys
# Add local models directory to path (Python will find indextts and wetext packages there)
sys.path.insert(0, "E:/projects/novel-voice-cast/backend/models")
sys.path.insert(0, "E:/projects/novel-voice-cast/backend")

import time
from app.tts.indextts2_provider import IndexTTS2Provider

OUTPUT_DIR = "E:/projects/novel-voice-cast/backend/data/presets"

print("=== IndexTTS2 Provider Test ===\n")

# 1. Test provider initialization
print("1. Initializing provider...")
provider = IndexTTS2Provider(
    voice_dir=f"{OUTPUT_DIR}/../voices",
)
print(f"   Provider name: {provider.name}")
print(f"   Model dir: {provider.model_dir}")

# 2. Check availability
print("\n2. Checking availability...")
available = provider.check_available()
print(f"   Available: {available}")

# 3. Load model
print("\n3. Loading model (this takes ~25-30 seconds)...")
t0 = time.time()
provider._ensure_model()
print(f"   Model loaded in {time.time() - t0:.1f}s")

# 4. Clone voice (zero-shot, just copies file)
print("\n4. Cloning voice...")
import os
ref_audio = "E:/projects/音色克隆/index-tts2/index-tts-main/examples/voice_01.wav"
if os.path.exists(ref_audio):
    import asyncio
    profile = asyncio.run(provider.clone_voice(ref_audio, "Test Voice"))
    print(f"   Cloned: voice_id={profile.voice_id}, name={profile.name}")

    # 5. Synthesize with cloned voice
    print("\n5. Synthesizing audio...")
    t0 = time.time()
    audio = asyncio.run(provider.synthesize(
        text="你好，我是罗伦斯。今天天气真不错。",
        voice_id=profile.voice_id,
        emotion="happy",
        tone="gentle",
    ))
    elapsed = time.time() - t0
    print(f"   Synthesized in {elapsed:.1f}s, {len(audio)} bytes")

    # save
    output_path = f"{OUTPUT_DIR}/indextts2_test.wav"
    with open(output_path, "wb") as f:
        f.write(audio)
    print(f"   Saved: {output_path}")

    # 6. Test with different emotions
    print("\n6. Testing different emotions...")
    emotions = [
        ("calm", "serious", "indextts2_calm.wav"),
        ("sad", "soft", "indextts2_sad.wav"),
        ("angry", "loud", "indextts2_angry.wav"),
        ("happy", "gentle", "indextts2_happy.wav"),
    ]
    for emo, tone, filename in emotions:
        t0 = time.time()
        audio = asyncio.run(provider.synthesize(
            text="你好，我是罗伦斯。",
            voice_id=profile.voice_id,
            emotion=emo,
            tone=tone,
        ))
        elapsed = time.time() - t0
        output_path = f"{OUTPUT_DIR}/{filename}"
        with open(output_path, "wb") as f:
            f.write(audio)
        print(f"   {emo}+{tone}: {elapsed:.1f}s -> {filename}")

    print("\n=== All tests passed! ===")
else:
    print(f"   Reference audio not found: {ref_audio}")
    print("   Skipping voice clone and synthesis tests")
