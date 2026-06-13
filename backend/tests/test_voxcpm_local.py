"""Test VoxCPM from local models directory."""
import sys
sys.path.insert(0, 'models')

from voxcpm import VoxCPM
import soundfile as sf

print("Loading VoxCPM model from local...")
model = VoxCPM.from_pretrained('models/VoxCPM2', load_denoiser=False)
SR = model.tts_model.sample_rate
print(f"Model loaded! SR={SR}Hz")

print("Generating test audio...")
wav = model.generate(text="你好，这是本地测试。", cfg_value=2.0, inference_timesteps=10)
sf.write('data/presets/voxcpm_local_test.wav', wav, SR)
print(f"Done! Saved to data/presets/voxcpm_local_test.wav ({len(wav)/SR:.2f}s)")
