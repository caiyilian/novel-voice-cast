"""
VoxCPM2 TTS Provider — 音色克隆引擎对接

依赖:
  - voxcpm 包（pip install -e /path/to/VoxCPM）
  - PyTorch + CUDA
  - 模型文件位于 VoxCPM2/ 目录

使用前先加载模型（约 16 秒），之后保持单例复用。
"""
import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from app.tts.base import TTSProvider, VoiceInfo, VoiceProfile


class VoxCPMProvider(TTSProvider):
    """VoxCPM2 音色克隆 Provider"""

    def __init__(
        self,
        model_path: str = "./VoxCPM2",
        voice_dir: str = "./data/voices",
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.voice_dir = Path(voice_dir)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self._mapping_file = self.voice_dir / "voice_mapping.json"
        self._mapping: dict[str, VoiceProfile] = {}
        self._model = None
        self._device = device

        # 加载已有映射
        self._load_mapping()

    def _ensure_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from voxcpm import VoxCPM
                print(f"[VoxCPM] Loading model from {self.model_path}...")
                t0 = time.time()
                self._model = VoxCPM.from_pretrained(
                    self.model_path,
                    load_denoiser=False,
                    device=self._device,
                )
                print(f"[VoxCPM] Model loaded in {time.time() - t0:.1f}s")
            except ImportError:
                raise ImportError("voxcpm package not installed. Install with: pip install -e /path/to/VoxCPM")
            except Exception as e:
                raise RuntimeError(f"Failed to load VoxCPM model: {e}")

    @property
    def name(self) -> str:
        return "voxcpm"

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        **params,
    ) -> bytes:
        """Synthesize speech with cloned voice."""
        self._ensure_model()

        profile = self._mapping.get(voice_id)
        if not profile:
            raise ValueError(f"Voice not found: {voice_id}")

        # build style control from emotion/tone
        style_control = params.get("style_control")
        if not style_control and (emotion or tone):
            style_control = self._build_style_control(emotion, tone)

        # apply style control
        if style_control:
            final_text = f"({style_control}){text}"
        else:
            final_text = text

        # synthesize
        import soundfile as sf
        wav = self._model.generate(
            text=final_text,
            reference_wav_path=profile.audio_path,
            cfg_value=params.get("cfg_value", 2.0),
            inference_timesteps=params.get("inference_timesteps", 10),
        )

        # convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, wav, self._model.tts_model.sample_rate, format="WAV")
        return buf.getvalue()

    def _build_style_control(self, emotion: Optional[str], tone: Optional[str]) -> str:
        """Build style control string from emotion and tone."""
        parts = []

        # emotion mapping to Chinese descriptions
        emotion_map = {
            "happy": "欢快活泼",
            "sad": "低落悲伤",
            "angry": "愤怒生气",
            "surprised": "惊讶震惊",
            "calm": "平静冷静",
            "nervous": "紧张焦虑",
            "cold": "冷漠淡漠",
        }

        # tone mapping to Chinese descriptions
        tone_map = {
            "loud": "大声",
            "soft": "轻声",
            "whisper": "低语",
            "gentle": "温柔",
            "serious": "严肃",
            "sarcastic": "讽刺",
            "stutter": "结巴",
        }

        if emotion and emotion in emotion_map:
            parts.append(emotion_map[emotion])
        if tone and tone in tone_map:
            parts.append(tone_map[tone])

        return "，".join(parts) if parts else ""

    async def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
        **params,
    ) -> VoiceProfile:
        """Clone voice from reference audio."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        voice_id = f"voxcpm_{uuid.uuid4().hex[:8]}"

        # copy audio to voice_dir
        import shutil
        ext = Path(audio_path).suffix or ".wav"
        dest_path = self.voice_dir / f"{voice_id}{ext}"
        shutil.copy2(audio_path, str(dest_path))

        profile = VoiceProfile(
            voice_id=voice_id,
            name=voice_name,
            audio_path=str(dest_path),
            metadata={"source": "voxcpm", "created_at": time.time()},
        )

        self._mapping[voice_id] = profile
        self._save_mapping()
        return profile

    async def get_voices(self) -> List[VoiceInfo]:
        """Get list of cloned voices."""
        voices = []
        for profile in self._mapping.values():
            voices.append(VoiceInfo(
                voice_id=profile.voice_id,
                name=profile.name,
                provider="voxcpm",
                is_preset=False,
            ))
        return voices

    async def check_available(self) -> bool:
        """Check if VoxCPM is available."""
        try:
            from voxcpm import VoxCPM
            return os.path.exists(self.model_path)
        except ImportError:
            return False

    def _load_mapping(self):
        """Load voice mapping from file."""
        if self._mapping_file.exists():
            with open(self._mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                profile = VoiceProfile(
                    voice_id=item["voice_id"],
                    name=item["voice_name"],
                    audio_path=item["ref_audio_path"],
                    metadata=item.get("metadata", {}),
                )
                self._mapping[profile.voice_id] = profile

    def _save_mapping(self):
        """Save voice mapping to file."""
        data = []
        for profile in self._mapping.values():
            data.append({
                "voice_id": profile.voice_id,
                "voice_name": profile.name,
                "ref_audio_path": profile.audio_path,
                "metadata": profile.metadata,
            })
        with open(self._mapping_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _clone_voice_sync(self, audio_path: str, voice_name: str) -> VoiceProfile:
        """Synchronous version of clone_voice for testing."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        voice_id = f"voxcpm_{uuid.uuid4().hex[:8]}"

        import shutil
        ext = Path(audio_path).suffix or ".wav"
        dest_path = self.voice_dir / f"{voice_id}{ext}"
        shutil.copy2(audio_path, str(dest_path))

        profile = VoiceProfile(
            voice_id=voice_id,
            name=voice_name,
            audio_path=str(dest_path),
            metadata={"source": "voxcpm", "created_at": time.time()},
        )

        self._mapping[voice_id] = profile
        self._save_mapping()
        return profile

    def check_available_sync(self) -> bool:
        """Synchronous version of check_available."""
        try:
            from voxcpm import VoxCPM
            return os.path.exists(self.model_path)
        except ImportError:
            return False
