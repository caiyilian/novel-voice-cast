"""
pyttsx3 TTS Provider — 离线语音合成
"""
import io
import os
import tempfile
from typing import List, Optional

import pyttsx3

from app.tts.base import TTSProvider, VoiceInfo, VoiceProfile


class Pyttsx3Provider(TTSProvider):
    """pyttsx3 离线 TTS Provider"""

    def __init__(self):
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = pyttsx3.init()

    @property
    def name(self) -> str:
        return "pyttsx3"

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        **params,
    ) -> bytes:
        """Synthesize speech using pyttsx3."""
        self._ensure_engine()

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self._engine.save_to_file(text, temp_path)
            self._engine.runAndWait()

            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def clone_voice(self, audio_path: str, voice_name: str) -> VoiceProfile:
        """pyttsx3 doesn't support voice cloning."""
        raise NotImplementedError("pyttsx3 does not support voice cloning.")

    async def get_voices(self) -> List[VoiceInfo]:
        """Get list of available voices."""
        self._ensure_engine()
        voices = self._engine.getProperty("voices")
        result = []
        for v in voices:
            gender = "female" if "female" in v.name.lower() or "zira" in v.name.lower() else "male"
            result.append(VoiceInfo(
                voice_id=v.id,
                name=v.name,
                provider="pyttsx3",
                gender=gender,
            ))
        return result

    async def check_available(self) -> bool:
        """Check if pyttsx3 is available."""
        try:
            self._ensure_engine()
            return True
        except Exception:
            return False
