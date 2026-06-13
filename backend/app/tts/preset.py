import io
from typing import List, Optional

import edge_tts

from app.tts.base import TTSProvider, VoiceInfo, VoiceProfile


# Predefined Chinese voices (fallback: one male, one female)
PRESET_VOICES = [
    VoiceInfo(voice_id="zh-CN-YunxiNeural", name="云希 (男声)", provider="edge-tts", gender="male"),
    VoiceInfo(voice_id="zh-CN-XiaoxiaoNeural", name="晓晓 (女声)", provider="edge-tts", gender="female"),
]


class EdgeTTSProvider(TTSProvider):
    """TTS provider using edge-tts (Microsoft Edge TTS)."""

    def __init__(self):
        self._voices = PRESET_VOICES

    @property
    def name(self) -> str:
        return "edge-tts"

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        **params,
    ) -> bytes:
        """Synthesize speech using edge-tts."""
        # edge-tts doesn't support SSML through Communicate, use plain text
        communicate = edge_tts.Communicate(text, voice_id)
        audio_data = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])

        return audio_data.getvalue()

    async def clone_voice(self, audio_path: str, voice_name: str) -> VoiceProfile:
        """Edge-tts doesn't support voice cloning."""
        raise NotImplementedError("edge-tts does not support voice cloning. Use VoxCPM instead.")

    async def get_voices(self) -> List[VoiceInfo]:
        """Get list of available preset voices."""
        return self._voices

    async def check_available(self) -> bool:
        """Check if edge-tts is available."""
        try:
            # try to list voices
            voices = await edge_tts.list_voices()
            return len(voices) > 0
        except Exception:
            return False
