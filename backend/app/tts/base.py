from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VoiceProfile:
    """Profile for a cloned voice."""
    voice_id: str
    name: str
    audio_path: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VoiceInfo:
    """Information about an available voice."""
    voice_id: str
    name: str
    provider: str
    is_preset: bool = True
    gender: Optional[str] = None


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        **params,
    ) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            emotion: Optional emotion label (happy, sad, etc.)
            tone: Optional tone label (loud, soft, etc.)
            **params: Additional parameters

        Returns:
            Audio bytes (wav format)
        """
        pass

    @abstractmethod
    async def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
    ) -> VoiceProfile:
        """Clone a voice from reference audio.

        Args:
            audio_path: Path to reference audio file
            voice_name: Name for the cloned voice

        Returns:
            VoiceProfile with voice_id for later synthesis
        """
        pass

    @abstractmethod
    async def get_voices(self) -> List[VoiceInfo]:
        """Get list of available voices.

        Returns:
            List of VoiceInfo objects
        """
        pass

    async def check_available(self) -> bool:
        """Check if this provider is available."""
        try:
            await self.get_voices()
            return True
        except Exception:
            return False
