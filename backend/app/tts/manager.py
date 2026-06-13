from typing import Dict, List, Optional

from app.tts.base import TTSProvider, VoiceInfo, VoiceProfile


class TTSProviderManager:
    """Manages multiple TTS providers and routes requests."""

    def __init__(self):
        self._providers: Dict[str, TTSProvider] = {}
        self._voice_provider_map: Dict[str, str] = {}

    def register(self, provider: TTSProvider) -> None:
        """Register a TTS provider."""
        self._providers[provider.name] = provider

    def unregister(self, name: str) -> None:
        """Unregister a TTS provider."""
        self._providers.pop(name, None)
        # remove voice mappings
        to_remove = [vid for vid, pname in self._voice_provider_map.items() if pname == name]
        for vid in to_remove:
            del self._voice_provider_map[vid]

    def get_provider(self, name: str) -> Optional[TTSProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_provider_for_voice(self, voice_id: str) -> Optional[TTSProvider]:
        """Get the provider that owns a specific voice."""
        provider_name = self._voice_provider_map.get(voice_id)
        if provider_name:
            return self._providers.get(provider_name)
        return None

    def map_voice(self, voice_id: str, provider_name: str) -> None:
        """Map a voice_id to a provider."""
        self._voice_provider_map[voice_id] = provider_name

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        **params,
    ) -> bytes:
        """Synthesize speech using the appropriate provider.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            emotion: Optional emotion label
            tone: Optional tone label

        Returns:
            Audio bytes

        Raises:
            ValueError: If no provider found for voice_id
        """
        provider = self.get_provider_for_voice(voice_id)
        if not provider:
            # try to find any available provider
            for p in self._providers.values():
                try:
                    voices = await p.get_voices()
                    if any(v.voice_id == voice_id for v in voices):
                        provider = p
                        break
                except Exception:
                    continue

        if not provider:
            raise ValueError(f"No provider found for voice_id: {voice_id}")

        return await provider.synthesize(text, voice_id, emotion=emotion, tone=tone, **params)

    async def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
        provider_name: Optional[str] = None,
    ) -> VoiceProfile:
        """Clone a voice using the specified or first available provider.

        Args:
            audio_path: Path to reference audio
            voice_name: Name for the cloned voice
            provider_name: Specific provider to use (optional)

        Returns:
            VoiceProfile

        Raises:
            ValueError: If no provider available
        """
        provider = None
        if provider_name:
            provider = self._providers.get(provider_name)
        else:
            # use first available provider
            for p in self._providers.values():
                try:
                    if await p.check_available():
                        provider = p
                        break
                except Exception:
                    continue

        if not provider:
            raise ValueError("No TTS provider available for voice cloning")

        profile = await provider.clone_voice(audio_path, voice_name)
        self._voice_provider_map[profile.voice_id] = provider.name
        return profile

    async def get_all_voices(self) -> List[VoiceInfo]:
        """Get voices from all registered providers."""
        all_voices = []
        for provider in self._providers.values():
            try:
                voices = await provider.get_voices()
                all_voices.extend(voices)
            except Exception:
                continue
        return all_voices

    async def check_available_providers(self) -> Dict[str, bool]:
        """Check which providers are available."""
        status = {}
        for name, provider in self._providers.items():
            try:
                status[name] = await provider.check_available()
            except Exception:
                status[name] = False
        return status
