"""Application-facing text-to-speech service."""

from pathlib import Path

from app.tts.base import SpeechAudio, TTSProvider, Voice
from app.tts.windows_sapi import WindowsSapiTTSProvider


class TextToSpeechService:
    def __init__(self, provider: TTSProvider | None = None) -> None:
        self.provider = provider or WindowsSapiTTSProvider()

    def available_voices(self) -> list[Voice]:
        return self.provider.list_voices()

    def create_speech(self, text: str, output_path: Path, *, voice_id: str | None, rate: int) -> SpeechAudio:
        return self.provider.synthesize(text, output_path, voice_id=voice_id, rate=rate)
