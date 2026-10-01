"""Contracts shared by text-to-speech providers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Voice:
    id: str
    name: str


@dataclass(frozen=True)
class SpeechAudio:
    path: Path
    voice: Voice | None
    rate: int


class TTSProvider(Protocol):
    def list_voices(self) -> list[Voice]:
        """Return voices available on the current machine."""

    def synthesize(self, text: str, output_path: Path, *, voice_id: str | None, rate: int) -> SpeechAudio:
        """Create spoken audio from text and save it to ``output_path``."""
