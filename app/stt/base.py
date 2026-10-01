"""Contracts shared by all STT providers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    segments: list[Segment]


class STTProvider(Protocol):
    def transcribe(self, audio_path: Path, *, task: str = "transcribe") -> Transcription:
        """Convert a supported audio file to timestamped text."""
