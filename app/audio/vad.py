"""A small, transparent energy-based voice activity detector for VOCA."""

from dataclasses import dataclass

import numpy as np


@dataclass
class VoiceActivityDetector:
    """Track speech and trailing silence from sequential mono audio chunks."""

    sample_rate: int
    threshold_db: float
    minimum_speech_seconds: float
    silence_seconds: float
    speech_started: bool = False
    speech_duration: float = 0.0
    trailing_silence: float = 0.0

    def process(self, chunk: np.ndarray) -> bool:
        """Process one chunk and return whether it contains audible speech."""
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return False

        duration = samples.size / self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(samples))))
        level_db = 20 * np.log10(max(rms, 1e-10))
        is_speech = level_db >= self.threshold_db

        if is_speech:
            self.speech_started = True
            self.speech_duration += duration
            self.trailing_silence = 0.0
        elif self.speech_started:
            self.trailing_silence += duration
        return is_speech

    @property
    def should_stop(self) -> bool:
        """True once meaningful speech has ended with enough trailing silence."""
        # Chunk durations are floating-point values; one sample of tolerance avoids
        # waiting for an extra chunk when the target duration is represented as 0.999... .
        tolerance = 1 / self.sample_rate
        return (
            self.speech_duration >= self.minimum_speech_seconds - tolerance
            and self.trailing_silence >= self.silence_seconds - tolerance
        )
