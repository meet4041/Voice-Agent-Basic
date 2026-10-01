"""Microphone recording and WAV persistence."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from app.audio.vad import VoiceActivityDetector


class RecordingError(RuntimeError):
    """Raised when audio cannot be captured from the selected microphone."""


class SpeechNotDetectedError(RecordingError):
    """Raised when listening ends without enough detected speech."""


@dataclass(frozen=True)
class Recording:
    samples: np.ndarray
    sample_rate: int


def record_microphone(duration_seconds: float, sample_rate: int) -> Recording:
    """Capture mono float32 audio from the default input device."""
    if duration_seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    try:
        samples = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as error:  # sounddevice exposes platform-specific errors
        raise RecordingError(
            "We couldn't access your microphone. Please check your microphone permissions "
            "and make sure an input device is connected."
        ) from error

    return Recording(samples=np.asarray(samples).reshape(-1), sample_rate=sample_rate)


def record_until_silence(
    *,
    sample_rate: int,
    maximum_duration_seconds: float,
    minimum_speech_seconds: float,
    silence_seconds: float,
    threshold_db: float,
) -> Recording:
    """Listen until speech is followed by sustained silence or time runs out."""
    if minimum_speech_seconds <= 0 or silence_seconds <= 0 or maximum_duration_seconds <= 0:
        raise ValueError("VAD durations must be greater than zero.")

    block_size = 1024
    detector = VoiceActivityDetector(
        sample_rate=sample_rate,
        threshold_db=threshold_db,
        minimum_speech_seconds=minimum_speech_seconds,
        silence_seconds=silence_seconds,
    )
    captured_chunks: list[np.ndarray] = []
    elapsed_seconds = 0.0

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size) as stream:
            while elapsed_seconds < maximum_duration_seconds:
                chunk, _overflowed = stream.read(block_size)
                mono_chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                elapsed_seconds += mono_chunk.size / sample_rate
                if detector.speech_started:
                    captured_chunks.append(mono_chunk)
                elif detector.process(mono_chunk):
                    captured_chunks.append(mono_chunk)
                    continue
                else:
                    continue

                detector.process(mono_chunk)
                if detector.should_stop:
                    break
    except Exception as error:  # sounddevice exposes platform-specific errors
        raise RecordingError(
            "We couldn't access your microphone. Please check your microphone permissions "
            "and make sure an input device is connected."
        ) from error

    if not detector.speech_started or detector.speech_duration < minimum_speech_seconds:
        raise SpeechNotDetectedError(
            "We didn't detect enough speech. Please speak a little louder and try again."
        )
    return Recording(samples=np.concatenate(captured_chunks), sample_rate=sample_rate)


def save_wav(recording: Recording, path: Path) -> Path:
    """Write a recording as a standard mono WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, recording.sample_rate, recording.samples.astype(np.float32))
    return path
