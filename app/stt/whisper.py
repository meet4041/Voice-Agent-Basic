"""OpenAI Whisper integration, isolated from the rest of the application."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly

from app.stt.base import Segment, Transcription


class WhisperProvider:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                import whisper
            except ImportError as error:
                raise RuntimeError(
                    "Whisper is not installed. Install the project dependencies before transcribing."
                ) from error
            except OSError as error:
                raise RuntimeError(
                    "PyTorch could not start on this Windows installation. Reinstall the "
                    "CPU-only Torch version listed in requirements.txt, then try again."
                ) from error
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio_path: Path, *, task: str = "transcribe") -> Transcription:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if task not in {"transcribe", "translate"}:
            raise ValueError("Task must be 'transcribe' or 'translate'.")

        try:
            audio = _load_wav_for_whisper(audio_path)
            result = self._get_model().transcribe(audio, task=task)
        except FileNotFoundError as error:
            raise RuntimeError(
                "VOCA could not read this WAV file. Confirm that the file still exists and try again."
            ) from error
        segments = [
            Segment(start=float(item["start"]), end=float(item["end"]), text=item["text"].strip())
            for item in result.get("segments", [])
        ]
        return Transcription(
            text=str(result.get("text", "")).strip(),
            language=str(result.get("language", "unknown")),
            segments=segments,
        )



def _load_wav_for_whisper(audio_path: Path) -> np.ndarray:
    """Read a WAV file into Whisper's expected 16 kHz mono float32 waveform."""
    try:
        sample_rate, samples = read(audio_path)
    except ValueError as error:
        raise RuntimeError("VOCA currently supports valid WAV input files only.") from error

    audio = np.asarray(samples)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            audio = (audio.astype(np.float32) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)

    whisper_sample_rate = 16000
    if sample_rate != whisper_sample_rate:
        divisor = np.gcd(sample_rate, whisper_sample_rate)
        audio = resample_poly(audio, whisper_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
    return audio
