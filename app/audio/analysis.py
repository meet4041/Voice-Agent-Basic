"""Generate visual learning artifacts from recorded audio."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
from scipy.fft import rfft, rfftfreq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class AnalysisArtifacts:
    waveform: Path
    spectrum: Path
    spectrogram: Path
    mel_spectrogram: Path


def create_visualizations(samples: np.ndarray, sample_rate: int, output_dir: Path) -> AnalysisArtifacts:
    """Save waveform, FFT spectrum, STFT spectrogram, and mel spectrogram PNGs."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot analyze an empty recording.")

    output_dir.mkdir(parents=True, exist_ok=True)
    waveform = output_dir / "waveform.png"
    spectrum = output_dir / "spectrum.png"
    spectrogram = output_dir / "spectrogram.png"
    mel_spectrogram = output_dir / "mel_spectrogram.png"

    time_axis = np.arange(audio.size) / sample_rate
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(time_axis, audio, linewidth=0.6)
    axis.set(title="VOCA Waveform", xlabel="Time (seconds)", ylabel="Amplitude")
    figure.tight_layout()
    figure.savefig(waveform, dpi=150)
    plt.close(figure)

    frequencies = rfftfreq(audio.size, 1 / sample_rate)
    magnitudes = np.abs(rfft(audio))
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(frequencies, magnitudes, linewidth=0.6)
    axis.set(title="VOCA Frequency Spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude")
    axis.set_xlim(0, min(8000, sample_rate / 2))
    figure.tight_layout()
    figure.savefig(spectrum, dpi=150)
    plt.close(figure)

    stft = librosa.stft(audio)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    figure, axis = plt.subplots(figsize=(12, 5))
    image = librosa.display.specshow(stft_db, sr=sample_rate, x_axis="time", y_axis="hz", ax=axis)
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    axis.set_title("VOCA Spectrogram")
    figure.tight_layout()
    figure.savefig(spectrogram, dpi=150)
    plt.close(figure)

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    figure, axis = plt.subplots(figsize=(12, 5))
    image = librosa.display.specshow(mel_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=axis)
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    axis.set_title("VOCA Mel Spectrogram")
    figure.tight_layout()
    figure.savefig(mel_spectrogram, dpi=150)
    plt.close(figure)

    return AnalysisArtifacts(waveform, spectrum, spectrogram, mel_spectrogram)
