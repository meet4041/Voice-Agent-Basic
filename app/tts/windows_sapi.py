"""Offline Windows SAPI text-to-speech provider."""

from pathlib import Path

from app.tts.base import SpeechAudio, Voice


class TextToSpeechError(RuntimeError):
    """Raised when voice synthesis or playback cannot be completed."""


class WindowsSapiTTSProvider:
    """Use the voices installed in Windows without sending text to a cloud service."""

    @staticmethod
    def _engine():
        try:
            import pyttsx3
        except ImportError as error:
            raise TextToSpeechError(
                "The Windows text-to-speech package is missing. Run: pip install -r requirements.txt"
            ) from error
        try:
            return pyttsx3.init()
        except Exception as error:
            raise TextToSpeechError(
                "We couldn't start Windows text-to-speech. Check that a Windows voice is installed."
            ) from error

    def list_voices(self) -> list[Voice]:
        try:
            return [
                Voice(id=str(voice.id), name=str(voice.name))
                for voice in self._engine().getProperty("voices")
            ]
        except Exception as error:
            raise TextToSpeechError("We couldn't retrieve the installed Windows voices.") from error

    def synthesize(self, text: str, output_path: Path, *, voice_id: str | None, rate: int) -> SpeechAudio:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("VOCA can't speak an empty message.")
        if rate < 80 or rate > 400:
            raise ValueError("Speech rate must be between 80 and 400 words per minute.")
        if output_path.suffix.lower() != ".wav":
            raise ValueError("Windows SAPI output must use a .wav filename.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine = self._engine()
        try:
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", rate)
            engine.save_to_file(cleaned_text, str(output_path))
            engine.runAndWait()
        except Exception as error:
            raise TextToSpeechError("VOCA couldn't generate speech with the selected Windows voice.") from error

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise TextToSpeechError("Windows did not produce a usable speech audio file.")

        selected_voice = next((voice for voice in self.list_voices() if voice.id == voice_id), None)
        return SpeechAudio(path=output_path, voice=selected_voice, rate=rate)

    @staticmethod
    def play(audio_path: Path) -> None:
        """Play a WAV file through the system's default speakers."""
        if not audio_path.is_file():
            raise FileNotFoundError(f"Speech audio file not found: {audio_path}")
        try:
            import winsound

            winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
        except Exception as error:
            raise TextToSpeechError(
                "We couldn't play VOCA's response. Check that your speakers are available."
            ) from error
