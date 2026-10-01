"""Application settings loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    sample_rate: int = int(os.getenv("VOCA_SAMPLE_RATE", "16000"))
    maximum_recording_seconds: float = float(os.getenv("VOCA_MAX_RECORDING_SECONDS", "30"))
    minimum_speech_seconds: float = float(os.getenv("VOCA_MIN_SPEECH_SECONDS", "0.3"))
    silence_seconds: float = float(os.getenv("VOCA_SILENCE_SECONDS", "0.7"))
    vad_threshold_db: float = float(os.getenv("VOCA_VAD_THRESHOLD_DB", "-40"))
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "90"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    tts_enabled: bool = os.getenv("VOCA_TTS_ENABLED", "true").lower() in {"1", "true", "yes"}
    tts_voice_id: str | None = os.getenv("VOCA_TTS_VOICE_ID") or None
    tts_rate: int = int(os.getenv("VOCA_TTS_RATE", "220"))
    web_save_visualizations: bool = os.getenv("VOCA_WEB_SAVE_VISUALIZATIONS", "false").lower() in {"1", "true", "yes"}
    recordings_dir: Path = Path(os.getenv("VOCA_RECORDINGS_DIR", "recordings"))
    database_path: Path = Path(os.getenv("VOCA_DATABASE_PATH", "data/voca.db"))


settings = Settings()
