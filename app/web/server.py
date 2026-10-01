"""FastAPI server exposing VOCA's private local workspace in a browser."""

from datetime import datetime
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio.analysis import create_visualizations
from app.audio.recorder import RecordingError, record_until_silence, save_wav
from app.config import settings
from app.conversations.store import ConversationStore
from app.web.export import export_conversation
from app.llm.ollama import LLMError
from app.llm.service import LLMService
from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.search.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.search.service import SemanticSearchService
from app.stt.whisper import WhisperProvider
from app.tts.service import TextToSpeechService
from app.tts.windows_sapi import TextToSpeechError, WindowsSapiTTSProvider


WEB_DIRECTORY = Path(__file__).parent / "static"
app = FastAPI(title="VOCA Local Workspace", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIRECTORY), name="static")

# These expensive local models stay in memory between browser requests.
_whisper = WhisperProvider(settings.whisper_model)
_llm = LLMService(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    timeout_seconds=settings.ollama_timeout_seconds,
    max_tokens=settings.ollama_max_tokens,
    keep_alive=settings.ollama_keep_alive,
)


def _warm_local_models() -> None:
    """Prepare local models after the page is available, without blocking it."""
    try:
        _whisper._get_model()
    except RuntimeError:
        pass
    _llm.provider.warm()


@app.on_event("startup")
def warm_local_models() -> None:
    Thread(target=_warm_local_models, daemon=True).start()


class MemoryCreate(BaseModel):
    content: str


class MemorySetting(BaseModel):
    enabled: bool


class ConversationRename(BaseModel):
    title: str


class TextTurn(BaseModel):
    text: str


def _stores() -> tuple[ConversationStore, MemoryStore]:
    return ConversationStore(settings.database_path), MemoryStore(settings.database_path)


def _recording_directory() -> Path:
    return settings.recordings_dir / datetime.now().strftime("%Y%m%d-%H%M%S")


def _create_and_play_speech(response: str, message_id: str, output_directory: Path | None = None) -> str | None:
    """Save and play a response without delaying the browser for playback."""
    if not settings.tts_enabled:
        return None
    try:
        speech = TextToSpeechService().create_speech(
            response,
            (output_directory or _recording_directory()) / "voca_response.wav",
            voice_id=settings.tts_voice_id,
            rate=settings.tts_rate,
        )
        Thread(target=WindowsSapiTTSProvider.play, args=(speech.path,), daemon=True).start()
        conversations, _ = _stores()
        conversations.update_message_audio(message_id, str(speech.path))
        return str(speech.path)
    except (TextToSpeechError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/")
def workspace() -> FileResponse:
    return FileResponse(WEB_DIRECTORY / "index.html")


@app.get("/api/conversations")
def list_conversations() -> list[dict]:
    conversations, _ = _stores()
    return [conversation.__dict__ for conversation in conversations.list_conversations()]


@app.get("/api/conversations/active")
def active_conversation() -> dict:
    conversations, _ = _stores()
    conversation = conversations.get_active_conversation()
    return {
        "conversation": conversation.__dict__,
        "messages": [message.__dict__ for message in conversations.messages(conversation.id)],
    }


@app.post("/api/conversations")
def new_conversation() -> dict:
    conversations, _ = _stores()
    return conversations.create_conversation().__dict__


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict:
    conversations, _ = _stores()
    conversation = conversations.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "conversation": conversation.__dict__,
        "messages": [message.__dict__ for message in conversations.messages(conversation_id)],
    }


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: ConversationRename) -> dict:
    conversations, _ = _stores()
    conversation = conversations.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        conversations.update_title(conversation_id, payload.title[:80])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    renamed = conversations.get_conversation(conversation_id)
    return renamed.__dict__ if renamed else {}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    conversations, _ = _stores()
    if not conversations.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": True}


@app.get("/api/conversations/{conversation_id}/export")
def export_saved_conversation(conversation_id: str, format: str = "txt") -> Response:
    conversations, _ = _stores()
    conversation = conversations.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        content, media_type, filename = export_conversation(
            conversation, conversations.messages(conversation_id), format
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/memories")
def list_memories() -> dict:
    _, memories = _stores()
    return {"enabled": memories.is_enabled(), "memories": [memory.__dict__ for memory in memories.list()]}


@app.post("/api/memories")
def add_memory(payload: MemoryCreate) -> dict:
    _, memories = _stores()
    try:
        return memories.add(payload.content).__dict__
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict:
    _, memories = _stores()
    if not memories.delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"deleted": True}


@app.post("/api/memories/enabled")
def set_memory_enabled(payload: MemorySetting) -> dict:
    _, memories = _stores()
    memories.set_enabled(payload.enabled)
    return {"enabled": payload.enabled}


@app.get("/api/search")
def search(query: str) -> dict:
    conversations, memories = _stores()
    service = SemanticSearchService(
        settings.database_path,
        OllamaEmbeddingProvider(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        ),
    )
    try:
        indexed = service.refresh(conversations, memories)
        results = service.search(query)
    except (EmbeddingError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"indexed": indexed, "results": [result.__dict__ for result in results]}


@app.post("/api/text-turn")
def text_turn(payload: TextTurn) -> dict:
    """Run a typed message through the same private history, LLM, and speech flow."""
    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Type a message before sending it.")
    conversations, memories = _stores()
    conversation = conversations.get_active_conversation()
    if conversation.title == "New conversation":
        conversations.update_title(conversation.id, user_text[:60])
    conversations.add_message(conversation.id, "user", user_text)
    MemoryService(memories).consider_user_message(user_text)
    history = [
        {"role": message.role, "content": message.content}
        for message in conversations.messages(conversation.id, limit=12)
    ]
    try:
        response = _llm.respond(
            user_text,
            history=history,
            memories=[memory.content for memory in memories.list(limit=12)] if memories.is_enabled() else [],
        )
    except (LLMError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    assistant_message = conversations.add_message(conversation.id, "assistant", response)
    speech_path = _create_and_play_speech(response, assistant_message.id)
    return {"response": response, "audio_path": speech_path}


@app.post("/api/messages/{message_id}/play")
def replay_message(message_id: str) -> dict:
    conversations, _ = _stores()
    message = conversations.get_message(message_id)
    if message is None or not message.audio_path:
        raise HTTPException(status_code=404, detail="No saved audio is available for this response.")
    audio_path = Path(message.audio_path)
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="The saved audio file is no longer available.")
    Thread(target=WindowsSapiTTSProvider.play, args=(audio_path,), daemon=True).start()
    return {"playing": True}


@app.post("/api/voice-turn")
def voice_turn() -> dict:
    """Run the existing microphone → STT → local LLM → TTS pipeline."""
    conversations, memories = _stores()
    memory_service = MemoryService(memories)
    conversation = conversations.get_active_conversation()
    try:
        recording = record_until_silence(
            sample_rate=settings.sample_rate,
            maximum_duration_seconds=settings.maximum_recording_seconds,
            minimum_speech_seconds=settings.minimum_speech_seconds,
            silence_seconds=settings.silence_seconds,
            threshold_db=settings.vad_threshold_db,
        )
        output_directory = _recording_directory()
        wav_path = save_wav(recording, output_directory / "input.wav")
        artifacts = None
        if settings.web_save_visualizations:
            artifacts = create_visualizations(recording.samples, recording.sample_rate, output_directory)
        transcription = _whisper.transcribe(wav_path)
    except (RecordingError, RuntimeError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if not transcription.text:
        return {"transcript": "", "language": transcription.language, "segments": [], "response": None}

    if conversation.title == "New conversation":
        conversations.update_title(conversation.id, transcription.text[:60])
    conversations.add_message(conversation.id, "user", transcription.text)
    memory_service.consider_user_message(transcription.text)
    history = [
        {"role": message.role, "content": message.content}
        for message in conversations.messages(conversation.id, limit=12)
    ]
    try:
        response = _llm.respond(
            transcription.text,
            history=history,
            memories=[memory.content for memory in memories.list(limit=12)] if memories.is_enabled() else [],
        )
    except (LLMError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    assistant_message = conversations.add_message(conversation.id, "assistant", response)
    speech_path = _create_and_play_speech(response, assistant_message.id, output_directory)

    return {
        "transcript": transcription.text,
        "language": transcription.language,
        "segments": [segment.__dict__ for segment in transcription.segments],
        "response": response,
        "audio_path": speech_path,
        "artifacts": [str(path) for path in vars(artifacts).values()] if artifacts else [],
    }
