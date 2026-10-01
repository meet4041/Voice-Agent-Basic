"""The first interactive VOCA command-line workspace."""

from datetime import datetime
import argparse
from pathlib import Path

from app.audio.analysis import create_visualizations
from app.audio.recorder import RecordingError, record_until_silence, save_wav
from app.config import settings
from app.conversations.store import ConversationStore
from app.llm.ollama import LLMError
from app.llm.service import LLMService
from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.search.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.search.service import SemanticSearchService
from app.stt.whisper import WhisperProvider
from app.tts.service import TextToSpeechService
from app.tts.windows_sapi import TextToSpeechError, WindowsSapiTTSProvider


def _new_recording_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return settings.recordings_dir / timestamp


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOCA local voice workspace")
    parser.add_argument("--new-conversation", action="store_true", help="Start a new local conversation.")
    parser.add_argument("--list-conversations", action="store_true", help="List saved local conversations.")
    parser.add_argument("--show-conversation", metavar="ID", help="Show messages from a saved conversation.")
    parser.add_argument("--list-memories", action="store_true", help="List local long-term memories.")
    parser.add_argument("--add-memory", metavar="TEXT", help="Add a local memory manually.")
    parser.add_argument("--delete-memory", metavar="ID", help="Delete one local memory.")
    parser.add_argument("--memory", choices=("on", "off"), help="Enable or disable automatic memory saving.")
    parser.add_argument("--search", metavar="QUERY", help="Search chats and memories by meaning.")
    return parser.parse_args()


def _print_conversation_messages(store: ConversationStore, conversation_id: str) -> None:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        print("Conversation not found.")
        return
    print(f"\n{conversation.title} ({conversation.id})")
    for message in store.messages(conversation.id):
        print(f"\n{message.role.title()}: {message.content}")


def main() -> None:
    arguments = _parse_arguments()
    store = ConversationStore(settings.database_path)
    memory_store = MemoryStore(settings.database_path)
    memory_service = MemoryService(memory_store)
    if arguments.list_conversations:
        for conversation in store.list_conversations():
            print(f"{conversation.id}  {conversation.title}  {conversation.updated_at}")
        return
    if arguments.show_conversation:
        _print_conversation_messages(store, arguments.show_conversation)
        return
    if arguments.list_memories:
        print(f"Memory is {'on' if memory_store.is_enabled() else 'off'}.")
        for memory in memory_store.list():
            print(f"{memory.id}  [{memory.category}] {memory.content}")
        return
    if arguments.add_memory:
        memory = memory_store.add(arguments.add_memory)
        print(f"Memory saved: {memory.id}")
        return
    if arguments.delete_memory:
        print("Memory deleted." if memory_store.delete(arguments.delete_memory) else "Memory not found.")
        return
    if arguments.memory:
        memory_store.set_enabled(arguments.memory == "on")
        print(f"Memory is now {arguments.memory}.")
        return
    if arguments.search:
        search_service = SemanticSearchService(
            settings.database_path,
            OllamaEmbeddingProvider(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.ollama_timeout_seconds,
            ),
        )
        try:
            indexed = search_service.refresh(store, memory_store)
            results = search_service.search(arguments.search)
        except (EmbeddingError, ValueError) as error:
            print(f"Search unavailable: {error}")
            return
        print(f"Indexed {indexed} new or changed item(s).")
        if not results:
            print("No matching conversations or memories found.")
            return
        for result in results:
            print(f"\n[{result.source_type} | score: {result.score:.2f}] {result.title}")
            print(result.content)
        return
    if arguments.new_conversation:
        conversation = store.create_conversation()
        print(f"Started new conversation: {conversation.id}")

    print("=" * 40)
    print("             VOCA")
    print("=" * 40)
    print(f"\nWhisper model: {settings.whisper_model}")
    conversation = store.get_active_conversation()
    print(f"Conversation: {conversation.title} ({conversation.id[:8]})")
    input("\nPress ENTER to start recording. ")

    try:
        print("\nListening... Speak now. VOCA will stop after you finish speaking.")
        recording = record_until_silence(
            sample_rate=settings.sample_rate,
            maximum_duration_seconds=settings.maximum_recording_seconds,
            minimum_speech_seconds=settings.minimum_speech_seconds,
            silence_seconds=settings.silence_seconds,
            threshold_db=settings.vad_threshold_db,
        )
        output_dir = _new_recording_directory()
        wav_path = save_wav(recording, output_dir / "input.wav")
        artifacts = create_visualizations(recording.samples, recording.sample_rate, output_dir)
    except (RecordingError, ValueError) as error:
        print(f"\nError: {error}")
        return

    print("Recording complete.")
    print(f"Audio and visualizations saved in: {output_dir.resolve()}")
    print("\nTranscribing...")
    try:
        transcription = WhisperProvider(settings.whisper_model).transcribe(wav_path)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\nTranscription failed: {error}")
        return

    print("\n" + "-" * 40)
    print("TRANSCRIPTION")
    print("-" * 40)
    print(f"\n{transcription.text or '(No speech detected.)'}")
    print(f"\nLanguage: {transcription.language}")
    print("\nSegments:")
    for segment in transcription.segments:
        print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")

    if transcription.text:
        if conversation.title == "New conversation":
            store.update_title(conversation.id, transcription.text[:60])
        store.add_message(conversation.id, "user", transcription.text)
        new_memory = memory_service.consider_user_message(transcription.text)
        if new_memory is not None:
            print(f"Memory saved: [{new_memory.category}] {new_memory.content}")
        history = [
            {"role": message.role, "content": message.content}
            for message in store.messages(conversation.id, limit=12)
        ]
        print("\nVOCA is thinking...")
        try:
            if settings.llm_provider != "ollama":
                raise LLMError("This version of VOCA supports the free local Ollama provider only.")
            response_text = LLMService(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.ollama_timeout_seconds,
            ).respond(
                transcription.text,
                history=history,
                memories=[memory.content for memory in memory_store.list(limit=12)] if memory_store.is_enabled() else [],
            )
        except (LLMError, ValueError) as error:
            print(f"VOCA couldn't answer: {error}")
            return

        print("\n" + "-" * 40)
        print("VOCA")
        print("-" * 40)
        print(f"\n{response_text}")
        assistant_message = store.add_message(conversation.id, "assistant", response_text)

    if settings.tts_enabled and transcription.text:
        print("\nVOCA is speaking...")
        try:
            speech = TextToSpeechService().create_speech(
                response_text,
                output_dir / "voca_response.wav",
                voice_id=settings.tts_voice_id,
                rate=settings.tts_rate,
            )
            WindowsSapiTTSProvider.play(speech.path)
            store.update_message_audio(assistant_message.id, str(speech.path))
            print(f"Speech saved as: {speech.path.name}")
        except (FileNotFoundError, TextToSpeechError, ValueError) as error:
            print(f"VOCA couldn't speak: {error}")
    print("\nAnalysis files:")
    for path in (artifacts.waveform, artifacts.spectrum, artifacts.spectrogram, artifacts.mel_spectrogram):
        print(f"- {path.name}")
