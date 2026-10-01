# VOCA — Speak. Think. Remember.

VOCA is a voice-first AI workspace being built incrementally. This repository implements Milestones 1–8: microphone capture, audio visualizations, local Whisper transcription, voice activity detection, offline Windows text-to-speech, a free local Ollama LLM, conversation history, selective long-term memory, and semantic search.

## Included now

- Mono microphone recording at 16 kHz
- WAV output in a timestamped `recordings/` folder
- Waveform, frequency spectrum, spectrogram, and 80-band mel-spectrogram PNGs
- Whisper transcription with detected language and timestamped segments
- Energy-based voice activity detection: starts saving audio on speech and stops after silence
- Offline Windows SAPI text-to-speech with configurable voice and rate
- A saved `voca_response.wav` file and automatic speaker playback after transcription
- A free local Ollama LLM response; VOCA sends no transcript to a cloud LLM
- SQLite conversation history stored locally in `data/voca.db`
- Selective local memory with view, add, delete, and disable controls
- Semantic search across local conversations and memories
- Modular audio and STT boundaries ready for later VAD, TTS, LLM, memory, and realtime work

## Prerequisites

- Python 3.10, 3.11, or 3.12 is supported. Milestones 1–2 transcribe WAV files directly, so no separate FFmpeg installation is needed.
- A working microphone with permission enabled for your terminal or editor.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

`openai-whisper` downloads the selected model automatically on its first transcription. The `base` model is the default; set `WHISPER_MODEL=tiny` in your environment for a smaller, quicker learning setup.

## Free local LLM setup

VOCA uses [Ollama](https://ollama.com/) locally. Install Ollama for Windows, then download the default free model:

```powershell
ollama pull llama3.2:1b
```

This is a one-time 1.3 GB model download. The model runs on your computer; no API key, subscription, or per-message charge is involved. `llama3.2:1b` is the default because it is a good fit for a typical personal computer. If your computer has more available RAM and you want higher-quality answers, use `llama3.2:3b` and set `OLLAMA_MODEL=llama3.2:3b`.

If a prior Torch installation reports a `c10.dll` / `WinError 1114` error, reinstall the pinned CPU build:

```powershell
pip install --force-reinstall --no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

## Run

```powershell
python -m app
```

Press Enter, then speak naturally. VOCA waits for speech and stops after one second of silence (or after 30 seconds at most). Results are saved in `recordings/<timestamp>/`.

After transcription, VOCA says a short confirmation aloud. To disable speaker playback while retaining recording and transcription, set `VOCA_TTS_ENABLED=false` before running.

VOCA now asks the local LLM for an answer and speaks that answer instead of repeating the transcript.

## Local web interface

Run VOCA's browser interface on this computer:

```powershell
python -m app.web
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The voice button uses the same local microphone, Ollama, history, memory, and speaker pipeline as the command-line app. For speed, the dashboard keeps Whisper and the local model warm in memory, skips visualisation image generation by default, returns while speech is playing, and asks for concise replies. Keep the PowerShell window open while using the dashboard; press `Ctrl+C` there to stop it.

## Conversation history

VOCA keeps the active conversation locally and sends its latest messages to the local model as context. No conversation data is uploaded.

```powershell
# Continue the current conversation
python -m app

# Start a fresh conversation
python -m app --new-conversation

# See recent conversation IDs and titles
python -m app --list-conversations

# Show one conversation's saved messages
python -m app --show-conversation <conversation-id>
```

## Long-term memory

Memory is local, optional, and user-controlled. VOCA saves only direct statements that begin with clear project, goal, or preference phrases—for example, “I am building…”, “My goal is…”, or “I prefer…”. It stores your exact statement, not an invented summary.

```powershell
# View stored memories
python -m app --list-memories

# Add a memory yourself
python -m app --add-memory "I prefer concise responses."

# Stop VOCA from automatically saving new memories
python -m app --memory off

# Re-enable automatic memory saving
python -m app --memory on

# Delete one memory after copying its ID from --list-memories
python -m app --delete-memory <memory-id>
```

## Semantic search

Semantic search finds related meaning, not merely matching words. It uses the free local Ollama `nomic-embed-text` embedding model. Download it once:

```powershell
ollama pull nomic-embed-text
```

Then search all stored conversations and memories:

```powershell
python -m app --search "cricket tracking project"
```

The first search indexes existing content. Later searches only embed new or changed items.

### Adjust voice detection

Set these environment variables before running if your room is especially quiet or noisy:

- `VOCA_VAD_THRESHOLD_DB` — default `-40`; use `-35` to require louder speech, or `-45` for quieter speech.
- `VOCA_SILENCE_SECONDS` — default `1.0`, the silence required to finish recording.
- `VOCA_MAX_RECORDING_SECONDS` — default `30`, the safety limit.

### Adjust speech output

- `VOCA_TTS_RATE` — default `180`; accepted values are 80–400.
- `VOCA_TTS_VOICE_ID` — optional Windows voice identifier. Leave blank to use the default installed voice.
- `VOCA_TTS_ENABLED` — default `true`; set to `false` to suppress speech playback.

### Adjust the local LLM

- `OLLAMA_MODEL` — default `llama3.2:1b`; use `llama3.2:3b` for stronger responses on capable hardware.
- `OLLAMA_TIMEOUT_SECONDS` — default `120`; increase it if a CPU-only response takes longer.

## Tests

```powershell
pytest
```

The tests do not access your microphone or download a Whisper model.

## Project layout

```text
app/
  audio/       microphone capture and signal visualizations
  web/         private local browser dashboard and API
  stt/         provider contract and Whisper implementation
  cli.py       first VOCA user experience
tests/         offline checks for audio/STT behavior
```

## Next milestone

Milestone 9 adds meeting mode: extended recording, transcript, summaries, and action items.
